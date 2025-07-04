import time
import torch
import pickle
import logging
from tqdm import tqdm
import numpy as np

from dataUtils import data

logger = logging.getLogger(__name__) 

class TrainLoop:
    def __init__(
        self, 
        model, 
        optimizer, 
        criterion, 
        scheduler,
        train_loader, 
        valid_loader,
        test_loader, 
        eval_fn,
        evalCalc_fn,
        save_path,
        device, 
        num_epochs,  
        save_every
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.eval_fn = eval_fn
        self.evalCalc_fn = evalCalc_fn
        self.save_path = save_path
        self.save_every = save_every

        self.train_loss, self.valid_loss, self.test_loss = [], [], []
        self.train_eval, self.valid_eval, self.test_eval = [], [], []
        self.jaccard_train, self.jaccard_valid, self.jaccard_test = [], [], []

    def run(self):
        start_time = time.time()
        logger.info("Start training...")

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} | Progress: {epoch / self.num_epochs:.2%} | Time: {self._time_since(start_time)}")

            train_metrics = self._run_epoch(
                self.train_loader, 
                is_train=True
            )
            self.train_loss.append(train_metrics['loss'])
            self.train_eval.append(train_metrics['metrics'])
            self.jaccard_train.append(train_metrics['jaccard'])

            if epoch % self.save_every == 0:
                val_metrics = self._run_epoch(
                    self.valid_loader, 
                    is_train=False, 
                    mode="validation"
                )
                self.valid_loss.append(val_metrics['loss'])
                self.valid_eval.append(val_metrics['metrics'])
                self.jaccard_valid.append(val_metrics['jaccard'])
                for k, v in val_metrics['metrics'].items():
                    logger.info(f"Validation {k}: {v}")

            self.scheduler_step()
        logger.info("Training complete.")

        logger.info("Start testing...")
        test_metrics = self._run_epoch(
            self.test_loader, 
            is_train=False, 
            mode="testing"
        )
        self.test_loss.append(test_metrics['loss'])
        self.test_eval.append(test_metrics['metrics'])
        self.jaccard_test.append(test_metrics['jaccard'])
        self.save_model()
        logger.info("Testing complete.")

    def _run_epoch(
        self, 
        loader, 
        is_train=True,
        mode = "training"
    ):
        self.model.train() if is_train else self.model.eval()

        total_loss = 0
        macro = torch.zeros((3, data.vocab_size_zh))
        tp = fp = fn = 0
        jaccard = set()

        with torch.set_grad_enabled(is_train):
            loop = tqdm(loader, leave=False)
            for en_index, zh_index in loop:
                en_index, zh_index = en_index.to(self.device), zh_index.to(self.device)
                outputs = self.model(en_index)
                loss = self.criterion(outputs, zh_index.float())
                total_loss += loss.item()

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                batch_tp, batch_fp, batch_fn, batch_macro, indices = self.eval_fn(outputs.sigmoid().cpu(), zh_index.cpu())
                tp += batch_tp
                fp += batch_fp
                fn += batch_fn
                macro += batch_macro
                jaccard |= indices

                loop.set_description(f"{mode.capitalize()} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        metrics = self.evalCalc_fn(tp, fp, fn, macro, jaccard, mode)
        logger.info(f"{mode.capitalize()} Loss: {avg_loss:.4f}")
        return {"loss": avg_loss, "metrics": metrics, "jaccard": jaccard}

    def scheduler_step(self):
        self.scheduler.step()  
        for group in self.optimizer.param_groups:
            logger.info(f"Learning Rate: {group['lr']:.6f}")


    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'test_loss': self.test_loss,
            'train_eval': self.train_eval,
            'valid_eval': self.valid_eval,
            'test_eval': self.test_eval,
            'jaccard_train': self.jaccard_train,
            'jaccard_valid': self.jaccard_valid,
            'jaccard_test': self.jaccard_test
        }, self.save_path)

    def _time_since(self, start):
        elapsed = time.time() - start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m {secs}s"

class TranslateLoop:
    def __init__(
        self, 
        model, 
        test_loader, 
        hist_file,
        device,
        num_trans,
        en_word
    ):
        self.model  = model
        self.device = device
        self.en_word = en_word
        self.num_trans   = num_trans
        self.test_loader = test_loader
        self.history_file = hist_file

    def run(self):
        print("Begin translating...", file=open(self.history_file, 'a'))
        self.model.eval()
        with torch.no_grad():
            if self.en_word == None:
                loop = tqdm(enumerate(self.test_loader), total=len(self.test_loader), leave=False)
                for batch_idx, (en_index, _) in loop:
                    self._translate_step(en_index)
                    loop.set_description(f"Translation [{batch_idx}/{len(self.test_loader)}]")
            else:
                try:
                    en_index  = torch.tensor(data.word_to_index_en[self.en_word])
                    self._translate_step(en_index)
                except:
                    message = "The English word does not exist in golden_set.json. Please double check the file and choose another one."
                    logger.error(message, stack_info=True, exc_info=True)

        print("Translation complete.", file=open(self.history_file, 'a'))

    def _translate_step(self, en_index):
        en_index = en_index.to(self.device)

        output = self.model(en_index).sigmoid().detach().cpu()
        en_word = data.vocab_en[en_index.cpu()]
        correct_trans = data.golden_set[en_word]
        if self.num_trans == "all":
            pred  = (output[0]>=0.5).int().numpy()
            idx   = list(np.where(pred==1)[0])
        else:
            num = len(correct_trans) if self.num_trans == "default" else int(self.num_trans)
            idx = torch.topk(output[0], num)[1].tolist()
       
        pred_words = [data.vocab_zh[i] for i in idx]

        print(f"{en_word}\n  → Model Translation: {', '.join(pred_words)}\n  → Correct Translation: {', '.join(correct_trans)}", file=open(self.history_file, 'a'))
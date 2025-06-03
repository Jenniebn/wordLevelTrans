import time
import torch
import pickle
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__) 

class TrainLoop:
    def __init__(self, 
                 model, 
                 optimizer, 
                 criterion, 
                 scheduler,
                 train_loader, 
                 valid_loader,
                 device, 
                 num_epochs, 
                 vocab_size_zh, 
                 eval_fn,
                 evalCalc_fn,
                 save_path, 
                 save_every=5):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.num_epochs = num_epochs
        self.vocab_size_zh = vocab_size_zh
        self.eval_fn = eval_fn
        self.evalCalc_fn = evalCalc_fn
        self.save_path = save_path
        self.save_every = save_every

        self.train_loss, self.valid_loss = [], []
        self.train_eval, self.valid_eval = [], []
        self.jaccard_train, self.jaccard_valid = [], []

    def run(self):
        start_time = time.time()
        logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch} | Progress: {epoch / self.num_epochs:.2%} | Time: {self._time_since(start_time)}")

            train_metrics = self._run_epoch(self.train_loader, is_train=True)
            self.train_loss.append(train_metrics['loss'])
            self.train_eval.append(train_metrics['metrics'])
            self.jaccard_train.append(train_metrics['jaccard'])

            if epoch % self.save_every == 0:
                val_metrics = self._run_epoch(self.valid_loader, is_train=False)
                self.valid_loss.append(val_metrics['loss'])
                self.valid_eval.append(val_metrics['metrics'])
                self.jaccard_valid.append(val_metrics['jaccard'])
                for k, v in val_metrics['metrics'].items():
                    logger.info(f"Validation {k}: {v}")

            self.scheduler_step()
            self.save_model(epoch)
        logger.info("Training complete.")

    def _run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        mode = "training" if is_train else "validation"

        total_loss = 0
        macro = torch.zeros((3, self.vocab_size_zh))
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


    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'train_eval': self.train_eval,
            'valid_eval': self.valid_eval,
            'jaccard_train': self.jaccard_train,
            'jaccard_valid': self.jaccard_valid,
        }, self.save_path)

    def _time_since(self, start):
        elapsed = time.time() - start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m {secs}s"

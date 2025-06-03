import os
import time
import yaml
import logging
import argparse

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import *
from relrep import *
from dataUtils import *
from evalUtils import *
from modelUtils import *
from runUtils import TrainLoop

if torch.cuda.is_available():
    DEVICE = "cuda" 
elif torch.backends.mps.is_available():
    DEVICE = torch.DEVICE("mps")
else:
    DEVICE = "cpu"

torch.manual_seed(4)
np.random.seed(4)
random.seed(4)

HISTORY_PATH = os.path.join("output", (time.strftime("%Y%m%d-%H%M%S")))
os.makedirs(HISTORY_PATH, exist_ok=True)

log_path = os.path.join(HISTORY_PATH, "experiment.log")
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def main(conf):
    logger.info("Experiment started")
    logger.info("Creating relative latent space")
    en_rel_latent_space, zh_rel_latent_space = create_latent_space()

    logger.info("Creating models")
    pos_weight = np.where(data.pos_weight != 1, data.pos_weight * conf["pos_weight"], data.pos_weight)
    
    zhzh_model, enzh_model = create_models(zh_rel_latent_space, en_rel_latent_space, conf["zhzh_model_path"])
    zhzh_model, enzh_model = zhzh_model.to(DEVICE), enzh_model.to(DEVICE)

    optimizer = torch.optim.Adam(enzh_model.parameters(), lr=conf["base_lr"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    scheduler = CosineAnnealingLR(optimizer, T_max=conf["num_epochs"], eta_min=conf["final_lr"])

    logger.info("Creating data loader")
    train_loader, valid_loader = load_data(conf["prefix"], conf["batch_size"])

    logger.info("Training...")
    trainer = TrainLoop(
        enzh_model, 
        optimizer, 
        criterion, 
        scheduler,
        train_loader, 
        valid_loader,
        DEVICE, 
        conf["num_epochs"], 
        data.vocab_size_zh,
        eval, 
        evalCalc,
        HISTORY_PATH
    )

    trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    args.update(yamlread(args.get('conf_path')))
    main(args)
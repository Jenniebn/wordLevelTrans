import os
import time
import logging
import argparse

from relrep import *
from dataUtils import *
from evalUtils import *
from modelUtils import *
from runUtils import TestLoop

if torch.cuda.is_available():
    DEVICE = "cuda" 
elif torch.backends.mps.is_available():
    DEVICE = torch.DEVICE("mps")
else:
    DEVICE = "cpu"

torch.manual_seed(4)
np.random.seed(4)
random.seed(4)

def main(conf):

    HISTORY_PATH = os.path.join("output", (time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(HISTORY_PATH, exist_ok=True)

    log_path = os.path.join(HISTORY_PATH, "translation.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info("Translation started")
    logging.info("Creating relative latent space")
    en_rel_latent_space, zh_rel_latent_space = create_latent_space()

    logging.info("Creating models")
    pos_weight = np.where(data.pos_weight != 1, data.pos_weight * conf["pos_weight"], data.pos_weight)
    
    zhzh_model, enzh_model = create_models(zh_rel_latent_space, en_rel_latent_space, conf["zhzh_model_path"])
    zhzh_model, enzh_model = zhzh_model.to(DEVICE), enzh_model.to(DEVICE)

    optimizer = torch.optim.Adam(enzh_model.parameters(), lr=conf["lr"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    logging.info("Creating data loader")
    data = load_data(conf["prefix"], conf["batch_size"])

    logging.info("Testing...")
    TestLoop()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)
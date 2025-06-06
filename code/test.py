import os
import time
import yaml
import logging
import argparse

from utils import *
from relrep import *
from dataUtils import *
from evalUtils import *
from modelUtils import *
from runUtils import TestLoop

if torch.cuda.is_available():
    DEVICE = "cuda" 
else:
    DEVICE = "cpu"

torch.manual_seed(4)
np.random.seed(4)
random.seed(4)

HISTORY_PATH = os.path.join("output", (time.strftime("%Y%m%d-%H%M%S")))
os.makedirs(HISTORY_PATH, exist_ok=True)

log_path = os.path.join(HISTORY_PATH, "test.log")
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def main(conf):
    logger.info("Testing started")
    logger.info("Creating relative latent space")
    en_rel_latent_space, zh_rel_latent_space = create_latent_space()

    pos_weight = np.where(data.pos_weight != 1, data.pos_weight * conf["pos_weight"], data.pos_weight)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(conf["device"])
    
    logger.info("Creating models")
    zhzh_model, enzh_model = create_models(
        zh_rel_latent_space, 
        en_rel_latent_space, 
        **conf
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(conf["device"]))

    logger.info("Creating data loader")
    test_loader = load_data(conf["prefix"], conf["batch_size"])

    logger.info("Testing...")
    tester = TestLoop()

    tester.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())
    args.update(yamlread(args.get('conf_path')))
    args['device'] = DEVICE

    main(args)
import os
import sys
import time
import logging
import argparse

from utils import *
from relrep import *
from dataUtils import *
from evalUtils import *
from modelUtils import *
from runUtils import TranslateLoop

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
    assert conf["zhzh_model_path"] != None, "Must specify path to pretrained ZhZhAutoencoder"
    assert conf["enzh_model_path"] != None, "Must specify path to pretrained EnZhEncoderDecoder"

    logger.info("Testing started")
    logger.info("Creating relative latent space")
    en_rel_latent_space, zh_rel_latent_space = create_latent_space()

    pos_weight = np.where(data.pos_weight != 1, data.pos_weight * conf["pos_weight"], data.pos_weight)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(conf["device"])
    
    logger.info("Creating models")
    _, enzh_model = create_models(
        zh_rel_latent_space, 
        en_rel_latent_space, 
        conf["prefix"],
        conf["zhzh_model_path"], 
        conf["enzh_model_path"],
        conf["device"]
    )

    logger.info("Creating data loader")
    test_loader = load_data(
        conf["prefix"], 
        conf["batch_size"], 
    )

    trans_file = os.path.join(HISTORY_PATH, "translation.txt")
    logger.info(f"Translations output to {trans_file}")

    translater = TranslateLoop(
        enzh_model, 
        test_loader, 
        trans_file,
        conf["device"],
        conf["num_trans"],
        conf["en_word"]
    )

    translater.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument('--en_word', type=str, required=False, default=None)
    parser.add_argument('--help_config', action='store_true', help='Print help for config keys')
    args = vars(parser.parse_args())

    if args.get('help_config'):
        if args.get('conf_path') is None:
            print("Please provide --conf_path to use --help_config.")
            sys.exit(0)
        print_yaml_help(args['conf_path'])
        sys.exit(0)

    args.update(yamlread(args.get('conf_path')))
    args['device'] = DEVICE

    main(args)
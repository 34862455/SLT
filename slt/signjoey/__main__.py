import argparse #Handles command-line arguments.
import os #Manages environment variables (e.g., GPU selection).

import sys
# Imports training and testing functions from signjoey.training and signjoey.prediction
from signjoey.training import train
from signjoey.prediction import test

# sys.path.append("/vol/research/extol/personal/cihan/code/SignJoey")
# signjoey lies within slt, making it reachable from this path
sys.path.append("/home/minneke/Documents/Projects/SignExperiments/slt")


def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test"], help="train a model or test")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")

    ap.add_argument(
        "--output_path", type=str, help="path for saving translation output"
    )
    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    args = ap.parse_args()

    # ensures that training or testing runs on the specified GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        # Imported from signjoey.training
        train(cfg_file=args.config_path)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()

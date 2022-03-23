import argparse
from run_methods import run

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="TRAIN", type=str)  # TRAIN | PREDICT
parser.add_argument("--model", default="LSTM", type=str)  # DENSE | LSTM
parser.add_argument("--dataset", default="CIC", type=str)  # KDD | CIC | ES
parser.add_argument("--data_predict_path", default="", type=str)
args = parser.parse_args()


run(args.mode, args.model, args.dataset, args.data_predict_path)

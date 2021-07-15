import argparse
import sys
import json
from numpy.core.fromnumeric import trace
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from Models import ResNet,UNET
sys.path.append("/work/ariyanzarei/Charcoal_Dry_Rot")
from Preprocessing.dataset_generator import load_dataset

def get_args():
    parser = argparse.ArgumentParser(
        description='Charcoal Dryrot Classification/Segmentation training codes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e',
                        '--exp',
                        help='The name of the experiment',
                        metavar='exp',
                        required=True)

    return parser.parse_args()

def load_config(path):
    
    with open(path,"r") as f:
        config = json.load(f)

    return config
    

def main():
    args = get_args()
    config = load_config()
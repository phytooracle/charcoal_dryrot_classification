import argparse
import sys
import json
from numpy.core.fromnumeric import trace
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
from Models import ResNet,UNET
import socket
# sys.path.append("/work/ariyanzarei/Charcoal_Dry_Rot")
sys.path.append("/home/ariyan/projects/dry_rot/code/Charcoal_Dry_Rot")
from Preprocessing.dataset_generator import load_dataset
from tensorflow.keras.models import load_model
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(
        description='Charcoal Dryrot Classification/Segmentation training codes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e',
                        '--exp',
                        help='The name of the experiment',
                        metavar='exp',
                        required=False)

    return parser.parse_args()

def load_config(path):
    
    with open(path,"r") as f:
        config = json.load(f)

    return config['ariyan-Jetson']
    

def main():
    args = get_args()
    config = load_config('/home/ariyan/projects/dry_rot/code/Charcoal_Dry_Rot/config.json')
    
    # cnf = tf.ConfigProto()
    # cnf.gpu_options.allow_growth = True
    # cnf.gpu_options.per_process_gpu_memory_fraction = 0.4
    # session = tf.Session(config=cnf)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            print(gpus)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    x = np.load(os.path.join(config['datasets'],'x.npy'))
    y = np.load(os.path.join(config['datasets'],'y.npy'))

    model = load_model(os.path.join(config['models'],'ResNET_0.h5'))

    preds = model.predict(x[:1])

    first = datetime.now()
    first_time = first.strftime("%H:%M:%S")

    preds = model.predict(x[:285])

    second = datetime.now()
    second_time = second.strftime("%H:%M:%S")

    print((second-first))
    print(y[:285].shape)
    print(preds[:285].shape)

main()
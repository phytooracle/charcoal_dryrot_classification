import h5py
import cv2
import os 
import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(
        description='Visualizing images from the generated datasets of the dry rot images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d',
                        '--data',
                        help='The path to the directory in which the dataset files are located.',
                        metavar='data',
                        required=True)

    return parser.parse_args()

def load_datasets(path):
    with h5py.File(os.path.join(path,'segmentation_dataset.h5'),'r') as f:
        seg_dataset = {}
        for k in f.keys():
            seg_dataset[k] = f[k][:200]

    with h5py.File(os.path.join(path,'classification_dataset.h5'),'r') as f:
        cls_dataset = {}
        for k in f.keys():
            cls_dataset[k] = f[k][:200]

    return seg_dataset,cls_dataset

def visualize(image,mask,climage,label,i):
    
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image)
    axs[1].imshow(mask)
    axs[2].imshow(climage)
    fig.suptitle("present" if label else "absent")
    plt.savefig(f'fig{i}.png')

def main():
    args = get_args()
    seg,cls = load_datasets(args.data)
    
    for i in range(200):
        if cls['Y_train'][i]>0:
            visualize(seg['X_train'][i],seg['Y_train'][i],cls['X_train'][i],cls['Y_train'][i]>0,i)

if __name__ == "__main__":
    main()
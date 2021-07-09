import os
import random
import numpy as np
import multiprocessing
import math
import argparse
from skimage.io import imread
from skimage.transform import resize

def get_data_single(args):
    img_path = args[0]
    mask_path = args[1]
    img_name = args[2]
    width = args[3]
    height = args[4]
    channels = args[5]
    is_train = args[6]

    img = imread(img_path + '/' + img_name)[:,:,:channels]
    img = resize(img, (height, width), mode='constant', preserve_range=True)

    mask = imread(mask_path + '/' + img_name)[:,:,:channels]
    mask = resize(mask, (height, width), mode='constant', preserve_range=True)

    if np.max(mask) == 0:
        return img,0,is_train
    
    return img,1,is_train

def get_data_and_form_training_val_lists(training_img_path,training_mask_path,validation_img_path,validation_mask_path,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS,tr_n,val_n):

    training_images = os.listdir(training_img_path)
    training_masks = os.listdir(training_mask_path)

    validation_images = os.listdir(validation_img_path)
    validation_masks = os.listdir(validation_mask_path)

    train_n = min(len(training_images),tr_n) 
    validation_n = min(len(validation_images),val_n)

    X_train = np.zeros((train_n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros(train_n, dtype=np.uint8)

    X_val = np.zeros((validation_n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_val = np.zeros(validation_n, dtype=np.uint8)

    args = []

    for n,f in enumerate(training_images):
        if n>=train_n:
            break
        args.append((training_img_path,training_mask_path,f,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS,True))

    for n,f in enumerate(validation_images):
        if n>=validation_n:
            break
        args.append((validation_img_path,validation_mask_path,f,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS,False))

    processes = multiprocessing.Pool(int(math.floor(multiprocessing.cpu_count()*0.8)))
    results = processes.map(get_data_single,args)
    processes.close()

    t = 0
    v = 0

    for img,label,is_train in results:
        if img is None:
            continue

        if is_train:
            X_train[t] = img
            Y_train[t] = label
            t+=1
        else:
            X_val[v] = img
            Y_val[v] = label
            v+=1

    print('>>> Database generated successfully...')

    return X_train[:t],Y_train[:t],X_val[:v],Y_val[:v]

def save_train_val_data(X_train,Y_train,X_val,Y_val,path):

    np.save('{0}/X_train.npy'.format(path),X_train)
    np.save('{0}/Y_train.npy'.format(path),Y_train)
    np.save('{0}/X_val.npy'.format(path),X_val)
    np.save('{0}/Y_val.npy'.format(path),Y_val)

def load_train_val_data(path):

    X_train = np.load('{0}/X_train.npy'.format(path))
    Y_train = np.load('{0}/Y_train.npy'.format(path))
    X_val = np.load('{0}/X_val.npy'.format(path))
    Y_val = np.load('{0}/Y_val.npy'.format(path))

    return X_train,Y_train,X_val,Y_val

def get_args():
    
    parser = argparse.ArgumentParser(
        description='Generating dataset from the patches of the dry rot images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d',
                        '--data',
                        help='The path to the directory in which the dataset files will be created.',
                        metavar='data',
                        required=True)
    
    parser.add_argument('-p',
                        '--patches',
                        help='The path to all the dry rot patches.',
                        metavar='patches',
                        required=True)


    return parser.parse_args()

def main_data_gen():

    args = get_args()

    X_train,Y_train,X_val,Y_val = get_data_and_form_training_val_lists(\
        '{0}/training/images'.format(args.patches),\
        '{0}/training/annotation'.format(args.patches),\
        '{0}/validation/images'.format(args.patches),\
        '{0}/validation/annotation'.format(args.patches),\
        256,256,3,210000,70000)

    save_train_val_data(X_train,Y_train,X_val,Y_val,args.data)

main_data_gen()
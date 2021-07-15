import os
import random
import numpy as np
import sys
import multiprocessing
import math
import argparse
import h5py
from skimage.io import imread
from skimage.transform import resize

PROB_NON_DRYROT = 1

def get_data_single(args):
    img_path = args[0]
    mask_path = args[1]
    img_name = args[2]
    width = args[3]
    height = args[4]
    channels = args[5]

    img = imread(img_path + '/' + img_name)[:,:,:channels]
    img = resize(img, (height, width), mode='constant', preserve_range=True)

    mask = imread(mask_path + '/' + img_name)[:,:,:channels]
    mask = resize(mask, (height, width), mode='constant', preserve_range=True)

    if np.max(mask) == 0 and random.random()>PROB_NON_DRYROT:
        return None,None,None

    mask = np.expand_dims(np.max(mask,axis=-1),axis=-1)

    return img,mask

def get_data_and_form_training_val_lists(path,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS,tr_n=sys.maxsize,val_n=sys.maxsize):

    training_img_path = os.path.join(path,"training/images")
    training_mask_path = os.path.join(path,"training/annotation")
    validation_img_path = os.path.join(path,"validation/images")
    validation_mask_path = os.path.join(path,"validation/annotation")
    test_img_path = os.path.join(path,"test/images")
    test_mask_path = os.path.join(path,"test/annotation")

    training_images = os.listdir(training_img_path)
    training_masks = os.listdir(training_mask_path)
    validation_images = os.listdir(validation_img_path)
    validation_masks = os.listdir(validation_mask_path)
    test_images = os.listdir(test_img_path)
    test_masks = os.listdir(test_mask_path)

    train_n = min(len(training_images),tr_n) 
    validation_n = min(len(validation_images),val_n)
    test_n = min(len(test_images),val_n)

    X_train = np.zeros((train_n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((train_n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    X_val = np.zeros((validation_n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_val = np.zeros((validation_n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    X_test = np.zeros((test_n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_test = np.zeros((test_n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    # ----------------- training --------------------

    args = []

    for f in training_images:
        args.append((training_img_path,training_mask_path,f,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))

    print(">>> {0} training images to process.".format(len(args)))

    processes = multiprocessing.Pool(int(math.floor(multiprocessing.cpu_count()*0.8)))
    results = processes.map(get_data_single,args)
    processes.close()

    print(">>> {0} training images processed.".format(len(results)))

    ind = 0

    for img,mask in results:
        if img is None:
            continue
        
        X_train[ind] = img
        Y_train[ind] = mask
        ind+=1

    X_train = X_train[:ind]
    Y_train = Y_train[:ind]

    # ----------------- validation --------------------

    args = []

    for f in validation_images:    
        args.append((validation_img_path,validation_mask_path,f,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS,'val'))

    print(">>> {0} validation images to process.".format(len(args)))

    processes = multiprocessing.Pool(int(math.floor(multiprocessing.cpu_count()*0.8)))
    results = processes.map(get_data_single,args)
    processes.close()

    print(">>> {0} validation images processed.".format(len(results)))

    ind = 0

    for img,mask in results:
        if img is None:
            continue
        
        X_val[ind] = img
        Y_val[ind] = mask
        ind+=1

    X_val = X_val[:ind]
    Y_val = Y_val[:ind]

    # ----------------- test --------------------

    args = []

    for f in test_images:    
        args.append((test_img_path,test_mask_path,f,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS,'test'))

    print(">>> {0} test images to process.".format(len(args)))

    processes = multiprocessing.Pool(int(math.floor(multiprocessing.cpu_count()*0.8)))
    results = processes.map(get_data_single,args)
    processes.close()

    print(">>> {0} test images processed.".format(len(results)))

    ind = 0

    for img,mask in results:
        if img is None:
            continue
        
        X_test[ind] = img
        Y_test[ind] = mask
        ind+=1

    X_test = X_test[:ind]
    Y_test = Y_test[:ind]

    print('>>> Database generated successfully...')

    return X_train,Y_train,X_val,Y_val,X_test,Y_test

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

def generate_segmentation_dataset(path,X_train,Y_train,X_val,Y_val,X_test,Y_test):
    
    with h5py.File(path, "w") as f:
        f.create_dataset("X_train", data=X_train, dtype='uint8')
        f.create_dataset("Y_train", data=Y_train, dtype='uint8')
        f.create_dataset("X_val", data=X_val, dtype='uint8')
        f.create_dataset("Y_val", data=Y_val, dtype='uint8')
        f.create_dataset("X_test", data=X_test, dtype='uint8')
        f.create_dataset("Y_test", data=Y_test, dtype='uint8')

def generate_classification_dataset(path,X_train,Y_train,X_val,Y_val,X_test,Y_test):

    Y_train = np.reshape(Y_train,(Y_train.shape[0],Y_train.shape[1]*Y_train.shape[2]*Y_train.shape[3]))
    Y_train = np.max(Y_train,axis=1)

    Y_val = np.reshape(Y_val,(Y_val.shape[0],Y_val.shape[1]*Y_val.shape[2]*Y_val.shape[3]))
    Y_val = np.max(Y_val,axis=1)

    Y_test = np.reshape(Y_test,(Y_test.shape[0],Y_test.shape[1]*Y_test.shape[2]*Y_test.shape[3]))
    Y_test = np.max(Y_test,axis=1)


    with h5py.File(path, "w") as f:
        f.create_dataset("X_train", data=X_train, dtype='uint8')
        f.create_dataset("Y_train", data=Y_train, dtype='uint8')
        f.create_dataset("X_val", data=X_val, dtype='uint8')
        f.create_dataset("Y_val", data=Y_val, dtype='uint8')
        f.create_dataset("X_test", data=X_test, dtype='uint8')
        f.create_dataset("Y_test", data=Y_test, dtype='uint8')

def load_dataset(path,ds_type,count=None):
    if ds_type != "segmentation" and ds_type != "classification":
        print(":: Invalid dataset type. Enter either segmentation or classification.")
        return 

    with h5py.File(os.path.join(path,f'{ds_type}_dataset.h5'),'r') as f:
        dataset = {}
        for k in f.keys():
            if count is None:
                dataset[k] = f[k][:]
            else:
                dataset[k] = f[k][:count]

    return dataset

def main_data_gen():

    args = get_args()

    X_train,Y_train,X_val,Y_val,X_test,Y_test = get_data_and_form_training_val_lists(args.patches,256,256,3)

    generate_segmentation_dataset(os.path.join(args.data,"segmentation_dataset.h5"),X_train,Y_train,X_val,Y_val,X_test,Y_test)
    generate_classification_dataset(os.path.join(args.data,"classification_dataset.h5"),X_train,Y_train,X_val,Y_val,X_test,Y_test)

if __name__ == "__main__":
    main_data_gen()
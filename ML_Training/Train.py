import argparse
import sys
import json
from numpy.core.fromnumeric import trace
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from Models import ResNet,UNET,MobileNetV1
sys.path.append("/work/ariyanzarei/Charcoal_Dry_Rot")
from Preprocessing.dataset_generator import load_dataset
import socket

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

    return config[socket.gethostname()]
    
def create_and_train_model(experiment_name, dic):
    
    if experiment_name.split('_')[0] == 'UNET':
        model = UNET(dic['experiments'][experiment_name]).model
        data = load_dataset(dic['directories']['datasets'],'segmentation',1000)
    
    elif experiment_name.split('_')[0] == 'ResNET':
        model = ResNet(dic['experiments'][experiment_name]).model
        data = load_dataset(dic['directories']['datasets'],'classification',1000)

    elif experiment_name.split('_')[0] == 'MobileNETV1':
        model = MobileNetV1(dic['experiments'][experiment_name]).model
        data = load_dataset(dic['directories']['datasets'],'classification',1000)
    
    model.summary()

    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']

    if experiment_name.split('_')[0] == 'ResNET' or experiment_name.split('_')[0] == 'MobileNETV1':

        pos = np.sum(data['Y_train'])
        total = data['Y_train'].shape[0]
        neg = total-pos

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(class_weight)

        Y_train = tf.one_hot(Y_train,2)
        Y_val = tf.one_hot(Y_val,2)
    
    elif experiment_name.split('_')[0] == 'UNET':
        class_weight = None

        Y_train = Y_train.astype('float32')
        Y_val = Y_val.astype('float32')

    early_stopping_clbk = EarlyStopping(monitor='val_loss', min_delta=0, \
        patience=dic['experiments'][experiment_name]['patience'], verbose=0,mode='auto', baseline=None, restore_best_weights=True)
 
    history = model.fit(X_train, Y_train, epochs=dic['experiments'][experiment_name]['epochs'], batch_size=dic['experiments'][experiment_name]['batch_size'], 
                        validation_data=(X_val, Y_val), verbose=1,callbacks=[early_stopping_clbk],class_weight=class_weight)


    exp_dict = {'settings':dic['experiments'][experiment_name],'results':{}}
    exp_dict['results'] = history.history
    
    new_results = {}

    for key in exp_dict['results']:
        tmp = []
        for a in exp_dict['results'][key]:
            tmp.append(float(str(a)))
        new_results[key] = tmp

    exp_dict['results'] = new_results

    with open('{0}/results_{1}.json'.format(dic['directories']['results'],experiment_name), 'w') as file_pi:
        json.dump(exp_dict,file_pi)
    
    model.save('{}/{}.h5'.format(dic['directories']['models'], experiment_name))

    return model, history

def main():

    args = get_args()
    config = load_config("../config.json")
    
    dic = {}
    dic["directories"] = config

    experiment_name = args.exp

    with open("{0}/experiments_{1}.json".format(config['settings'],experiment_name.split('_')[0])) as f:
        dic["experiments"] = json.load(f)

    model, history = create_and_train_model(experiment_name, dic)

if __name__ == "__main__":
    main()
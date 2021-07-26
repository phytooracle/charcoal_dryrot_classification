import matplotlib.pyplot as plt
import numpy
import argparse
import json
import socket

def get_args():
    parser = argparse.ArgumentParser(
        description='Charcoal Dryrot Generate result plots.',
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

    host_name = socket.gethostname()
    
    if "hpc" in host_name:
        host_name = "hpc"

    return config[host_name]

def load_results(path):
    with open(path,'r') as f:
        results = json.load(f)

    return results

def main():

    args = get_args()
    config = load_config("../config.json")
    result = load_results('{0}/results_{1}.json'.format(config['results'],args.exp))

    plt.plot(result['results']['loss'],'r--',label='Training')
    plt.plot(result['results']['val_loss'],'b--',label='Validation')
    plt.legend()
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.title("Training and Validation Losses")
    plt.savefig('{0}/losses_{1}'.format(config['plots'],args.exp))
    plt.clf()
    plt.close()

    plt.plot(result['results']['accuracy'],'r--',label='Training')
    plt.plot(result['results']['val_accuracy'],'b--',label='Validation')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title("Training and Validation Accuracies")
    plt.savefig('{0}/accuracies_{1}'.format(config['plots'],args.exp))
    plt.clf()
    plt.close()

if __name__ == "__main__":
    main()


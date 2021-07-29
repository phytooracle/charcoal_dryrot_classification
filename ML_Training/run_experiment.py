import numpy as np
import argparse
import json
import socket
import os
import subprocess

def get_args():
    parser = argparse.ArgumentParser(
        description='Charcoal Dryrot Run experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e',
                        '--exp',
                        help='The name of the experiment',
                        metavar='exp',
                        required=True)
    
    parser.add_argument('-m',
                        '--machine',
                        help='The name of the machine or HPC cluster to run the experiment on',
                        metavar='machine',
                        required=True)

    return parser.parse_args()
    
def load_config(path):
    
    with open(path,"r") as f:
        config = json.load(f)

    host_name = socket.gethostname()
    
    if "hpc" in host_name:
        host_name = "hpc"

    return config[host_name]

def run_puma(exp, config):

    group = "kobus"
    model = exp.split('_')[0]
    outdir = config['results']
    code = os.path.join(os.getcwd(),"Train.py")
    run_time = "15:00:00"
    script_content = f"#!/bin/bash\n#SBATCH --job-name=DryRot-Training\n#SBATCH --account={group}\n#SBATCH --partition=standard\n#SBATCH --ntasks=94\n#SBATCH --ntasks-per-node=94\n#SBATCH --nodes=1\n#SBATCH --mem=470gb\n#SBATCH --gres=gpu:1\n#SBATCH --time={run_time}\n#SBATCH -o {outdir}/%x_{model}.out\nexptype={model}\nmodule load anaconda\nconda init bash\nsource ~/.bashrc\nconda deactivate\nconda deactivate\nconda activate imerg\npython {code} -e {exp}"
    # print(script_content)
    with open(os.path.join(config['results'],"tmp.sh"),'w') as f:
        f.write(script_content)
    
    res = subprocess.run(["sbatch",os.path.join(config['results'],"tmp.sh")])
    
    if res.returncode != 0:
        print(f"Error: {res}")

    os.remove(os.path.join(config['results'],"tmp.sh"))



def main():

    args = get_args()
    config = load_config("../config.json")

    if args.machine == "puma":
        run_puma(args.exp,config)

if __name__ == "__main__":
    main()
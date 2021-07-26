import itertools
import json
import socket

def generate_exp_UNET(model_name):
    
    input_shape_values = [[256,256,3]]
    start_filter_values = [64]
    conv_kernel_size_values = [3]
    activation_values = ["relu"]
    drop_out_rate_values = [0.2]
    deconv_kernel_size_values = [3]    
    optimizer_values = ["Adam"]
    loss_values = ["weighted_dice_coef"]
    lr_values = [1e-3]
    epochs_values = [10]
    batch_size_values = [8]
    patience_values = [20]

    experiments = list(itertools.product(input_shape_values,start_filter_values,conv_kernel_size_values,\
        activation_values,drop_out_rate_values,deconv_kernel_size_values,optimizer_values,loss_values,lr_values,\
            epochs_values,batch_size_values,patience_values))

    final_experiments = {}

    for i,e in enumerate(experiments):
        new_e = list(e)
        
        name = model_name+"_"+str(i)
        final_experiments[name] = {\
            "input_shape":new_e[0],\
            "start_filter":new_e[1],\
            "conv_kernel_size":new_e[2],\
            "activation":new_e[3],\
            "dr_rate":new_e[4],\
            "deconv_kernel_size":new_e[5],\
            "optimizer":new_e[6],\
            "loss":new_e[7],\
            "lr":new_e[8],\
            "epochs":new_e[9],\
            "batch_size":new_e[10],\
            "patience":new_e[11]
            }

    print(":: Number of total experiments: "+str(len(final_experiments)))

    return final_experiments

def generate_exp_ResNet(model_name):

    input_shape_values = [[256,256,3]]
    lr_values = [1e-3]
    optimizer_values = ["Adam"]
    loss_values = ["categorical_crossentropy"]
    epochs_values = [20]
    batch_size_values = [16]
    patience_values = [20]

    experiments = list(itertools.product(input_shape_values,lr_values,optimizer_values,loss_values,\
            epochs_values,batch_size_values,patience_values))

    final_experiments = {}

    for i,e in enumerate(experiments):
        new_e = list(e)
        
        name = model_name+"_"+str(i)
        final_experiments[name] = {\
            "input_shape": new_e[0],\
            "lr": new_e[1],\
            "optimizer":new_e[2],\
            "loss": new_e[3],\
            "epochs":new_e[4],\
            "batch_size":new_e[5],\
            "patience":new_e[6]
            }

    print(":: Number of total experiments: "+str(len(final_experiments)))

    return final_experiments

def generate_exp_MobileNetV1(model_name):

    input_shape_values = [[256,256,3]]
    lr_values = [1e-3]
    optimizer_values = ["Adam"]
    loss_values = ["categorical_crossentropy"]
    epochs_values = [20]
    batch_size_values = [16]
    patience_values = [20]

    experiments = list(itertools.product(input_shape_values,lr_values,optimizer_values,loss_values,\
            epochs_values,batch_size_values,patience_values))

    final_experiments = {}

    for i,e in enumerate(experiments):
        new_e = list(e)
        
        name = model_name+"_"+str(i)
        final_experiments[name] = {\
            "input_shape": new_e[0],\
            "lr": new_e[1],\
            "optimizer":new_e[2],\
            "loss": new_e[3],\
            "epochs":new_e[4],\
            "batch_size":new_e[5],\
            "patience":new_e[6]
            }

    print(":: Number of total experiments: "+str(len(final_experiments)))

    return final_experiments

def save_settings(final_experiments,path):

    with open(path,"w") as f:
        json.dump(final_experiments,f)

def load_config(path):
    
    with open(path,"r") as f:
        config = json.load(f)
    
    host_name = socket.gethostname()

    if "hpc" in host_name:
        host_name = "hpc"

    return config[host_name]

config = load_config("../config.json")

exp_UNET = generate_exp_UNET("UNET")
exp_ResNET = generate_exp_ResNet("ResNET")
exp_MobileNETV1 = generate_exp_MobileNetV1("MobileNETV1")

save_settings(exp_UNET,config['settings']+"/experiments_UNET.json")
save_settings(exp_ResNET,config['settings']+"/experiments_ResNET.json")
save_settings(exp_MobileNETV1,config['settings']+"/experiments_MobileNETV1.json")


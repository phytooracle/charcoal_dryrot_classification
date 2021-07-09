import numpy as np
import json
import os
import cv2
import multiprocessing
import math 
import random
import argparse
from PIL import Image, ImageDraw

def read_json_label_files(path):

    data = None

    with open(path,"r") as f:
        text = f.read()

        data = json.loads(text)

    return data

def get_polygons(data):

    polygons = {}

    for d in data:
        image_name = d['External ID'].replace('.JPG','')
        label_dict = d['Label']

        polygons[image_name] = []

        if len(label_dict) == 0:
            continue

        for obj in label_dict['objects']:
            if 'polygon' in obj:
                polygon = obj['polygon']
                polygons[image_name].append(polygon)

    return polygons

def get_single_image_size(img_name):
    img = cv2.imread(img_name)
    return img_name,img.shape

def get_image_sizes(image_path,no_cores):

    image_sizes = {}

    files = os.listdir(image_path)

    args_list = []

    for f in files:
        args_list.append('{0}/{1}'.format(image_path,f))
    
    processes = multiprocessing.Pool(no_cores)
    results = processes.map(get_single_image_size,args_list)
    processes.close()

    for path,size in results:
        image_sizes[path.split('/')[-1].replace('.JPG','')] = size

    return image_sizes

def generate_single_mask(args):
    
    width = args[0]
    height = args[1]
    polygons = args[2]
    path_to_write = args[3]

    img = Image.new('L', (width, height), 0)
    
    for polygon in polygons:
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=255)

    img.save(path_to_write)

def generate_masks(list_polygons,image_sizes,path_to_write,no_cores):
    
    args = []

    for img in list_polygons:

        size = image_sizes[img]
        polygons = list_polygons[img]
        new_polygons = []

        for p in polygons:
            new_polygons.append([(int(d['x']),int(d['y'])) for d in p])

        args.append([size[1],size[0],new_polygons,'{0}/{1}.png'.format(path_to_write,img)])

    processes = multiprocessing.Pool(no_cores)
    processes.map(generate_single_mask,args)
    processes.close()

def generate_single_patch(args):

    img_address = args[0]
    mask_address = args[1]
    patch_address = args[2]
    annotation_address = args[3]
    width = args[4]
    height = args[5]
    img_name = args[6]

    img = cv2.imread(img_address)
    mask = cv2.imread(mask_address)

    if mask is None:
        return 

    x = 0
    y = 0
    i = 0

    mask = mask/255
    mask = mask.astype('uint8')

    while x+width<img.shape[1]:

        y = 0

        while y+height<img.shape[0]:
            sub_img = img[y:y+height,x:x+width]
            sub_mask = mask[y:y+height,x:x+width]

            cv2.imwrite('{0}/{1}-{2}.png'.format(patch_address,img_name,i),sub_img)
            cv2.imwrite('{0}/{1}-{2}.png'.format(annotation_address,img_name,i),sub_mask)

            i+=1
            y+=height

        x+=width

def generate_patches(image_path,mask_path,dataset_address,width,height,training_perc,validation_perc,test_perc,no_cores):

    list_images = os.listdir(image_path)

    if training_perc+validation_perc+test_perc!=1.0:
        print('Training, validation and test ratios are not correct.')
        return

    training_n = int(math.floor(training_perc*len(list_images)))
    validation_n = int(math.floor(validation_perc*len(list_images)))
    test_n = int(math.ceil(test_perc*len(list_images)))

    list_training = [list_images.pop(random.randrange(len(list_images))) for _ in range(training_n)]
    list_validation = [list_images.pop(random.randrange(len(list_images))) for _ in range(validation_n)]
    list_test = [list_images.pop(random.randrange(len(list_images))) for _ in range(test_n)]

    if not os.path.exists(dataset_address):
        os.mkdir(dataset_address)

    if not os.path.exists('{0}/training'.format(dataset_address)):
        os.mkdir('{0}/training'.format(dataset_address))

    if not os.path.exists('{0}/training/images'.format(dataset_address)):
        os.mkdir('{0}/training/images'.format(dataset_address))

    if not os.path.exists('{0}/training/annotation'.format(dataset_address)):
        os.mkdir('{0}/training/annotation'.format(dataset_address))


    if not os.path.exists('{0}/validation'.format(dataset_address)):
        os.mkdir('{0}/validation'.format(dataset_address))

    if not os.path.exists('{0}/validation/images'.format(dataset_address)):
        os.mkdir('{0}/validation/images'.format(dataset_address))

    if not os.path.exists('{0}/validation/annotation'.format(dataset_address)):
        os.mkdir('{0}/validation/annotation'.format(dataset_address))

    
    if not os.path.exists('{0}/test'.format(dataset_address)):
        os.mkdir('{0}/test'.format(dataset_address))

    if not os.path.exists('{0}/test/images'.format(dataset_address)):
        os.mkdir('{0}/test/images'.format(dataset_address))

    if not os.path.exists('{0}/test/annotation'.format(dataset_address)):
        os.mkdir('{0}/test/annotation'.format(dataset_address))

    args_list = []

    for img_file in list_training:
        patch_address = '{0}/training/images'.format(dataset_address)
        annotation_address = '{0}/training/annotation'.format(dataset_address)
        args_list.append(('{0}/{1}'.format(image_path,img_file),'{0}/{1}'.format(mask_path,img_file.replace('JPG','png')),patch_address,annotation_address,width,height,img_file.replace('.JPG','')))

    for img_file in list_validation:
        patch_address = '{0}/validation/images'.format(dataset_address)
        annotation_address = '{0}/validation/annotation'.format(dataset_address)
        args_list.append(('{0}/{1}'.format(image_path,img_file),'{0}/{1}'.format(mask_path,img_file.replace('JPG','png')),patch_address,annotation_address,width,height,img_file.replace('.JPG','')))

    for img_file in list_test:
        patch_address = '{0}/test/images'.format(dataset_address)
        annotation_address = '{0}/test/annotation'.format(dataset_address)
        args_list.append(('{0}/{1}'.format(image_path,img_file),'{0}/{1}'.format(mask_path,img_file.replace('JPG','png')),patch_address,annotation_address,width,height,img_file.replace('.JPG','')))

    processes = multiprocessing.Pool(no_cores)
    processes.map(generate_single_patch,args_list)
    processes.close()

def get_args():
    
    parser = argparse.ArgumentParser(
        description='Generating masks and patches for the dry rot images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-l',
                        '--label',
                        help='The path to the labels json file.',
                        metavar='label',
                        required=True)

    parser.add_argument('-i',
                        '--images',
                        help='The path to all the dry rot images.',
                        metavar='images',
                        required=True)

    parser.add_argument('-m',
                        '--masks',
                        help='The path to all the dry rot masks.',
                        metavar='masks',
                        required=True)
    
    parser.add_argument('-p',
                        '--patches',
                        help='The path to all the dry rot patches.',
                        metavar='patches',
                        required=True)

    parser.add_argument('-c',
                        '--cores',
                        help='The number of cores to use.',
                        metavar='cores',
                        type=int,
                        required=True,
                        default=90)


    return parser.parse_args()

def main():

    args = get_args()

    data = read_json_label_files(args.label)

    polygons = get_polygons(data)

    image_sizes = get_image_sizes(args.images,args.cores)

    generate_masks(polygons,image_sizes,args.masks,args.cores)

    generate_patches(args.images,args.masks,args.patches,256,256,0.6,0.2,0.2,args.cores)

main()
# Charcoal_Dry_Rot
In this repository, I maintain the codes and scripts for the charcoal dry rot project. We do preprocessing, classification, and segmentation of the images.

# Running Scripts
In this section I keep some templates for running the different scripts we have in this repository. These are mainly intended for HPC systems.

* Using Singularity

    1. Preprocessing: ```mask_and_patch_generatory.py```\
    This script generate masks for the segmentation task and splits up the raw images and their masks into patches of the specific size. Use this template to run the script:
    ```singularity exec --nv /xdisk/ericlyons/data/ariyanzarei/IMERGE-Project/singularity_images/imerg_deep_learning_latest.sif python mask_and_patch_generator.py -l /xdisk/kobus/ariyanzarei/charcoal_dryrot/data/json_labels/labels_2021-06-21.json -i /xdisk/kobus/ariyanzarei/charcoal_dryrot/data/raw_images -m /xdisk/kobus/ariyanzarei/charcoal_dryrot/data/raw_masks -p /xdisk/kobus/ariyanzarei/charcoal_dryrot/data/patches -c 16```

    2. Preprocessing: ```dataset_generator.py```\
    This script generate train and validation datasets in the form of numpy files from the patches and their masks. Use this template to run the script:
    ```singularity exec --nv /xdisk/kobus/ariyanzarei/singularity_images/deep_learning_full_image.simg python dataset_generator.py -d /xdisk/kobus/ariyanzarei/charcoal_dryrot/data/dataset -p /xdisk/kobus/ariyanzarei/charcoal_dryrot/data/patches```

    3. Preprocessing: ```dataset_generator_classification.py```\
    This script generate train and validation datasets in the form of numpy files from the patches and their masks for the classification models. Use this template to run the script:
    ```singularity exec --nv /xdisk/kobus/ariyanzarei/singularity_images/deep_learning_full_image.simg python dataset_generator_classification.py -d /xdisk/kobus/ariyanzarei/charcoal_dryrot/data/dataset/classification -p /xdisk/kobus/ariyanzarei/charcoal_dryrot/data/patches```

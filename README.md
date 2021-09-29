# Charcoal_Dry_Rot
This repository contains codes, scripts and link to the dataset for the charcoal dry rot classification project. There are scripts available for preprocessing images, training and testing models.

# Data
The raw images as well as the annoted masks are available at XXX. You can use the scripts in the preprocessing folder to prepare the dataset for training classification models. 

# Code
The following explains how to prepare data and train models. 

* Preparing the data (Preprocessing Folder)
  * Use the ```mask_and_patch_generator.py``` to generate training, validation and test patches. 
  * Use ```dataset_generator.py``` to generate h5 dataset that is used in the training procedure.
  * Use ```dataset_visualizer.py``` to visualize and validate data.
* Training models (ML_Training Folder)
  * Generate grid search hyper parameter tuning: ```Experiment_Generator.py```
  * Generate results plots: ```Generate_Plots.py```
  * Definition of the models: ```Models.py```
  * Testing the models (on Jetson and any other machines): ```Test.py```
  * Training the models: ```Train.py```
  * Some utility functions: ```Utils.py```
  * Script to run experiments in batch on HPC systems: ```run_experiment.py``` 

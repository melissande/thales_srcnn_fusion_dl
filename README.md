# thales_srcnn_fusion_dl


#########################
####SRCNN_FUSION#########
#########################

CAREFUL: this code has to be run on GPU as it's heavy computation

# 0) Setting up the environment

from linux terminal in the folder SRCNN_FUSION/

```sh
$conda env create -f env_srcnn_fusion.yml
$source activate special_course_jmg
```
Then, install tensorflow on gpu (version depends on server used) with the conda virtual environment special_course_jmg (see tensorflow documentation online)
Then, install imgaug: 
```sh
$sudo pip install git+https://github.com/aleju/imgaug
```

# 1) Extract the patches =subset of images

Create a folder called SRCNN_FUSION/DATA (if not created), place each of the data available (pan ,pxs and the 4 xs) in distinct folders. For instance if I have two datasets, I create two sub folders of DATA/:
- images_1/ ->pan ,pxs and the 4 xs
- images_2/ -> pan ,pxs and the 4 xs

In each of these folder, rename:
- the pansharpened image pxs.tif
- the panchromatic image pan.tif
- the xs images xs1.tif, xs2.tif, xs3.tif, xs4.tif

Now go out of the DATA/ folder, in SRCNN_FUSION/, and build the subset of images with:
```sh
$python patche_extraction.py  DATA/
```

This operation takes a bit of time
Check the amount of subimages created in each of the images_i folders for PAN_FOLDER,XS_HR_FOLDER and PXS_FOLDER. There should be the same amount and let's call it AMOUNT.



# 2) Build the patches

In the folder SRCNN_FUSION/, create 3 folders: TRAINING, VALIDATION and VERIFICATION 

Then run the following, depending on which set you want to build the patches (TRAINING, VALIDATION and VERIFICATION). I personally choose to have one ORIGINAL image in the TRAINING (let's say images_1), one ORIGINAL in  VERIFICATION (let's say images_2),one ORIGINAL in  VALIDATION (let's say images_3). For the moment set COUNT to 0 (last argument). (I will explain below).

For instance, let's build the training set using images_1:
```sh	
$python build_data.py  DATA/images_1/ AMOUNT TRAINING/ 0
```
or let's build the validation set using images_3:
```sh	
$python build_data.py  DATA/images_3/ AMOUNT VALIDATION/ 0
```
or let's build the verification set using images_2:
```sh	
$python build_data.py  DATA/images_2/ AMOUNT VERIFICATION/ 0
```


These operations take a bit of time.

Now, if you want to add images to a certain set (TRAINING, VALIDATION and VERIFICATION), let's say we want to increase the TRAINING set with data from images_4, we first need to know how many patches there are in TRAINING.

Go to the TRAINING/INPUT/ folder, and run
```sh	
$ls -1 | wc -l
```
The result will be the COUNT. Now come back to SRCNN_FUSION/ and run
```sh
$python build_data.py  DATA/images_4/ AMOUNT TRAINING/ COUNT
```

# 3) Training

The model is going to be stored (from scratch) or is stored (from existing model)  at the path defined in the variable GLOBAL_PATH which can me modified in the script simple_cnn_baseline.py line 26. (GLOBAL_PATH=‘MODEL_FINAL’)

The true patches and the estimation patches from the validation test after training are saved in MODEL_FINAL/TEST_SAVE. 


## To train the model from scratch 

run:
```sh
python simple_cnn_baseline.py train
```

## To train the model from existing model 


run:
```sh
python simple_cnn_baseline.py train path_to_model
```
where path_to_model=MODEL_FINAL/srcnn_meli_model.ckpt is the existing model I did which is described in  the report.

In the script simple_cnn_baseline.py, the parameters that can be tuned have been grouped as following

### Parameters to tune
```py
rec_save=1#400 #How often do you save the model? every  400 iterations for instance
dropout=0.7
LEARNING_RATE=0.0001
```

### Network parameters
```py
LAYERS=3
FILTERS_NB=[INPUT_SIZE,64,32,OUTPUT_SIZE]
FILTERS_WIDTH=[9,5,5]
```
###  Tune model training
```py
DEFAULT_BATCH_SIZE = 36
DEFAULT_EPOCHS = 2#500 is advised
DEFAULT_ITERATIONS =3#850 is advised
DEFAULT_VERIF = 50
DEFAULT_VALID=4
```

# 4) Prediction
```sh
$python simple_cnn_baseline.py predict path_folder_image_to_predict path_to_model x w y h
```

where path_folder_image_to_predict=DATA/images_5/ for instance
path_to_model=MODEL_FINAL/srcnn_meli_model.ckpt
x w y h allows to crop the image to see the results as they are really big! For instance if I want the prediction of the image beginning at pixel [14,15] and of width=200 and height=300, we will have to set x=13, y=15, width=200 and h=300. If you want to predict the full image, just set x=0, y=0 and width= width of the image and height= height of the image.

The true image and the estimation are saved in MODEL_FINAL/TEST_SAVE. 

# 5) Jupyter Notebook
The script simple_cnn_baseline.py  is to be run from terminal using gpu but it can also be done from jupyter notebook and that’s why a jupiter notebook (simple_cnn_basline_nb.ipynb) is available. Go to the main, where it’s written ‘Configuration to tune’ to set training or prediction, restore model, dimensions of the crop for the image to predict etc.   


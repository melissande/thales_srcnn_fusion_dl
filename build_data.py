import tensorflow as tf
import h5py
import numpy as np
import cv2
import sys
import os

OUTPUT_PXS_CHANNELS=4
INPUT_PAN_CHANNELS=1
INPUT_XS_CHANNELS=4

PATH_PAN='PAN_FOLDER/pan_'
PATH_XS_HR='XS_HR_FOLDER/xs_hr_'
PATH_PXS='PXS_FOLDER/pxs_'


PATH_INPUT='INPUT/input'
PATH_OUTPUT='OUTPUT/output'



def read_data_h5(path):
    '''
    Reads h5 file (subimages contained in PAN_FOLDER, XS_HR_FOLDER and PXS_FOLDER)
    :path path of the file to read
    returns data as an array
    '''
    with h5py.File(path, 'r') as hf:
        data=np.array(hf.get('data'))
    return data

def write_data_h5(path,data_,number):
    '''
    Writes h5 file (patch in each set (TRAINING, VALIDATION, VERIFICATION))
    :path path of the file to write
    :data_ data to write into the file
    :number patch number
    '''
    with h5py.File(path+'_'+str(number)+'.h5','w') as hf:
           hf.create_dataset('data',data=data_)
    print('File'+path+'_'+str(number)+'.h5'+' created')

if __name__ == '__main__':

    #See Readme.txt
    path = sys.argv[1] #DATA/images_1/ for instance
    nb_subfolders=int(float(sys.argv[2])) #AMOUNT=4 for instance 
    folder_set=sys.argv[3] #TRAINING/ for instance
    count=int(float(sys.argv[4])) #COUNT=0 if no patch has been created in the TRAINING/ set yet
    
    
    #Creates INPUT (panchromatic+4XS) and OUTPUT (true output= pansharpened image) folders 
    
    if not os.path.exists(folder_set+'INPUT/'):
        os.makedirs(folder_set+'INPUT/')
    if not os.path.exists(folder_set+'OUTPUT/'):
        os.makedirs(folder_set+'OUTPUT/')
        
    #Go through all the subimages
    for i in range(nb_subfolders):
        print('Iteration: '+str(i))
        pxs=read_data_h5(path+PATH_PXS+str(i)+'.h5')
        pan=read_data_h5(path+PATH_PAN+str(i)+'.h5')
        xs_hr=read_data_h5(path+PATH_XS_HR+str(i)+'.h5')
        print('Data read')

        
        data_size=px.shape[0]#the number of patches
        for j in range(0,data_size):
            count=count+1
            input_=np.concatenate((pan[j,:,:,:],xs_hr[j,:,:,:]),axis=-1)
            write_data_h5(folder_set+PATH_INPUT,input_,count)
            write_data_h5(folder_set+PATH_OUTPUT,pxs[j,:,:,:],count)
            

    print('Number of files created: %d '% (count), end='\r')

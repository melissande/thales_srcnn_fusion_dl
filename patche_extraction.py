import tensorflow as tf
import h5py
import numpy as np
import cv2
import sys
import os


OUTPUT_PXS_CHANNELS=4
INPUT_PAN_CHANNELS=1
INPUT_XS_CHANNELS=1


NAME_XS1='xs1.tif'
NAME_XS2='xs2.tif'
NAME_XS3='xs3.tif'
NAME_XS4='xs4.tif'
NAME_PXS='pxs.tif'
NAME_PAN='pan.tif'

PATH_XS_HR_L='XS_HR_L_FOLDER/xs_hr_l.h5'
PATH_XS_HR_L_FOLD='XS_HR_L_FOLDER'

PATH_PAN='PAN_FOLDER/pan_'
PATH_PAN_FOLD='PAN_FOLDER'

PATH_XS_HR='XS_HR_FOLDER/xs_hr_'
PATH_XS_HR_FOLD='XS_HR_FOLDER'

PATH_PXS='PXS_FOLDER/pxs_'
PATH_PXS_FOLD='PXS_FOLDER'

WIDTH=33
STRIDE=14
def read_data(path):
    '''
    Reads tif images (pan.tif,pxs.tif, xs1.tif etc.)
    :path where is the image
    '''

    data=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    return data
def read_data_h5(path):
    '''
    Reads h5 file
    :path path of the file to read
    returns data as an array
    '''
    with h5py.File(path, 'r') as hf:
        data=np.array(hf.get('data'))
    return data
def write_data_h5(path,data_):
    '''
    Writes h5 file
    :path path of the file to write
    :data_ data to write into the file
    '''
    with h5py.File(path,'w') as hf:
           hf.create_dataset('data',data=data_)
    print('File'+path+' created')

def prepare_xs_hr_full(path,size_hr):
    '''
    Prepares the upsampled XS image and saves it under .h5 file format
    :path where to save the output image
    :size_hr to upsample the Low Resolution XS images to the dimension of the High Resolution panchromatic image
    '''
    xs1=read_data(path+NAME_XS1)
    xs2=read_data(path+NAME_XS2)
    xs3=read_data(path+NAME_XS3)
    xs4=read_data(path+NAME_XS4)
    xs1=xs1/255.
    xs2=xs2/255.
    xs3=xs3/255.
    xs4=xs4/255.

    size_lr=[xs1.shape[0],xs1.shape[1]]
    xs1_ph=tf.placeholder(tf.float64, [size_lr[0],size_lr[1]], name='xs1_placeholder')
    xs2_ph=tf.placeholder(tf.float64, [size_lr[0],size_lr[1]], name='xs2_placeholder')
    xs3_ph=tf.placeholder(tf.float64, [size_lr[0],size_lr[1]], name='xs3_placeholder')
    xs4_ph=tf.placeholder(tf.float64, [size_lr[0],size_lr[1]], name='xs4_placeholder')
    
    xs1_t=tf.reshape(xs1_ph,[1,size_lr[0],size_lr[1],INPUT_XS_CHANNELS],name='reshape_xs1')
    xs2_t=tf.reshape(xs2_ph,[1,size_lr[0],size_lr[1],INPUT_XS_CHANNELS],name='reshape_xs2')
    xs3_t=tf.reshape(xs3_ph,[1,size_lr[0],size_lr[1],INPUT_XS_CHANNELS],name='reshape_xs3')
    xs4_t=tf.reshape(xs4_ph,[1,size_lr[0],size_lr[1],INPUT_XS_CHANNELS],name='reshape_xs4')
    xs_hr=tf.concat((xs1_t,xs2_t,xs3_t,xs4_t),-1,name='concat_xs')

    xs_hr=tf.image.resize_images(xs_hr, [size_hr[0], size_hr[1]])
    xs_hr=tf.cast(xs_hr,tf.float64,name='cast_xs_hr')


    with tf.Session() as sess:
        Xs_hr= sess.run(xs_hr,feed_dict={xs1_ph: xs1,xs2_ph: xs2,xs3_ph: xs3,xs4_ph:xs4})
        write_data_h5(path+PATH_XS_HR_L,Xs_hr) 

def extract_patches(data,width,stride,path_out):
    '''
    Extract patches from images and writes the output to .h5 file format
    :data input image (pan+xs_hr or pxs)
    :width dimensiton of the patch
    :stride stride of patch selection on the image
    :path_out where to save the patches (input or output)
    '''
    
    data_pl=tf.placeholder(tf.float64, [data.shape[0],data.shape[1],data.shape[2],data.shape[3]], name='data_placeholder')
    data_o=tf.extract_image_patches(images=data_pl,ksizes=[1,width,width,1],strides=[1,stride,stride,1],rates=[1,1,1,1],padding='VALID')
    size_tot=data_o.get_shape().as_list()
    data_o=tf.reshape(data_o,[size_tot[1]*size_tot[2],width,width,data.shape[3]])
    with tf.Session() as sess:
        Data_o= sess.run(data_o,feed_dict={data_pl: data})
        write_data_h5(path_out,Data_o)
if __name__ == '__main__':
    #See Readme.txt
    paths=sys.argv[1] # DATA/
    
    
    
    for path in sorted(os.listdir(paths)):
        ##PANCHROMATIC PATCHES
        path=paths+path+'/'
        pan=read_data(path+NAME_PAN)
        print('PAN loaded')
        pan=pan/255.
        size_hr=[pan.shape[0],pan.shape[1]]
        pan=np.reshape(pan,[1,size_hr[0],size_hr[1],INPUT_PAN_CHANNELS])
        if not os.path.exists(path+PATH_PAN_FOLD):
            os.makedirs(path+PATH_PAN_FOLD)

        ##Determined in how many subimages to crop the big image
        rate_w=int(round(size_hr[0]/6000))
        rate_h=int(round(size_hr[1]/6000))

        ##PREPARE XS_HR (UPSAMPLING -> NEED SIZE HR)
        if not os.path.exists(path+PATH_XS_HR_L_FOLD):
            os.makedirs(path+PATH_XS_HR_L_FOLD)
        prepare_xs_hr_full(path,size_hr)
        print('XS_HR prepared')
        xs_hr_full_l=read_data_h5(path+PATH_XS_HR_L)
        print('XS_HR loaded')
        if not os.path.exists(path+PATH_XS_HR_FOLD):
            os.makedirs(path+PATH_XS_HR_FOLD)


        ##PANCHARPENED PATCHES
        if not os.path.exists(path+PATH_PXS_FOLD):
            os.makedirs(path+PATH_PXS_FOLD)
        pxs=read_data(path+NAME_PXS)
        pxs=pxs/255.
        pxs=np.reshape(pxs,[1,pxs.shape[0],pxs.shape[1],OUTPUT_PXS_CHANNELS])
        print('PXS loaded')

        wr=int(size_hr[0]/rate_w)
        hr=int(size_hr[1]/rate_h)
        count=0

        for i in range(rate_w):
            for j in range(rate_h):

                pan_i=pan[:,(i*wr):((i+1)*wr),(j*hr):((j+1)*hr),:]
                extract_patches(pan_i,WIDTH,STRIDE,path+PATH_PAN+str(count)+'.h5')
                pxs_i=pxs[:,(i*wr):((i+1)*wr),(j*hr):((j+1)*hr),:]
                extract_patches(pxs_i,WIDTH,STRIDE,path+PATH_PXS+str(count)+'.h5')
                xs_hr_full_l_i=xs_hr_full_l[:,(i*wr):((i+1)*wr),(j*hr):((j+1)*hr),:]
                extract_patches(xs_hr_full_l_i,WIDTH,STRIDE,path+PATH_XS_HR+str(count)+'.h5')
                count+=1

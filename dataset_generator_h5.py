import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa


PAN_SIZE=1
TOT_XS_SIZE=4
PXS_SIZE=4

PATH_INPUT='INPUT/'
PATH_OUTPUT='OUTPUT/'
SIZE_PATCH=33




def _parse_images(paths_input,paths_output):
    '''
    Reads and saves as as an array image input and output
    :paths_input array of paths of the input images that have to be read  
    :paths_output array of paths of the output images that have to be read  
    returns input and output images as array
    '''
    input_ = []
    output_ = []

    for path_i in paths_input:
        with h5py.File(path_i, 'r') as hf:
            X =np.array(hf.get('data'))
            input_.append(X)

    for path_o in paths_output:
        with h5py.File(path_o, 'r') as hf:
            Y =np.array(hf.get('data'))
            output_.append(Y)

        
            
    return np.asarray(input_),np.asarray(output_)
        
def data_augment(input_,output_,batch_size):
    '''
    Augments the data with transformation but only select batch_size number of data
    :input_ input data
    :output_ output data
    :batch_size size of the batch
    returns input and output arrays of data augmented with transformation or with identity  
    '''
    seq = iaa.Sequential([iaa.Add((-40, 40)),iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),iaa.Multiply((0.5, 1.5))])
    input_int=input_*255
    input_int.astype(int)
    pan_new=seq.augment_images(input_int[:,:,:,0])
    
    pan_new=np.reshape(pan_new,[pan_new.shape[0],pan_new.shape[1],pan_new.shape[2],1])  
    input_new=np.concatenate((pan_new,input_int[:,:,:,1:]),axis=-1)
    
    min_t=np.amin(np.reshape(input_new,[len(input_new)*SIZE_PATCH*SIZE_PATCH,(PAN_SIZE+PXS_SIZE)]), axis=0)
    max_t=np.amax(np.reshape(input_new,[len(input_new)*SIZE_PATCH*SIZE_PATCH,(PAN_SIZE+PXS_SIZE)]), axis=0)
    input_new=(input_new-min_t)/(max_t-min_t)
    
    input_tot=np.concatenate((input_,input_new),axis=0)
    output_tot=np.concatenate((output_,output_),axis=0)

   
    idx = np.arange(len(input_tot))
    np.random.shuffle(idx)
    input_tot=input_tot[idx]
    output_tot=output_tot[idx]
    

    return input_tot[:batch_size,:,:,:],output_tot[:batch_size,:,:,:]


class DatasetGenerator():
    '''
    DatasetGenerator class
    '''

    # This decides whether "unique" keys should be included in the generator for each datapoint (typically useful for feature caching)
    include_keys = False

    #img_size = DATA_PATCH_INPUT_SIZE



    def __init__(self, paths_input: np.ndarray,paths_output: np.ndarray, batch_size: int = None):
        self.paths_input = paths_input
        self.paths_output = paths_output
        self.batch_size = batch_size

    @classmethod
    def from_root_folder(cls, root_folder: str, *, batch_size: int = None,max_data_size:  int = None):
        paths_input = []
        paths_output=[]
        
        
        for filename in sorted(os.listdir(root_folder+PATH_INPUT))[:max_data_size]:
            paths_input.append(os.path.join(root_folder+PATH_INPUT, filename))

        for filename in sorted(os.listdir(root_folder+PATH_OUTPUT))[:max_data_size]:

            paths_output.append(os.path.join(root_folder+PATH_OUTPUT, filename))
        
        
        return DatasetGenerator(np.asarray(paths_input), np.asarray(paths_output), batch_size=batch_size)

    def shuffled(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        idx = np.arange(len(self.paths_input))
        np.random.shuffle(idx)
        generator = DatasetGenerator(self.paths_input[idx], self.paths_output[idx],batch_size=self.batch_size)
        generator.include_keys = self.include_keys


        return generator

    def __iter__(self):
        if self.batch_size is None:
            raise ValueError('Must set a batch size before iterating!')

        self.index = 0

        return self

    def __next__(self):

        while(self.index * self.batch_size) < len(self.paths_input):
            start = self.index * self.batch_size
            stop = min(start + self.batch_size, len(self.paths_input))

            X,Y = _parse_images(self.paths_input[start:stop],self.paths_output[start:stop])


            self.index += 1
            if self.include_keys:
                return self.paths_input[start:stop], X,self.paths_output[start:stop], Y
            else:
                return X, Y


        raise StopIteration
    def __data_aug__(self,X,Y):

        X,Y=data_augment(X,Y,self.batch_size)

        return X,Y


    def __len__(self):
        return len(self.paths_input)

    def __getitem__(self, val):
        if type(val) is not slice:
            raise ValueError('DatasetGenerators can only be sliced')

        sliced = DatasetGenerator(self.paths_input[val], self.paths_output[val],batch_size=self.batch_size)
        sliced.include_keys = self.include_keys


        return sliced


if __name__ == '__main__':

    root_folder = sys.argv[1]
    test_save= sys.argv[2]

    batch_size = 5
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[2])

    generator = DatasetGenerator.from_root_folder(root_folder, batch_size=batch_size)

    generator.shuffled()
    generator =generator.__iter__()
    
    
    for iteration in range(1):
        X,Y=generator.__next__()


        for i in range(len(X)):

            plt.imsave(test_save+'X_iter_pan'+str(iteration)+'batch_'+str(i)+'.jpg',np.squeeze(X[i,:,:,:PAN_SIZE]))
            plt.imsave(test_save+'X_iter'+str(iteration)+'batch_'+str(i)+'.jpg',X[i,:,:,PAN_SIZE:(PAN_SIZE+TOT_XS_SIZE)])
            plt.imsave(test_save+'Y_iter'+str(iteration)+'batch_'+str(i)+'.jpg',Y[i,:,:,:])
        exit()
    


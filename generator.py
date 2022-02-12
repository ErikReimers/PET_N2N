from __future__ import division
from pathlib import Path
import random
import numpy as np
import cv2
import itertools
from scipy.ndimage.interpolation import rotate
import imutils
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import time

#Grabs a random batch of noisy pairs
class TrainingImageGenerator(Sequence):
    #Initalize the class
    def __init__(self, volume_dir, batch_size=8, image_size=64, rotations=True, sum_label=False):
        volume_suffixes = (".i")
        self.volume_paths = [p for p in Path(volume_dir).glob("**/*") if p.suffix.lower() in volume_suffixes]
        self.volume_nb = len(self.volume_paths)
        self.batch_size = batch_size
        self.image_size = image_size
        self.volume_shape = (207,256,256)
        #self.volume_shape = (89,276,276)
        self.volume_dir = volume_dir
        self.rotations = rotations
        self.sum_label = sum_label


        #Check if the specified folder is empty
        if self.volume_nb == 0:
            raise ValueError("volume dir '{}' does not include any volumes".format(volume_dir))

        if self.sum_label == "True":
            volume_two = np.expand_dims(np.zeros(self.volume_shape),-1)

            for ii in range(len(self.volume_paths)):

                volume_path_two = self.volume_paths[ii]
                fid = open(volume_path_two, "r")
                volume_two = volume_two + np.expand_dims(np.reshape(np.fromfile(fid, dtype=np.float32), self.volume_shape),-1)
                fid.close()

            volume_two.astype('float32').tofile(volume_dir+'summed.ii')


    #len shows how many batches fit within the number of training image pairs
    def __len__(self):
        return self.volume_nb*self.volume_shape[0] // self.batch_size
    
    #getitem returns a batch worth of noisy image patch pairs
    def __getitem__(self, idx):

        volume_suffixes = (".i")
        
        #make the variables to hold the patches
        x = np.zeros((self.batch_size, self.image_size, self.image_size, 1), dtype=np.float64)
        y = np.zeros((self.batch_size, self.image_size, self.image_size, 1), dtype=np.float64)
        sample_id = 0
        
        #Come up with a list of combinations for the data
        #combinations will give only AB and not BA
        #permutations will give AB and BA

        combinations = list(itertools.combinations(range(len(self.volume_paths)), 2))
        #combinations = list(itertools.permutations(range(len(self.volume_paths)),2))
         

        #Keep creating patch pairs until the counter equals the batch size

        while True:
            #Choose a random combination of data
            combination = random.choice(combinations)
            
            #Open and read those volumes
            volume_path_one = self.volume_paths[combination[0]]
            
            fid = open(volume_path_one, "r")
            volume_one = np.expand_dims(np.reshape(np.fromfile(fid, dtype=np.float32), self.volume_shape),-1)
            fid.close()

            #If you want the target to be a sum, than do that
            if self.sum_label == "True":
                volume_two = np.expand_dims(np.zeros(self.volume_shape),-1)

                volume_path_two = self.volume_dir+'summed.ii'
                fid = open(volume_path_two, "r")
                volume_two = volume_two + np.expand_dims(np.reshape(np.fromfile(fid, dtype=np.float32), self.volume_shape),-1)
                fid.close()


            else:
                a=time.perf_counter()
                volume_path_two = self.volume_paths[combination[1]]
                fid = open(volume_path_two, "r")
                volume_two = np.expand_dims(np.reshape(np.fromfile(fid, dtype=np.float32), self.volume_shape),-1)
                fid.close()

            #Choose a random slice
            axial_slice_nb = random.choice(range(self.volume_shape[0]))

            image_one = volume_one[axial_slice_nb,:,:]
            image_two = volume_two[axial_slice_nb,:,:]

            #Check if both images are just all zeros, if so skip and try again
            if image_one.any() or image_two.any():

                image_one = image_one.astype(np.float64)#/3000
                image_two = image_two.astype(np.float64)#/3000

                #If you want to randomly rotate the dataset, do that
                if self.rotations == True:
                    degrees = random.random()*360

                    #image_one = rotate(image_one, angle=degrees,reshape = False)
                    #image_two = rotate(image_two, angle=degrees,reshape = False)
                    #image_one[np.where(image_one<0)] = 0;
                    #image_two[np.where(image_two<0)] = 0;
                    image_one = np.expand_dims(imutils.rotate(image_one,angle=degrees),-1)
                    image_two = np.expand_dims(imutils.rotate(image_two,angle=degrees),-1)

                    

                h, w, _ = image_one.shape

                #Check that the patch "image_size" is smaller than the original image itself    
                if h >= self.image_size and w >= self.image_size:
                    h, w, _ = image_one.shape
                    i = np.random.randint(h - self.image_size + 1)
                    j = np.random.randint(w - self.image_size + 1)
                    patch_one = image_one[i:i + self.image_size, j:j + self.image_size]
                    patch_two = image_two[i:i + self.image_size, j:j + self.image_size]
                    if patch_one.any() or patch_two.any():
                        x[sample_id] = patch_one
                        y[sample_id] = patch_two

                        sample_id += 1
                        
                        #Once the counter reaches the batch_size return the set of noisy pairs
                        if sample_id == self.batch_size:
                            return x,y

#Grabs all the volumes and returns them as matching validation image pairs
class ValGenerator(Sequence):
    #Initalize the class
    def __init__(self, volume_dir, gt_dir="not_specified", nb_val_images=32, rotations=True,sum_label=True):     
        volume_suffixes = (".i")
        self.volume_paths = [p for p in Path(volume_dir).glob("**/*") if p.suffix.lower() in volume_suffixes]
        self.gt_paths = [p for p in Path(gt_dir).glob("**/*") if p.suffix.lower() in volume_suffixes]
        self.volume_shape = (207,256,256)
        #self.volume_shape = (89,276,276)
        self.volume_nb = len(self.volume_paths)
        self.data = []
        self.rotations = rotations
        self.sum_label = sum_label
        self.nb_val_images = nb_val_images
        self.volume_dir = volume_dir


        #Check if the specified folder is empty
        if self.volume_nb == 0:
            raise ValueError("image dir '{}' does not include any volumes".format(volume_dir))
        if gt_dir != "not_specified":
            gt_volume_nb = len(self.gt_paths)
            if gt_volume_nb == 0:
                raise ValueError("gt image dir '{}' does not include any volumes".format(gt_dir))
 

        #combinations will give only AB and not BA
        #permutations will give AB and BA
        combinations = list(itertools.combinations(range(len(self.volume_paths)), 2))
        #combinations = list(itertools.permutations(range(len(self.volume_paths)),2))

        #Run through each validation pair and add them to the self.data
        for ii in range(nb_val_images):
            
            combination = random.choice(combinations)
        
            volume_path_one = self.volume_paths[combination[0]]


            fid = open(volume_path_one, "r")
            volume_one = np.expand_dims(np.reshape(np.fromfile(fid, dtype=np.float32), self.volume_shape),-1)
            fid.close()

            if self.sum_label == "True":
                volume_two = np.expand_dims(np.zeros(self.volume_shape),-1)
                
                if gt_dir != "not_specified":
                    for ii in range(len(self.gt_paths)):
                        volume_path_two = self.gt_paths[ii]
                        fid = open(volume_path_two, "r")
                        volume_two_temp = np.expand_dims(np.reshape(np.fromfile(fid, dtype=np.float32), self.volume_shape),-1)
                        fid.close()
                        volume_two = volume_two + volume_two_temp


                else:

                    volume_path_two = self.volume_dir+'summed.ii'


                    fid = open(volume_path_two, "r")
                    volume_two = np.expand_dims(np.reshape(np.fromfile(fid, dtype=np.float32), self.volume_shape),-1)
                    fid.close()
            else:

                if gt_dir != "not_specified":
                    volume_path_two = self.gt_paths[0]
                else:
                    volume_path_two = self.volume_paths[combination[1]]
                fid = open(volume_path_two, "r")
                volume_two = np.expand_dims(np.reshape(np.fromfile(fid, dtype=np.float32), self.volume_shape),-1)
                fid.close()

            image_one = np.zeros((self.volume_shape[1],self.volume_shape[2]))
            image_two = np.zeros((self.volume_shape[1],self.volume_shape[2]))
            while not image_one.any() or not image_two.any():
                axial_slice_nb = random.choice(range(self.volume_shape[0]))

                image_one = volume_one[axial_slice_nb,:,:]
                image_two = volume_two[axial_slice_nb,:,:]

            if self.rotations == True:
                degrees = random.random()*360
                #image_one = rotate(image_one, angle=degrees,reshape = False)
                #image_two = rotate(image_two, angle=degrees,reshape = False)
                #image_one[np.where(image_one<0)] = 0;
                #image_two[np.where(image_two<0)] = 0;
                image_one = np.expand_dims(imutils.rotate(image_one,angle=degrees),-1)
                image_two = np.expand_dims(imutils.rotate(image_two,angle=degrees),-1)


            #if image_one.any() or image_two.any():

            x = image_one.astype(np.float64)#/3000
            y = image_two.astype(np.float64)#/3000

            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)]) 

    #len will be the number of validation pairs
    def __len__(self):
        return self.nb_val_images

    #getitem will return the self.data of the index specified
    def __getitem__(self, idx):
        return self.data[idx]


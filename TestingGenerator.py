import numpy as np
import cv2
import matplotlib.pyplot as plt
from generator import TrainingImageGenerator, ValGenerator
import time

gt_dir="not_specified"
sum_label="True"
image_dir='PET_images/Summed_label/'
nb_val_images = 5
rotations=True
batch_size = 5
image_size=64


generator = TrainingImageGenerator(image_dir, batch_size=batch_size, image_size=image_size,rotations=rotations,sum_label=sum_label)


val_generator = ValGenerator(image_dir,gt_dir=gt_dir,nb_val_images=nb_val_images,rotations=rotations,sum_label=sum_label)


xg,yg=generator[0]

print(len(generator))


for ii in range(5):
    
    print('xg Min: %.3f, Max: %.3f' % (xg.min(), xg.max()))
    print('yg Min: %.3f, Max: %.3f' % (yg.min(), yg.max()))


    xv,yv=val_generator[ii]

    print('xv Min: %.3f, Max: %.3f' % (xv.min(), xv.max()))
    print('yv Min: %.3f, Max: %.3f' % (yv.min(), yv.max()))


    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(np.squeeze(xg[ii,:,:,:]), vmin=0, vmax=np.max(xg[ii,:,:,:])*0.1)
    axarr[0,1].imshow(np.squeeze(yg[ii,:,:,:]), vmin=0, vmax=np.max(yg[ii,:,:,:]) if sum_label == "True" or gt_dir != "not_specified" else np.max(yg[ii,:,:,:])*0.1)
    axarr[1,0].imshow(np.squeeze(xv), vmin=0, vmax=np.max(xv)*0.1)
    axarr[1,1].imshow(np.squeeze(yv), vmin=0, vmax=np.max(yv) if sum_label == "True" or gt_dir != "not_specified" else np.max(xv)*0.1)
    
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.show()
    


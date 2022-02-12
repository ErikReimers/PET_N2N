import argparse
import numpy as np
import numpy.matlib
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from model import get_model
import glob
import os
from PIL import Image
import tempfile



#Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="unet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_folder", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="if set, will display the ground truth image")
    args = parser.parse_args()
    return args

#I'm pretty sure this function is reduntant after changes
def get_image(image):
    return image.astype(dtype=np.float64)


def main():
    args = get_args()
    image_dir = args.image_dir
    weight_folder = args.weight_folder
    ground_truth = args.ground_truth

    #Take the most recently saved weight file in the folder
    list_of_files = glob.glob(weight_folder + '/*.hdf5')
    weight_file = max(list_of_files, key=os.path.getctime)

    model = get_model(args.model)
    model.load_weights(weight_file)
    
    #if saving the images, make the specified folder
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))
    image_paths.sort()

    #Snag an example image in order to see the image shape
    example_image = np.expand_dims(cv2.imread(str(image_paths[0]), cv2.IMREAD_ANYDEPTH), -1)
    h, w, _ = example_image.shape

    noise_image = np.zeros((h, w, len(image_paths)), dtype=np.float64)
    denoised_image = np.zeros((h, w, len(image_paths)), dtype=np.float64)
    ground_truth = np.zeros((h, w, len(image_paths)), dtype=np.float64) 

    #Go through each image in the Visualization folder and save the orginal noisey, ground truth, and denoised images
    for ii in range(len(image_paths)):
        noise_image[:,:,ii] = cv2.imread(str(image_paths[ii]), cv2.IMREAD_ANYDEPTH)

        pred = model.predict(np.expand_dims(noise_image[:,:,ii], 0))
        denoised_image[:,:,ii] = np.squeeze(get_image(pred[0]))
        ground_truth[:,:,ii] = cv2.GaussianBlur(noise_image[:,:,ii],(7,7),5)

    #Normalize the images according to their max pixel (noise image gets boosted inorder to see better)
    noise_image = noise_image/np.amax(noise_image)#*5
    noise_image = (noise_image>0.01)
    denoised_image = denoised_image/np.amax(denoised_image)
    #If a ground truth reference was specified open that up to display as well
    if args.ground_truth:
        juicy_slices = [120, 125, 130, 135, 140]
        juicy_slices = np.repeat(juicy_slices,9,axis=0)
        fid = open(args.ground_truth, "r")
        GT_volume = np.fromfile(fid, dtype=np.float32)
        GT_volume = np.reshape(GT_volume, (207,256,256))
        for ii in range(len(juicy_slices)):
            ground_truth[:,:,ii] = np.transpose(GT_volume[juicy_slices[ii],:,:])
        ground_truth = ground_truth/np.amax(ground_truth)
    else:
        ground_truth = ground_truth/np.amax(ground_truth)


    out_image = np.zeros((h, w * 3), dtype=np.float64)

    #Run through each image and display or save the three versions of that image
    for ii in range(len(image_paths)):
        out_image[:, :w] = ground_truth[:,:,ii]
        out_image[:, w:w * 2] = noise_image[:,:,ii]
        out_image[:, w * 2:] = denoised_image[:,:,ii]

        if args.output_dir:

       

            plt.imsave(str(output_dir.joinpath(image_paths[ii].name))[:-4] + ".png", np.squeeze(out_image), vmin=0, vmax=1)
            example1 = np.repeat(np.repeat(np.squeeze(out_image[96:127,111:148]),15,axis=0),15,axis=1)
            plt.imsave(str(output_dir.joinpath(image_paths[ii].name))[:-4] + "zoomGT.png", example1, vmin=0, vmax=1) 
            example2 = np.repeat(np.repeat(np.squeeze(out_image[96:127,(111+256):(148+256)]),15,axis=0),15,axis=1)
            plt.imsave(str(output_dir.joinpath(image_paths[ii].name))[:-4] + "zoomN.png", example2, vmin=0, vmax=1)
            example3 = np.repeat(np.repeat(np.squeeze(out_image[96:127,(111+256*2):(148+256*2)]),15,axis=0),15,axis=1)
            plt.imsave(str(output_dir.joinpath(image_paths[ii].name))[:-4] + "zoomDN.png", example3, vmin=0, vmax=1) 
        else:
            plt.imshow(out_image, vmin=0, vmax=1)
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.show()
        

if __name__ == '__main__':
    main()

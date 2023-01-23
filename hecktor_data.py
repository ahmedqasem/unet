import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  


# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt
# !pip install git+https://github.com/miykael/gif_your_nifti # nifti to gif 
import gif_your_nifti.core as gif2nif

# DEFINE seg-areas  
SEGMENT_CLASSES = {
    0 : 'background',
    1 : 'GTVp', # primary Gross Tumor Volumes (GTVp)
    2 : 'GTVn' #  nodal Gross Tumor Volumes (GTVn)
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 91
VOLUME_START_AT = 4 # first slice of volume that we will include

TRAIN_DATASET_PATH = 'data/train_images/'
TRAIN_MASKS_PATH = 'data/train_masks/'
# VALIDATION_DATASET_PATH = '../input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

# test_image_ct=nib.load(TRAIN_DATASET_PATH + 'CT/CHUM-001__CT.nii.gz').get_fdata()
# test_image_pet=nib.load(TRAIN_DATASET_PATH + 'PET/CHUM-001__PT.nii.gz').get_fdata()
# # test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
# # test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
# test_mask=nib.load(TRAIN_MASKS_PATH + 'CHUM-001.nii.gz').get_fdata()

#car_img = cv2.imread('./data/carvana/train_images/0d3adbbc9a8b_01.jpg')
# print(car_img.shape)


# print(test_image_ct.shape)
# print(test_image_pet.shape)
# pet_r = cv2.resize(test_image_pet, (test_image_ct.shape[0], test_image_ct.shape[1]) )
# print(pet_r.shape)

# # convert PET CT to 2 channel image
# concat_img = np.moveaxis(np.concatenate((np.expand_dims(test_image_ct, axis=0), np.expand_dims(pet_r, axis=0)), axis=0), 0, -2)
# print(concat_img.shape)
# print(concat_img[:,:,:,56].shape)

''' preview one slice '''
# fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20, 10))
# slice_w = 56
# ax1.imshow(test_image_ct[:,:,slice_w], cmap = 'gray')
# ax1.set_title('CT Image')
# ax2.imshow(test_image_pet[:,:,slice_w], cmap = 'gray')
# ax2.set_title('PET Image')
# ax3.imshow(concat_img[:,:,:,slice_w], cmap='gray', interpolation='hanning')
# ax3.set_title('PET Image')
# plt.show()


# Skip 50:-50 slices since there is not much to see
# fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
# ax1.imshow(rotate(montage(test_image_ct[50:-50,:,:]), 90, resize=True), cmap ='gray')
# plt.show()

# Skip 50:-50 slices since there is not much to see
# fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
# ax1.imshow(rotate(montage(test_mask[:,:,30:-30]), 90, resize=True), cmap ='gray')
# plt.show()

# print(test_image_ct[50:-50,:,:].shape)

# gif2nif.write_gif_normal(f'./{TRAIN_DATASET_PATH}/CHUM-001__CT.nii.gz')


# niimg = nl.image.load_img(TRAIN_DATASET_PATH + 'CHUM-001__CT.nii.gz')
# nimask = nl.image.load_img(TRAIN_MASKS_PATH + 'CHUM-001.nii.gz')

# fig, axes = plt.subplots(nrows=4, figsize=(30, 40))

''' preview ct image '''
# nlplt.plot_anat(niimg,
#                 title='BraTS20_Training_001_flair.nii plot_anat',
#                 axes=axes[0])

# nlplt.plot_epi(niimg,
#                title='BraTS20_Training_001_flair.nii plot_epi',
#                axes=axes[1])

# nlplt.plot_img(niimg,
#                title='BraTS20_Training_001_flair.nii plot_img',
#                axes=axes[2])

# nlplt.plot_roi(nimask, 
#                title='BraTS20_Training_001_flair.nii with mask plot_roi',
#                bg_img=niimg, 
#                axes=axes[3], cmap='Paired')

# plt.show()

''' new '''
# 1. load images and masks
test_image_ct=nib.load(TRAIN_DATASET_PATH + 'CHUM-001__CT.nii.gz').get_fdata()
test_image_pet=nib.load(TRAIN_DATASET_PATH + 'CHUM-001__PT.nii.gz').get_fdata()
test_mask=nib.load(TRAIN_MASKS_PATH + 'CHUM-001.nii.gz').get_fdata()

pet_r = cv2.resize(test_image_pet, (test_image_ct.shape[0], test_image_ct.shape[1]) )
print(pet_r.shape)

# convert PET CT to 2 channel image
concat_img = np.moveaxis(np.concatenate((np.expand_dims(test_image_ct, axis=0), np.expand_dims(pet_r, axis=0)), axis=0), 0, -2)
print(concat_img.shape)



# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)
# ax1.imshow(np.moveaxis(concat_img[:,:,1], 2, 0)[55], cmap='gray')
# ax1.set_title('CT Image')
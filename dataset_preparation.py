import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import glob
import random
import shutil


# load image
def getimage(img_path, mask_path, folder_name, PET=False):
        # set file names and path
        # folder_name = img_path.split('\\')[-1]
        CT_img_path = img_path+'__CT.nii.gz'
        PET_img_path = CT_img_path.replace('__CT.nii.gz', '__PT.nii.gz')
        mask_path = mask_path + '.nii.gz'

        # load images and mask
        CT_image = nib.load(CT_img_path).get_fdata()
        PET_image = nib.load(PET_img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        print(CT_image.shape, PET_image.shape, mask.shape)

        # resize PET image
        pet_r = cv2.resize(PET_image, (CT_image.shape[0], CT_image.shape[1]) )
        # normalize PET image
        PET_image = (pet_r-np.min(pet_r))/(np.max(pet_r)-np.min(pet_r))

        return CT_image, PET_image, mask

def apply_window(img, width=500, center=40):
    # np.interp(a, (a.min(), a.max()), (-1, +1))

    # convert below window to black
    img[img<(center-(width/2))]=center-(width/2)
    # convert above window to white
    img[img>(center+(width/2))]=center+(width/2)

    # normalize image
    img_scaled = np.interp(img, (img.min(), img.max()), (0, +1))
    # print(img_scaled.shape)
    # print(np.min(img_scaled), np.max(img_scaled))
    return img

def separate_labels(mask):
    gtvp = mask[:]
    gtvn = mask[:]

    # 1 = GTVp, 2 = GTVn
    gtvp = np.where(gtvp <= 1, gtvp, 0)
    gtvn = np.where(gtvn>1, gtvn, 0)

    return gtvp

def save_images(imgs, trgt_path, img_name, number):
    
    # create suffix
    sfx=f"{number:04d}"
    # save CT and PET into images folder
    plt.imsave(f'{trgt_path}/train_images/{img_name}-{sfx}__CT.png', imgs[0], cmap='gray')
    plt.imsave(f'{trgt_path}/train_images/{img_name}-{sfx}__PT.png', imgs[1], cmap='gray')
    plt.imsave(f'{trgt_path}/train_masks/{img_name}-{sfx}.png', imgs[2], cmap='gray')
    # save label into masks folder
    return 

def display_single(image, idx=0, mk=True):
    if mk:
        plt.imshow(image[:,:,idx], cmap='gray')
        plt.show()
        return
    plt.imshow(image, cmap='gray')
    plt.show()


def display_multiple(ct, pet, mask, idx=32, volume=True):
    if volume:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15,30))
        ax[0][0].imshow(ct[:,:,idx], cmap='gray')
        ax[0][0].set_title('CT Image')
        ax[0][1].imshow(mask[:,:,idx], cmap='gray')
        ax[0][1].set_title('GT')
        ax[0][2].imshow(pet[:,:,idx], cmap='gray')
        ax[0][2].set_title('PET')
        # # contour
        ax[1][0].imshow(ct[:,:,idx], cmap='gray')
        ax[1][0].contour(mask[:,:,idx], colors='red')
        ax[1][0].set_title('GT on CT')
        ax[1][1].imshow(pet[:,:,idx], cmap='gray')
        ax[1][1].contour(mask[:,:,idx], colors='red')
        ax[1][1].set_title('GT on PET')
        ax[1][2].imshow(ct[:,:,idx], cmap='gray')
        ax[1][2].contour(pet[:,:,idx])
        ax[1][2].contour(mask[:,:,idx], colors='red')
        ax[1][2].set_title('All together')
        plt.show()
    else:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15,30))
        ax[0][0].imshow(ct, cmap='gray')
        ax[0][0].set_title('CT Image')
        ax[0][1].imshow(mask, cmap='gray')
        ax[0][1].set_title('GT')
        ax[0][2].imshow(pet, cmap='gray')
        ax[0][2].set_title('PET')
        # # contour
        ax[1][0].imshow(ct, cmap='gray')
        ax[1][0].contour(mask, colors='red')
        ax[1][0].set_title('GT on CT')
        ax[1][1].imshow(pet, cmap='gray')
        ax[1][1].contour(mask, colors='red')
        ax[1][1].set_title('GT on PET')
        ax[1][2].imshow(ct, cmap='gray')
        ax[1][2].contour(pet)
        ax[1][2].contour(mask, colors='red')
        ax[1][2].set_title('All together')
        plt.show()

def get_image_list(path):
    files = os.listdir(path)
    img_list = []
    for file in files:
        file = file.split('__')[0]
        img_list.append(file)
    return set(img_list)

def get_length(gpath, mpath):
    img_lst = sorted(get_image_list(gpath))
    img_dict = dict() 
    for m in img_lst:
        img_path = os.path.join(gpath, m)
        msk_path = os.path.join(mpath, m)
        try:
            ct, pet, mask = getimage(img_path, msk_path, m)
        except Exception as e:
            print(f'failed at {m}')
            print(e)
            continue
        img_dict[m] = [ct.shape[2], pet.shape[2], mask.shape[2]]
        print(m, img_dict[m])

def main():
    ''' convert ct pet and mask to png '''
    # path to data
    image_dir ='E:/Datasets/monte_carlo_segmentation/hecktor2022_training/hecktor2022/imagesTr'
    mask_dir = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/hecktor2022/labelsTr'
    trgt_dir = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg' 
    
    image_list = sorted(get_image_list(image_dir))
    print(f'found {len(image_list)} images ')
    # print(image_list)

    # get image path
    for img in image_list: 
        print(f'working on {img}')
    # img = image_list[0]
        img_path = os.path.join(image_dir, img)
        # print(img_path)
        
        # get mask path 
        msk_path = os.path.join(mask_dir, img)
        # print(msk_path)
        
        # get image
        try:
            ct, pet, mask = getimage(img_path, msk_path, img)
            assert ct.shape == mask.shape
            assert ct.shape == pet.shape
            assert ct.shape == mask.shape

            with open(trgt_dir+'/success.txt', 'a') as f:
                f.write(f'{img}\n')

        except Exception as e:
            print(f'failed at {img}')
            print(e)
            with open(trgt_dir+'/log.txt', 'a') as f:
                f.write(f'{img} -- failed \n')
            continue
        lengths = [ct.shape[2], pet.shape[2]]
        print(lengths[np.argmin(lengths)])

        for i in range(lengths[np.argmin(lengths)]):
            # CT windowing image
            # i=32
            scaled_ct = apply_window(ct[:,:,i], width=500, center=40)
            pet_slice = pet[:,:, i]
            # add function to extract each class from mask
            mask_slice = separate_labels(mask[:,:,i])
            
            
            # add function to save CT, PET, MASK as png
            # save_images((scaled_ct, pet_slice, mask_slice), trgt_dir, img, number=i)
        print(f'{img} saved')
            # display image
            # display_single(scaled_ct, idx=32, mk=False)
            # display_multiple(scaled_ct, pet_slice, mask_slice, idx=i)

def test_image_extracts(img_path, msk_path):
    # img = 'CHUV-042'
    # img_path=os.path.join(img_path, img)
    # msk_path=os.path.join(msk_path, img)
    
    # ct, pet, mask = getimage(img_path, msk_path, img)
    # display_multiple(ct, pet, mask, idx=10)

    img = 'CHUM-001-0032__CT.png'
    img_path=os.path.join(img_path, img)
    pet_path=img_path.replace('__CT.png', '__PT.png')
    msk_path=os.path.join(msk_path, img).replace('__CT.png', '.png')
    print(img_path)
    print(pet_path)
    print(msk_path)
    ct = np.array(Image.open(img_path).convert('L'))
    pet = np.array(Image.open(pet_path).convert('L'))
    mask = np.array(Image.open(msk_path).convert('L'))
    print(mask.shape)
    # show images 
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15,30))
    ax[0][0].imshow(ct, cmap='gray')
    ax[0][0].set_title('CT Image')
    ax[0][1].imshow(mask, cmap='gray')
    ax[0][1].set_title('GT')
    ax[0][2].imshow(pet, cmap='gray')
    ax[0][2].set_title('PET')
    # # contour
    ax[1][0].imshow(ct, cmap='gray')
    ax[1][0].contour(mask, colors='red')
    ax[1][0].set_title('GT on CT')
    ax[1][1].imshow(pet, cmap='gray')
    ax[1][1].contour(mask, colors='red')
    ax[1][1].set_title('GT on PET')
    ax[1][2].imshow(ct, cmap='gray')
    ax[1][2].contour(pet)
    ax[1][2].contour(mask, colors='red')
    ax[1][2].set_title('All together')
    plt.show()
    

def copy_ct():
    ''' copy only ct images from root folder to target folder '''
    import shutil

    image_dir ='E:/Datasets/monte_carlo_segmentation/hecktor2022_training/hecktor2022/imagesTr'
    mask_dir = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/hecktor2022/labelsTr'
    # target folders
    image_trgt ='E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/imagesTr'
    mask_trgt = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/labelsTr'

    all_imgs = os.listdir(image_dir)
    print(f'found {len(all_imgs)} PET/CT images, preparing to copy')
    
    img_nms = sorted(set([x.replace(str(x[-11:]), '') for x in all_imgs]))
    
    print(len(img_nms))
    for im in img_nms:
        print(f'copying {im}')
        src_im = os.path.join(image_dir, im+'__CT.nii.gz')
        src_msk = os.path.join(mask_dir, im+'.nii.gz' )

        trgt_im = os.path.join(image_trgt, im+'__CT.nii.gz')
        trgt_msk = os.path.join(mask_trgt, im+'.nii.gz' )

        shutil.copy2(src_im, trgt_im)
        shutil.copy2(src_msk, trgt_msk)
        
    
    print('copy finished')


def split_dataset(data_dir, 
                    label_dir, 
                    train_data_dir ='', 
                    train_label_dir='', 
                    val_data_dir='', 
                    val_label_dir='',
                    test_data_dir='', 
                    test_label_dir='', 
                    train_split=0.75, 
                    val_split=0.15, 
                    seed=42):
    
    ''' split and move CT and labels from src to trgt '''
    # Set the random seed
    random.seed(seed)

    # Get a list of all image filenames in the data directory
    data_filenames = os.listdir(data_dir)
    data_filenames = [filename for filename in data_filenames if filename.endswith('.nii.gz') ]
    
    # Get a list of all label filenames in the label directory
    # label_filenames = os.listdir(label_dir)
    # label_filenames = [filename.replace('__CT.nii.gz', '.nii.gz') for filename in data_filenames if filename.endswith('.nii.gz') ]
    
    # Shuffle the filenames
    random.shuffle(data_filenames)

    # Calculate the number of images for each split
    num_train = int(len(data_filenames) * train_split)
    num_val = int(len(data_filenames) * val_split)
    print(num_train, num_val)

    # Split the filenames into the train, validation, and test sets
    train_data_filenames = data_filenames[:num_train]
    val_data_filenames = data_filenames[num_train:num_train + num_val]
    test_data_filenames = data_filenames[num_train + num_val:]

    print(train_data_filenames[0])
    # Copy the images and labels to the train, validation, and test directories
    for data_filename in data_filenames:
        print(f'moving {data_filename}')
        label_filename = data_filename.replace('__CT.nii.gz', '.nii.gz')
        if data_filename in train_data_filenames:
            shutil.move(f'{data_dir}/{data_filename}', f'{train_data_dir}/{data_filename}')
            shutil.move(f'{label_dir}/{label_filename}', f'{train_label_dir}/{label_filename}')
        elif data_filename in val_data_filenames:
            shutil.move(f'{data_dir}/{data_filename}', f'{val_data_dir}/{data_filename}')
            shutil.move(f'{label_dir}/{label_filename}', f'{val_label_dir}/{label_filename}')
        elif data_filename in test_data_filenames:
            shutil.move(f'{data_dir}/{data_filename}', f'{test_data_dir}/{data_filename}')
            shutil.move(f'{label_dir}/{label_filename}', f'{test_label_dir}/{data_filename}')
        
    print('test train split completed')


if __name__ == '__main__':
    image_dir ='E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/train_images'
    mask_dir = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/train_masks'
    # main()
    # test_image_extracts(image_dir, mask_dir)
    # get_length(image_dir, mask_dir)
    # copy_ct()


    ''' train test split for CT only '''
    # # target folders
    # image_root ='E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/imagesTr'
    # mask_root = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/labelsTr'
    # image_train ='E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/train_images'
    # mask_train = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/train_labels'
    # image_valid ='E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/valid_images'
    # mask_valid = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/valid_labels'
    # image_test ='E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/test_images'
    # mask_test = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only/test_tabels'
    # split_dataset(image_root,
    #              mask_root,
    #              image_train,
    #              mask_train,
    #              image_valid,
    #              mask_valid,
    #              image_test,
    #              mask_test )
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        
        # change mask values from 255 t0 0-1
        mask[mask==255.0] = 1

        # perform data augmentation using the albemntations library 
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) 
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

def test():
    img = CarvanaDataset(   image_dir='./data/train_images',
                            mask_dir= './data/train_masks')
    image, mask = img.__getitem__(1)
    print(f'found image: {image.shape}')
    fig, ax = plt.subplots(ncols=2, nrows=1)
    ax[0].imshow(image)
    ax[1].imshow(mask, cmap='gray')
    plt.show()


if __name__ == "__main__":
    test()

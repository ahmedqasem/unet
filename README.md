# unet
implementation of U-Net on Hecktor 2022 Data

## data structure
```
data
    |
    |__train_images
    |   |__ patient-001 
    |   |   |__ patient-001__CT.nii.gz
    |   |   |__ patient-001__PT.nii.gz
    |   |
    |   |__ patient-002
    |       |__ patient-002__CT.nii.gz
    |       |__ patient-002__PT.nii.gz
    |__train_masks
    |   |__ patient-001.nii.gz
    |   |__ patient-002.nii.gz
    |
    |__valid_images
    |   |__ patient-001 
    |   |   |__ patient-001__CT.nii.gz
    |   |   |__ patient-001__PT.nii.gz
    |   |
    |   |__ patient-002
    |       |__ patient-002__CT.nii.gz
    |       |__ patient-002__PT.nii.gz
    |__valid_masks
    |   |__ patient-001.nii.gz
    |   |__ patient-002.nii.gz
```
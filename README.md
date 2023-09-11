# AQC_left_heart_segmentation

This repository was created for the MICCAI STACOM workshop paper "Automated quality-controlled left heart segmentation from 2D echocardiography". 

Bram Geven - bgev545@aucklanduni.ac.nz

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## File structure


```
DatasetName/ 
├── Patient_0001
├── Patient_0002
├── ... 
```

Within each patient folder, the following structure is expected:

```
DatasetName/Patient_0001/
├── DICOM_files (optional)
├── images (optional)
├── segmentations_original
└── segmentations_post_processed

```

## Segmentations
The code in this repository was written to analyse segmentations of apical two-chamber and apical four-chamber views with a focus on the left side of the heart. These segmentations must contain labels for the left ventricular cavity (LV), myocardium (MYO) and left atrium (LA), each with a unique label value. The LV has label value 1, the MYO has label value 2 and the LA has label value 3. The segmentations can be provided in any format that is supported by [SimpleITK](https://simpleitk.org/).

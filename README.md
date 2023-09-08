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

# Automated quality controlled left heart segmentation and analysis from 2D echocardiography. 
This repository was created for the automated analysis of 2D echocardiography images, with the addition of a quality control and calculation of several routine clinical indices. The workflow is described in the conference paper "Automated quality-controlled left heart segmentation from 2D echocardiography", which was accepted for publication at the MICCAI-STACOM 2023 workshop. The code in this repository was used to generate the results in the paper.

- Bram Geven - bgev545@aucklanduni.ac.nz
- Debbie Zhao - debbie.zhao@auckland.ac.nz
- Stephen Creamer - stephen.creamer@auckland.ac.nz

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Warning
It is seen that the way of implementation of the Simpson's biplane method has a large impact on the calculated values. Therefore, we do not guarantee that the method that is used in this repository is optimal. 

## Installing libraries
In the Terminal or Command Prompt, change the directory to the repository: 

```
cd git_repository_location/AQC-left-heart-segmentation
```

Then use the requirements file to install the required libraries:

```
pip install -r requirements.txt
```

## File structure
```
path_to_data/ 
├── participant_id_1
├── participant_id_2
├── ... 
```

Within each participant folder, the following structure is expected:

```
path_to_data/participant_id/
├── DICOM_files (optional)
├── images (optional)
├── segmentations_original
└── segmentations_post_processed (optional)

```

Please note that it is possible to run the workflow without the DICOM files and images, but without information about e.g. pixel spacing, the clinical values cannot be calculated. 

## Segmentations
The code in this repository was written to analyse segmentations of apical two-chamber (A2CH) and apical four-chamber (A4CH) views with a focus on the left side of the heart. These segmentations must contain labels for the left ventricular cavity (LV), myocardium (MYO) and left atrium (LA), each with a unique label value. The LV has label value 1, the MYO has label value 2 and the LA has label value 3, background has value 0.

## Test data
Segmentations of 1 cardiac cycle of 1 participant and the population priors of the area-time curves of the LV and LA are publicly available, and accessible via: https://auckland.figshare.com/articles/dataset/AQC-left-heart-segmentation_zip/24376909.

This folder contains of the following:
    
* area_time_curves: the population priors of the area-time curves of the LV and LA. 
* patient_0001: the segmentations of 1 cardiac cycle of 1 participant. 

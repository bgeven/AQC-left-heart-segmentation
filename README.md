# Automated quality controlled left heart segmentation and analysis from 2D echocardiography. 

This repository was created for the automated analysis of 2D echocardiography images, with the addition of a quality control and calculation of several routine clinical indices. The workflow is described in the conference paper "Automated quality-controlled left heart segmentation from 2D echocardiography", which was accepted for publication at the MICCAI-STACOM 2023 workshop. The code in this repository was used to generate the results in the paper.

- Bram Geven - bgev545@aucklanduni.ac.nz
- Debbie Zhao - debbie.zhao@auckland.ac.nz
- Stephen Creamer - stephen.creamer@auckland.ac.nz

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
└── segmentations_post_processed (optional)

```

Please note that it is possible to run the workflow without the DICOM files and images, but without information about e.g. pixel spacing and R-wave peaks, the clinical values cannot be calculated. 

## Segmentations
The code in this repository was written to analyse segmentations of apical two-chamber (A2CH) and apical four-chamber (A4CH) views with a focus on the left side of the heart. These segmentations must contain labels for the left ventricular cavity (LV), myocardium (MYO) and left atrium (LA), each with a unique label value. The LV has label value 1, the MYO has label value 2 and the LA has label value 3, background has value 0.

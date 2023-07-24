# AQC_left_heart_segmentation

This repository was created for the MICCAI STACOM workshop paper "Automated quality-controlled left heart segmentation from 2D echocardiography". 

Bram Geven - b.w.m.geven@student.tue.nl


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
├── images
|   ├── a2ch 
|   └── a4ch
└── segmentations
    ├── a2ch
    └── a4ch 
```

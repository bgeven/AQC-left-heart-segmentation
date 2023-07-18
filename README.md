# AQC_left_heart_segmentation

This repository was created for the MICCAI STACOM workshop paper "Automated quality-controlled left heart segmentation from 2D echocardiography". 

Bram Geven - b.w.m.geven@student.tue.nl


## File structure

```
Dataset/ <br>
├── Patient_0001 <br>
├── Patient_0002 <br>
├── ... <br>
```

Within each patient folder, the following structure is expected:

```
Dataset/Patient_0001/ <br>
├── DICOM_files # optional <br>
├── images <br>
|   ├── a2ch <br>
|   └── a4ch <br>
└── segmentations <br>
    ├── a2ch <br>
    └── a4ch <br>
```

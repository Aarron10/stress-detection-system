# Dataset Setup Guide for Stress Detection

To train the MobileNetV2 model, you need to organize your images into a standard directory structure.

## 1. Directory Structure
Create a folder named `dataset` in the project root. Inside it, create `train` and `val` folders. Inside those, create folders for each class (Label).

**Example:**
```
stress_detector/
├── dataset/
│   ├── train/
│   │   ├── Focused/    <-- Place "Focused" images here
│   │   │   ├── image01.jpg
│   │   │   └── ...
│   │   └── Stressed/   <-- Place "Stressed" images here
│   │       ├── image02.jpg
│   │       └── ...
│   └── val/            <-- Validation set (approx 20% of data)
│       ├── Focused/
│       └── Stressed/
```

## 2. How to Collect Data?
- **Focused**: Record yourself or others working normally. Crop the faces.
- **Stressed**: Record yourself simulating stress (figness, frowning, etc.). Crop the faces.
- **Important**: The model extracts features from *faces*. Ensure your images are crops of faces, similar to what the app sees (the app crops faces before processing).

## 3. Running Training
Once your data is ready, run:
```bash
python train.py --data_dir dataset --epochs 10
```
This will save `stress_mobilenet.pth` which can be loaded into the app later.

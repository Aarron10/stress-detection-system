# Kaggle Dataset Setup Guide

To train the model that outputs three separate scores (Stressed %, Focused %, Distracted %), organize your images into folders.

## 1. Directory Structure

Upload your folder to Kaggle. It should be structured like this:

```
dataset/
├── train/
│   ├── Focused/        <-- Images of focused people
│   ├── Stressed/       <-- Images of stressed people
│   └── Distracted/     <-- Images of distracted people
└── val/                <-- (Optional but recommended) ~20% of your images
    ├── Focused/
    ├── Stressed/
    └── Distracted/
```

## 2. Training
Run the script `train_multi_label.py`. 

It will:
1.  Read images from these folders.
2.  Train a MobileNetV2 model.
3.  Save `stress_multilabel.pth`.

## 3. The Result
Even though you train with separate folders, the model learns independent features. When you run it in the app later, it will be able to say:
> "This person is 80% Focused AND 20% Stressed"
Instead of forcing just one label.

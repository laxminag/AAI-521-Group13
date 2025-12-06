# Waste Classification Using Deep Learning (TrashNet Dataset)  
### *Final Team Project – Computer Vision*

## Project Overview
This project implements a computer vision system to classify waste items into six categories—cardboard, glass, metal, paper, plastic, and trash—using deep learning and hybrid machine-learning approaches. The work was completed as part of the Final Team Project for a university course on Computer Vision.

The project includes:
- Dataset selection and preprocessing  
- Model development using PyTorch  
- Evaluation of multiple architectures  
- Error analysis and interpretation  
- A webcam-based real-time prediction app built with Gradio  
- A full technical report in APA format and a recorded presentation

This project follows the STAR framework (Situation, Task, Action, Results) to explain the problem, approach, challenges, and outcomes.

## Problem Statement
Manual waste sorting is labor-intensive, inconsistent, and often unsafe. Automating this process with computer vision can significantly enhance recycling efficiency and reduce contamination.

Goal: Develop a robust machine-learning system that predicts the material type of a waste item from a single RGB image.

## Dataset: TrashNet
- Source: Stanford TrashNet project (mirrored on Kaggle)
- ~2,500 images across six categories:
  - cardboard
  - glass
  - metal
  - paper
  - plastic
  - trash

Dataset characteristics:
- Clean, centered images  
- Plain background  
- Ideal for transfer learning  
- Imbalanced classes  

## Data Preprocessing
Transformations:
- Resize to 224×224  
- Normalize using ImageNet mean/std  
- Custom split: 70% train / 15% val / 15% test (stratified)

Augmentation:
- Horizontal flip  
- Rotation  
- Resized crop  
- Color jitter  
- Affine transforms  

## Models Developed

### Baseline Model – MobileNetV3 (Transfer Learning)
- Pretrained on ImageNet  
- Final FC layer replaced  
- Validation accuracy ~89–90%

### Improved Model – Enhanced Regularization
Enhancements:
- Stronger augmentation  
- Class-weighted loss  
- LR scheduler  

Accuracy remained ~87–88% due to dataset limitations and baseline saturation.

### Hybrid Model – Deep Features + Logistic Regression
Pipeline:
1. Extract features from MobileNetV3  
2. Train Logistic Regression  
3. Achieved ~85–88% accuracy  

## Results Summary
| Model | Validation Accuracy | Notes |
|-------|---------------------|------|
| Baseline | ~89–90% | Strong performance |
| Improved | ~87–88% | Lower overfitting |
| Hybrid | ~85–88% | Lightweight & interpretable |

## Error Analysis
Common misclassification causes:
- Material similarity (plastic ↔ glass)  
- Mixed materials  
- Unusual angles  
- Class imbalance  

## Webcam Demo (Gradio)
A lightweight app allows real-time predictions using a webcam in Colab.

## Tech Stack
- Python 3.10  
- Google Colab  
- PyTorch  
- Torchvision  
- Scikit-learn  
- Gradio  
## Future Work
- Collect more real-world data
- Add object detection
- Try Vision Transformers
- Deploy on mobile devices
- Build a smart recycling bin prototype

## References
- TrashNet: https://github.com/garythung/trashnet
- MobileNetV3 (Howard et al., 2019)
- PyTorch: https://pytorch.org
- Scikit-learn: https://scikit-learn.org

## Acknowledgements
- This project was completed as part of the Final Team Project for the Computer Vision course.

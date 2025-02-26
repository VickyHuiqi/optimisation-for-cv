# Optimization for Computer Vision
**February 2025**

## Introduction 
This repository contains the code and report for the project **Optimization for Computer Vision** completed as part of the **Master of Mathematics and AI** program at **Paris-Saclay University**. For this project we aimed to analyze an article focused on Optimization and computer vision, write a report and implement some of the algorithms presented. I chose to work on the paper: *YOLO9000: Better, Faster, Stronger* written by Joseph Redmon and Ali Farhadi in 2016 [1]. Key contributions include:

- Implementation of anchor box clustering using a custom IoU metric
- Validation on a new dataset (African wildlife) [2]

## Structure
```bash
├── notebooks/                     # Jupyter notebooks for key steps  
│   ├── 0-Data_Preparation.ipynb   # Splitting datasets, resizing images/boxes  
│   ├── 1-KMeans_Clustering.ipynb  # Anchor box clustering with custom IoU metric  
│   ├── 2-Training.ipynb           # YOLOv2 model training  
│   └── 3-Testing.ipynb            # Model evaluation on test data  
├── data/                          # Dataset directory  
├── best_model/                    # Directory for saved trained models  
├── utils/                         # Utility scripts  
│   ├── clustering.py              # Functions for anchor box clustering  
│   ├── dataset.py                 # Dataset handling functions  
│   ├── loadweights.py             # Script for loading pretrained model weights
│   ├── loss.py                    # Loss function implementations  
│   ├── metrics.py                 # Evaluation metrics
├── weights/                       # Original pretrained weights from the authors 
├── paper.pdf                      # Original research paper
└── report.pdf                     # My report
```
Detailed explanations and results can be found in the **report.pdf** located within this repository.

## Running the Code  
The notebook 0-Data_Preparation.ipynb needs to be executed only if you wish to preprocess the data yourself. Otherwise, you can directly use the provided dataset. The other notebooks can be run as they are.

## Required Python Libraries  
Here’s the list of Python libraries required to run the code:
 
- `numpy`  
- `seaborn`  
- `matplotlib`  
- `PIL`
- `torch`
- `torchvision`    
- `os`  

Ensure that all the necessary libraries are installed in your Python environment before running the code. You can install them using `pip`:

```bash
pip install numpy seaborn matplotlib torch torchvision pillow
```

## Author
- **Huiqi Vicky ZHENG**

[1]: https://arxiv.org/abs/1612.08242 
[2]: https://www.kaggle.com/datasets/biancaferreira/african-wildlife

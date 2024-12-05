
# Fish Identification using Siamese Networks

## Overview
This project implements a Siamese Network for fish identification and similarity comparison. The model was trained on the Fish4Knowledge dataset and evaluated for classification and similarity prediction tasks. The system leverages a GPU for efficient training and inference.

## Features
- **Siamese Network Architecture**: Extracts embeddings for image pairs and predicts similarity using a contrastive loss function.
- **Dataset Preparation**: Automatically loads, preprocesses, and augments images and their corresponding masks.
- **Balanced Training**: Oversampling is used to address class imbalance in the dataset.
- **Inference Pipeline**: Predicts similarity between images and provides meaningful similarity scores.

## Hardware and Framework
- **GPU**: This project was trained using a GPU from the **University of Agder (UiA)**, which significantly accelerated training.
- **Framework**: PyTorch, chosen for its flexibility and extensive library support.
- **Other Tools**: OpenCV, TQDM, and torchvision for data handling, augmentation, and visualization.

## Dataset
The Fish4Knowledge dataset was used, containing:
- 27,370 images grouped into 23 fish species.
- Each image is paired with a mask for object detection tasks.

The dataset was retrieved from [Fish4Knowledge](https://homepages.inf.ed.ac.uk/rbf/fish4knowledge/GROUNDTRUTH/RECOG/).

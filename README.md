# Machine_Learning_CNN_Architechture


# Convolutional Neural Network (CNN) for Image Classification

This repository contains a Python implementation of a Convolutional Neural Network (CNN) for image classification. It includes training and evaluation scripts for two popular datasets: MNIST and CIFAR-10.

## Introduction

Convolutional Neural Networks (CNNs) are a class of deep learning models that have proven to be highly effective for image classification tasks. This project demonstrates the use of CNNs to classify images from the MNIST and CIFAR-10 datasets.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm
- numpy

Install the required dependencies using pip:

pip install -r requirements.txt


To train the CNN model on a dataset run the following command:

python main.py --dataset DATASET_NAME

Replace DATASET_NAME with either "MNIST" or "CIFAR10" to choose the dataset for training. You can also specify the GPU for training using the --gpu flag.

To evaluate the trained model, run:

python main.py --dataset DATASET_NAME --gpu 1

This will test the model on the specified dataset and display accuracy and runtime information.

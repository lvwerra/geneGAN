# Adversarial Training for Precision Oncology

## Overview
This repo showcases to GAN approaches to precision oncology. In each folder the scripts and notebooks to train a model and to apply a trained model are provided.

## Installation
To install the necessary libraries run:
```
pip install -r requirements.txt
```

## geneGAN
With a dataset of co-occurring gene pairs, train a GAN to learn the pair distribution in order to generate and discriminate co-occurring gene pairs.

## treatGAN
With two dataset consisting of a patients disease, demographics and genetic information and the corresponding treatments train a conditional GAN to produce suitable treatment suggestions based on the patient information.
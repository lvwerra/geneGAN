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

## References
This work is based on the [medGAN](https://github.com/mp2893/medgan) approach introduced in the following [paper](https://arxiv.org/abs/1703.06490):

	Generating Multi-label Discrete Patient Records using Generative Adversarial Networks
	Edward Choi, Siddharth Biswal, Bradley Malin, Jon Duke, Walter F. Stewart, Jimeng Sun  
	Machine Learning for Healthcare (MLHC) 2017
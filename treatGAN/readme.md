# treatGAN

The treatGAN architecture is similar to the proposed geneGAN architecture with its parameters explained in the geneGAN notebooks. The difference being, that the GAN is not trained to produce gene pairs but suitable treatments. Additionally, the generator gets additional condition vector with the patients disease, demographics and genetic information encoded.

## Training Data
The training data consists of two files \*\_treat.npy and \*\_cond.npy with \[dim x n_samples] numpy arrays encoding treatments and patient conditions/demographics.

## Training
With the treatgan.py script a GAN can be trained and the model is saved.

## Applying the model
To apply the model use the treatgan_model.py module. For more detail on how to use this module consult the treatgan_apply.ipynb notebook.
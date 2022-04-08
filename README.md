
# Semantic Segmentation using Multimodal Fully Convolutional Networks in Keras
This project involves segmentation of various components of a scene using deep learning & Computer Vision. The multimodal nature of the input data - for instance, RGB image, Depth Image, NIR etc - cumulatively give superior results as opposed to individual modalities on their own.

The Fully Convolutional Network (FCN-32s) is implemented using Keras and trained to predict semantic segments of forest-like images with rgb & nir_color input images.

For a closer look at the project, check the [presentation link](https://docs.google.com/presentation/d/1z8-GeTXvSuVbcez8R6HOG1Tw_F3A-WETahQdTV38_uc/edit?usp=sharing).

## Post-Download Steps:
After downloading the dataset, carry out the following steps before training the models:
1. Run preprocess/process.sh - This renames the images.
2. Run preprocess/text_file_gen.py - This generates txt files for train,val,test used in data generator.
3. Run preprocess/aug_gen.py - This generates augmented image files ahead of the training - exercising caution as dynamic augmentation in runtime can be slow and often halt the training process.

## Files Description:

The files are grouped under two architectures - with and without augmentation & dropout.

- Improved Architecture includes:

1. late_fusion_improveed.py
2. late_fusion_improved_predict.py
3. late_fusion_improved_saved_model.hdf5

- Old Architecture includes:

4. late_fusion_old.py
5. late_fusion_old_predict.py()
6. late_fusion_improved_saved_model.hdf5

## Architecture:
![Alt text](/Arc.png)
Consult the first two models in this [Reference Link](http://deepscene.cs.uni-freiburg.de/index.html) for an indepth study of the architecture.

## Dataset:
![Alt text](/DS.png)
Make use of [Freiburg Forest Multimodal/Spectral Annotated](http://deepscene.cs.uni-freiburg.de/index.html#datasets) dataset. Note that due to a small dataset size, there's a risk of overfitting. To counter this, we use image augmentation by geometrically transforming images and adding these to the dataset pre-training.
![Alt text](/Aug.png)

## Training Details:
The loss function used is Categorical Cross Entropy. For optimization, we apply Stochastic Gradient Descent with a learning rate of 0.008, momentum of 0.9, and decay of 1e-6.

## Results:
![Alt text](/Res.png)

The files under `Deepscene` in this repository are accurate replications of the architectures as described on the Deepscene website.
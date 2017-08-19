# Fully convolutional neural network (FCN) for semantic segmentation tensorflow simple implementation.

This is a simple implementation of a fully convolutional neural network (FCN). The net is based on fully convolutional neural net described in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf).  The code is based on [FCN implementation](https://github.com/shekkizh/FCN.tensorflow)  by Sarath Shekkizhar with MIT license. The net is initialized using the pre-trained VGG16 model by Marvin Teichmann.

## Details input/output
The input for the net (Figure 1) are RGB image,
The net produces pixelwise annotation as a matrix in size of the image with the value of each pixel is the pixel label (This should be the input in training).

## Requirements
This network was run and trained with Python 3.6  Anaconda package and Tensorflow 1.1. The training was done using Nvidia GTX 1080, on Linux Ubuntu 16.04.

## Setup
1) Download the code from the repository.
2) Download pretrained vgg16 net and put in the /Model_Zoo subfolder in the main code folder. A pre-trained vgg16 net can be download from here[https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing] or from here [ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy]

## Tutorial

### Instructions for training (in TRAIN.py):
Code for training the net available in TRAIN.py
1) Set folder of train images in Train_Image_Dir
2) Set folder for ground truth labels in Label_DIR
3) The Label Maps should be saved as png image with the same name as the corresponding image and png ending
4) Download pretrained [vgg16](ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy) model and put in model_path (should be done automatically if you have internet connection)
5) Set number of classes number in NUM_CLASSES
6) If you are interested in using validation set during training, set UseValidationSet=True and the validation image folder to Valid_Image_Dir (assume that the labels for the validation image are also in  Label_Dir)

### Instructions for predicting pixelwise annotation using trained net (in Inference.py)
Code for predicting using a trained net is available in: Inference.py
1) Make sure you have trained model in logs_dir (See Train.py for creating trained model)
2) Set the Image_Dir to the folder where the input image for prediction located.
3) Set number of classes in NUM_CLASSES
4) Set Pred_Dir the folder where you want the output annotated images to be saved
5) Run script

### Evaluating net performance using intersection over union (IOU):
Code for evaluating net intersection over union appear in: (Evaluate_Net_IOU.py)
1) Make sure you have trained model in logs_dir (See Train.py for creating trained model)
2) Set the Image_Dir to the folder where the input images for prediction are located
3) Set folder for ground truth labels in Label_DIR. The Label Maps should be saved as png image with the same name as the corresponding image and png ending
4) Set number of classes number in NUM_CLASSES
5) Run script

## Supporting datasets
The net was tested on a dataset of annotated images of materials in glass vessels. 
This dataset can be downloaded from (here)[https://drive.google.com/file/d/0B6njwynsu2hXRFpmY1pOV1A4SFE/view?usp=sharing]

MIT Scene Parsing Benchmark with over 20k pixelwise annotated images can also be used for training and can be download from (here)[http://sceneparsing.csail.mit.edu/]

   


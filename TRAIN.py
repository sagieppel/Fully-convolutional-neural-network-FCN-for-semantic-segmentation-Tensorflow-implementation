# Train fully convolutional neural net for sematic segmentation
# Instructions:
# 1) Set folder of train images in Train_Image_Dir
# 2) Set folder for ground truth labels in Train_Label_Dir
#    The Label Maps should be saved as png image with same name as the corresponding image and png ending
# 3) Download pretrained vgg16 model and put in model_path (should be done autmatically if you have internet connection)
#    Vgg16 pretrained Model can be download from ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
#    or https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing
# 4) Set number of classes number in NUM_CLASSES
# 5) If you are interested in using validation set during training, set UseValidationSet=True and the validation image folder to Valid_Image_Dir and validation labels to Valid_Labels_Dir
# 6) Run scripty
##########################################################################################################################################################################
import tensorflow as tf
import numpy as np
import Data_Reader
import BuildNetVgg16
import os
import CheckVGG16Model
import scipy.misc as misc
#...........................................Input and output folders.................................................
Train_Image_Dir="Data_Zoo/Materials_In_Vessels/Train_Images/" # Images and labels for training
Train_Label_Dir="Data_Zoo/Materials_In_Vessels/LiquidSolidLabels/"# Annotetion in png format for train images and validation images (assume the name of the images and annotation images are the same (but annotation is always png format))
UseValidationSet=False# do you want to use validation set in training
Valid_Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All/"# Validation images that will be used to evaluate training
Valid_Labels_Dir="Data_Zoo/Materials_In_Vessels/LiquidSolidLabels/"#  (the  Labels are in same folder as the training set)
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
learning_rate=1e-5 #Learning rate for Adam Optimizer
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it
#-----------------------------Other Paramters------------------------------------------------------------------------
TrainLossTxtFile=logs_dir+"TrainLoss.txt" #Where train losses will be writen
ValidLossTxtFile=logs_dir+"ValidationLoss.txt"# Where validation losses will be writen
Batch_Size=2 # Number of files per training iteration
Weight_Loss_Rate=5e-4# Weight for the weight decay loss function
MAX_ITERATION = int(100010) # Max  number of training iteration
NUM_CLASSES = 4#Number of class for fine grain +number of class for solid liquid+Number of class for empty none empty +Number of class for vessel background
######################################Solver for model   training#####################################################################################################################
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

################################################################################################################################################################################
################################################################################################################################################################################
def main(argv=None):
    tf.reset_default_graph()
    keep_prob= tf.placeholder_with_default([1.0], shape=(1,), name="keep_probability") #Dropout probability : tflite cannot input shape 0
#.........................Placeholders for input image and labels...........................................................................................
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image") #Input image batch (batchnum, height, width, RGB)
    GTLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="GTLabel")#Ground truth labels for training
  #.........................Build FCN Net...............................................................................................
    Net =  BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path) #Create class for the network
    Net.build(image, NUM_CLASSES,keep_prob)# Create the net and load intial weights
#......................................Get loss functions for neural net work  one loss function for each set of label....................................................................................................
    Loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(GTLabel, squeeze_dims=[3]), logits=Net.Prob,name="Loss")))  # Define loss function for training
   #....................................Create solver for the net............................................................................................
    trainable_var = tf.trainable_variables() # Collect all trainable variables for the net
    train_op = train(Loss, trainable_var) #Create Train Operation for the net
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
    TrainReader = Data_Reader.Data_Reader(Train_Image_Dir,  GTLabelDir=Train_Label_Dir,BatchSize=Batch_Size) #Reader for training data
    if UseValidationSet:
        ValidReader = Data_Reader.Data_Reader(Valid_Image_Dir,  GTLabelDir=Valid_Labels_Dir,BatchSize=Batch_Size) # Reader for validation data
    sess = tf.Session() #Start Tensorflow session
# -------------load trained model if exist-----------------------------------------------------------------
    print("Setting up Saver...")
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init) #Initialize variables
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    
    saver_def = saver.as_saver_def()
    
    print('Run this operation to initialize variables     : ', init.name)
    print('Run this operation for a train step            : ', train_op.name)
    print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
    print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
    print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)
    
    with open('fcn.pb', 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())
    exit()
    
#--------------------------- Create files for saving loss----------------------------------------------------------------------------------------------------------
    f = open(TrainLossTxtFile, "w")
    f.write("Iteration\tloss\t Learning Rate="+str(learning_rate))
    f.close()
    if UseValidationSet:
       f = open(ValidLossTxtFile, "w")
       f.write("Iteration\tloss\t Learning Rate=" + str(learning_rate))
       f.close()
#..............Start Training loop: Main Training....................................................................
    for itr in range(MAX_ITERATION):
        Images,  GTLabels =TrainReader.ReadAndAugmentNextBatch() # Load  augmeted images and ground true labels for training
        feed_dict = {image: Images,GTLabel:GTLabels, keep_prob: 0.5}
        sess.run(train_op, feed_dict=feed_dict) # Train one cycle
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
        if itr % 50 == 0 and itr>0:
            print("Saving Model to file in "+logs_dir)
            saver.save(sess, logs_dir + "model.ckpt", itr) #Save model
#......................Write and display train loss..........................................................................
        if itr % 10==0:
            # Calculate train loss
            feed_dict = {image: Images, GTLabel: GTLabels, keep_prob: 1}
            TLoss=sess.run(Loss, feed_dict=feed_dict)
            print("Step "+str(itr)+" Train Loss="+str(TLoss))
            #Write train loss to file
            with open(TrainLossTxtFile, "a") as f:
                f.write("\n"+str(itr)+"\t"+str(TLoss))
                f.close()
#......................Write and display Validation Set Loss by running loss on all validation images.....................................................................
        if UseValidationSet and itr % 2000 == 0:
            SumLoss=np.float64(0.0)
            NBatches=np.int(np.ceil(ValidReader.NumFiles/ValidReader.BatchSize))
            print("Calculating Validation on " + str(ValidReader.NumFiles) + " Images")
            for i in range(NBatches):# Go over all validation image
                Images, GTLabels= ValidReader.ReadNextBatchClean() # load validation image and ground true labels
                feed_dict = {image: Images,GTLabel: GTLabels ,keep_prob: 1.0}
                # Calculate loss for all labels set
                TLoss = sess.run(Loss, feed_dict=feed_dict)
                SumLoss+=TLoss
                NBatches+=1
            SumLoss/=NBatches
            print("Validation Loss: "+str(SumLoss))
            with open(ValidLossTxtFile, "a") as f:
                f.write("\n" + str(itr) + "\t" + str(SumLoss))
                f.close()


##################################################################################################################################################
main()#Run script
print("Finished")

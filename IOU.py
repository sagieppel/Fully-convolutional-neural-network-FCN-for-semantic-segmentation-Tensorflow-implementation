##Calculate Intersection Over Union Score for predicted layer
import numpy as np
import scipy.misc as misc

def GetIOU(Pred,GT,NumClasses,ClassNames=[], DisplyResults=False): #Given A ground true and predicted labels return the intersection over union for each class
    # and the union for each class
    ClassIOU=np.zeros(NumClasses)#Vector that Contain IOU per class
    ClassWeight=np.zeros(NumClasses)#Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses): # Go over all classes
        Intersection=np.float32(np.sum((Pred==GT)*(GT==i)))# Calculate intersection
        Union=np.sum(GT==i)+np.sum(Pred==i)-Intersection # Calculate Union
        if Union>0:
            ClassIOU[i]=Intersection/Union# Calculate intesection over union
            ClassWeight[i]=Union

    #------------Display results-------------------------------------------------------------------------------------
    if DisplyResults:
       for i in range(len(ClassNames)):
            print(ClassNames[i]+") "+str(ClassIOU[i]))
       print("Mean Classes IOU) "+str(np.mean(ClassIOU)))
       print("Image Predicition Accuracy)" + str(np.float32(np.sum(Pred == GT)) / GT.size))
    #-------------------------------------------------------------------------------------------------

    return ClassIOU, ClassWeight






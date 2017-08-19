import TensorflowUtils
import os
#-----------------------------------------Check if pretrain vgg16 models and data are availale----------------------------------------------------
def CheckVGG16(model_path): # Check if pretrained vgg16 model avialable and if not try to download it
    TensorflowUtils.maybe_download_and_extract(model_path.split('/')[0], "ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy")  # If not exist try to download pretrained vgg16 net for network initiation
    if not os.path.isfile(model_path):
       print("Error: Cant find pretrained vgg16 model for network initiation. Please download model from:")
       print("ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy")
       print("Or from:")
       print("https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing")
       print("and place in the path pointed by model_path")

    #This can be download manualy from:"https://drive.google.com/drive/folders/0B6njwynsu2hXcDYwb1hxMW9HMEU" or from ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
# and placed in the /Model_Zoo folder in the code dir


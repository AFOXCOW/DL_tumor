import numpy as np 
import cv2
import glob
import os

def zero_pic(img):
    if img.max()==0 and img.min()==0:
        return True
    else:
        return False
#the opencv image channel is BGR
def blue_pic(img):
    B_channel = img[:,:,0]
    G_channel = img[:,:,1]
    R_channel = img[:,:,2]

    reverse_B = 255-B_channel
    if zero_pic(reverse_B) and zero_pic(G_channel) and zero_pic(R_channel):
        return True
    else:
        return False

classes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
           'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD',
           'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
for i in range(10):
    img_dir = './heatmaps/gradcam/'+str(i)
    for cs in classes:
        file_paths = glob.glob(img_dir+'/'+cs+'/*gradcam*')
        for file_path in file_paths:
            img = cv2.imread(file_path)
            if blue_pic(img):
                os.remove(file_path)



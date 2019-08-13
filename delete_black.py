import numpy as np 
import cv2
import glob
import os

def zero_pic(img):
    if img.max()==0 and img.min()==0:
        return True
    else:
        return False

classes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
           'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD',
           'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
for i in range(10):
    img_dir = './heatmaps/guided_gradcam/'+str(i)
    for cs in classes:
        file_paths = glob.glob(img_dir+'/'+cs+'/*guided*')
        for file_path in file_paths:
            img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
            if zero_pic(img):
                os.remove(file_path)


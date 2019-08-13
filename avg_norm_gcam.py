import numpy as np 
import cv2
import glob
import os
import argparse
classes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
           'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD',
           'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

parser = argparse.ArgumentParser()
parser.add_argument('--folder_no', type=int, metavar='N')
args = parser.parse_args()
folder_no = args.folder_no

img_dir = './heatmaps/gradcam/'+str(folder_no)
for cs in classes:
    gcam_img = []

    file_paths = glob.glob(img_dir+'/'+cs+'/*gradcam*')
    for file_path in file_paths:
        img = cv2.imread(file_path)
        gcam_img.append(img)
    gcam_img = np.array(gcam_img)
    avg_gcam = gcam_img.mean(axis=0)
    avg_gcam -= avg_gcam.min()
    if avg_gcam.max()!=0:
        avg_gcam = avg_gcam / avg_gcam.max()
    avg_gcam_norm = avg_gcam * 255.0
    cv2.imwrite(img_dir+'/'+cs+'/'+cs+'_avg_gcam_norm.png',np.uint8(avg_gcam_norm))
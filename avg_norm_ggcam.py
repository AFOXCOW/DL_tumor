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

img_dir = './heatmaps/guided_gradcam/'+str(folder_no)
for cs in classes:
    gcam_img = []
    guided_gcam = []
    file_paths = glob.glob(img_dir+'/'+cs+'/*guided*')
    for file_path in file_paths:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        guided_gcam.append(img)

    guided_gcam = np.array(guided_gcam)
    avg_guided_gcam  = guided_gcam.mean(axis=0)
    avg_guided_gcam =  np.uint8(avg_guided_gcam)

    avg_guided_gcam -= avg_guided_gcam.min()
    if avg_guided_gcam.max()!=0:
        avg_guided_gcam = avg_guided_gcam / avg_guided_gcam.max()
    avg_guided_gcam_norm = avg_guided_gcam * 255.0
    cv2.imwrite(img_dir+'/'+cs+'/'+cs+'_avg_guided_gcam_norm.png',np.uint8(avg_guided_gcam_norm))
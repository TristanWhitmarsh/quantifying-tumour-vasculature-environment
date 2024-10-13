
from __future__ import print_function, division

import numpy as np
import cv2
from matplotlib import pyplot as plt
#import torchvision
#import torch
#import torch.nn as nn
from numpy import zeros
from numpy import ones



#from eval import eval_net_all
#from unet_model import UNet
#from load import get_ids_dir, get_imgs_and_masks_all
#from utils import batch, get_train_val, split_train_val, read_train_val, hwc_to_chw, normalize



from PIL import Image
import math
from itertools import product
#import skimage
#import skimage.data
#import skimage.morphology
#import skimage.filters
from scipy import ndimage
#import settings
import os
#from apex import amp

import glob, os



exclude = ["685039_(1.00,17661,41360,19968,2561)",
    "685039_(1.00,7007,741,9216,1024)",
    "685039_(1.00,81054,80,2561,3585)",
    "685040_(1.00,12444,41415,17920,1536)",
    "685042_(1.00,14,40462,17409,1025)",
    "685042_(1.00,22058,190,12800,1024)",
    "688058_(1.00,15734,32438,3072,2560)",
    "688059_(1.00,22312,137,2560,2049)",
    "688059_(1.00,30214,40324,6656,1537)",
    "688060_(1.00,2000,40,9728,2048)",
    "688061_(1.00,11308,19,6656,513)",
    "688061_(1.00,3687,38203,10240,3072)",
    "688062_(1.00,2057,40517,2561,1537)",
    "688062_(1.00,29137,99,3072,1537)",
    "688063_(1.00,144,39239,8192,1536)",
    "688063_(1.00,59467,8417,1024,3584)",
    "688064_(1.00,36,16442,5120,2560)",
    "688065_(1.00,10100,27089,1024,1024)",
    "688065_(1.00,17683,18489,1024,2048)",
    "688065_(1.00,70846,38229,7168,2560)",
    "688066_(1.00,19682,78,2048,2561)",
    "688066_(1.00,37781,26018,1025,1025)",
    "688067_(1.00,36205,37462,2048,1024)",
    "688067_(1.00,36818,33694,1025,1025)",
    "688067_(1.00,6355,8220,3072,2560)",
    "688068_(1.00,29035,40904,3072,2560)",
    "688068_(1.00,68661,18472,3072,1024)",
    "688069_(1.00,29,3205,513,4096)",
    "688070_(1.00,39203,33602,2048,1024)",
    "688070_(1.00,42838,10,1536,2560)",
    "688071_(1.00,48355,40346,3584,1536)",
    "688071_(1.00,68742,11843,1536,1536)",
    "688072_(1.00,167,29474,4608,1536)",
    "688072_(1.00,37596,7981,2560,2048)",
    "688072_(1.00,43334,28816,2048,2560)",
    "688072_(1.00,51839,26482,1536,1536)",
    "688073_(1.00,41390,5217,1536,1536)",
    "688073_(1.00,42068,31365,2048,4096)",
    "688073_(1.00,6443,2178,1025,4097)",
    "688074_(1.00,5212,37585,3584,1024)",
    "688074_(1.00,6985,33897,1536,1024)",
    "688075_(1.00,52,24706,3585,2048)",
    "688076_(1.00,2373,8304,1024,2048)",
    "688076_(1.00,40252,19178,2048,3072)",
    "688076_(1.00,998,13583,1025,1025)",
    "688077_(1.00,19589,3339,1024,1024)",
    "688077_(1.00,43727,34903,2560,512)",
    "688077_(1.00,464,357,3072,1024)",
    "688077_(1.00,68948,32390,1536,512)",
    "688078_(1.00,44049,40715,5120,2560)",
    "688078_(1.00,63258,7450,1536,1024)",
    "688078_(1.00,65548,21132,2049,1025)",
    "688079_(1.00,23762,5020,1024,2048)",
    "688079_(1.00,4134,17635,1024,2560)",
    "688079_(1.00,46337,3527,1024,1536)",
    "688080_(1.00,2964,22951,2048,1536)",
    "688080_(1.00,36554,385,1024,1536)",
    "688081_(1.00,23205,154,8192,512)",
    "688081_(1.00,5887,36485,2048,2048)",
    "688082_(1.00,34697,37863,4608,3584)",
    "688083_(1.00,33758,10111,2048,1536)",
    "688084_(1.00,31634,38140,3072,2560)",
    "688135_(1.00,15442,3051,3072,1024)",
    "688135_(1.00,55423,30392,1024,2048)",
    "688085_(1.00,13134,1089,14848,4096)",]


    
if __name__ == '__main__':
    
    
    # f = open("confusion_matrix/confusion_matrix_MISSION.csv", "w")
    f = open("confusion_matrix/confusion_matrix_BEHOLD.csv", "w")
    f.write("Predicted,True,Count\n")
    
    
    confusion_matrix = np.zeros((10, 10))
    
    
    # directory_true = '/home/user/GitRepositories/Vessel/masksMISSION_corrected/'
    # directory_pred = '/home/user/GitRepositories/Vessel/Inference_MISSION/'
    directory_true = '/home/user/GitRepositories/Vessel/masksBEHOLD_corrected/'
    directory_pred = '/home/user/GitRepositories/Vessel/Inference_BEHOLD/'
    
    os.chdir(directory_true)
    for file in glob.glob("*.jpg"):
        
        print(file)
        
        filename = os.path.basename(file[6:-4])
        filename2 = os.path.basename(file[:-4])
        
        imageID = os.path.basename(file[:-4])
        patientID = imageID.split("_")
        
        imageID2 = patientID[0] + "_" + patientID[1]

        if imageID2 in exclude:
            print("excluding " + imageID2)
        else:

            true_mask_tumor = cv2.imread(directory_true + patientID[0] + '_Tumor_' + patientID[1] + '-mask.png', 0)
            true_mask_benign = cv2.imread(directory_true + patientID[0] + '_Benign_' + patientID[1] + '-mask.png', 0)
            true_mask_vessel = cv2.imread(directory_true + patientID[0] + '_vessel_' + patientID[1] + '-mask.png', 0)
            true_mask_adipose = cv2.imread(directory_true + patientID[0] + '_Adipose_' + patientID[1] + '-mask.png', 0)
            true_mask_background = cv2.imread(directory_true + patientID[0] + '_Background_' + patientID[1] + '-mask.png', 0)
            true_mask_lymphocytes = cv2.imread(directory_true + patientID[0] + '_Lymphocytes_' + patientID[1] + '-mask.png', 0)
            true_mask_macrophages = cv2.imread(directory_true + patientID[0] + '_Macrophages_' + patientID[1] + '-mask.png', 0)
            true_mask_muscle = cv2.imread(directory_true + patientID[0] + '_Muscle_' + patientID[1] + '-mask.png', 0)
            true_mask_nerve = cv2.imread(directory_true + patientID[0] + '_Nerve_' + patientID[1] + '-mask.png', 0)
            true_mask_stroma = 255 - true_mask_tumor - true_mask_benign - true_mask_adipose - true_mask_background - true_mask_lymphocytes - true_mask_macrophages - true_mask_muscle - true_mask_nerve - true_mask_vessel
            true_mask_stroma = np.maximum(true_mask_stroma, 0)

            mask_pred_tumor = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Tumor.png', 0)
            mask_pred_benign = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Benign.png', 0)
            mask_pred_vessel = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Vessel.png', 0)
            mask_pred_adipose = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Adipose.png', 0)
            mask_pred_background = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Background.png', 0)
            mask_pred_lymphocytes = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Lymphocytes.png', 0)
            mask_pred_macrophages = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Macrophages.png', 0)
            mask_pred_muscle = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Muscle.png', 0)
            mask_pred_nerve = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Nerve.png', 0)
            mask_pred_stroma = cv2.imread(directory_pred + patientID[0] + '_' + patientID[1] + '_Stroma.png', 0)

            h, w = mask_pred_tumor.shape

            # sanity test

            # true_mask_tumor = np.ones((h, w))
            # true_mask_benign = np.zeros((h, w))
            # true_mask_vessel = np.zeros((h, w))
            # true_mask_adipose = np.zeros((h, w))
            # true_mask_background = np.zeros((h, w))
            # true_mask_lymphocytes = np.zeros((h, w))
            # true_mask_macrophages = np.zeros((h, w))
            # true_mask_muscle = np.zeros((h, w))
            # true_mask_necrosis = np.zeros((h, w))
            # true_mask_nerve = np.zeros((h, w))
            # true_mask_stroma = np.zeros((h, w))

            # mask_pred_tumor = np.zeros((h, w))
            # mask_pred_benign = np.ones((h, w))
            # mask_pred_vessel = np.zeros((h, w))
            # mask_pred_adipose = np.zeros((h, w))
            # mask_pred_background = np.zeros((h, w))
            # mask_pred_lymphocytes = np.zeros((h, w))
            # mask_pred_macrophages = np.zeros((h, w))
            # mask_pred_muscle = np.zeros((h, w))
            # mask_pred_necrosis = np.zeros((h, w))
            # mask_pred_nerve = np.zeros((h, w))
            # mask_pred_stroma = np.zeros((h, w))

            for pos in product(range(h), range(w)):

                if mask_pred_stroma.item(pos) > 0:
                    i = 0
                elif mask_pred_background.item(pos) > 0:
                    i = 1
                elif mask_pred_adipose.item(pos) > 0:
                    i = 2
                elif mask_pred_tumor.item(pos) > 0:
                    i = 3
                elif mask_pred_vessel.item(pos) > 0:
                    i = 4
                elif mask_pred_lymphocytes.item(pos) > 0:
                    i = 5
                elif mask_pred_benign.item(pos) > 0:
                    i = 6
                elif mask_pred_muscle.item(pos) > 0:
                    i = 7
                elif mask_pred_nerve.item(pos) > 0:
                    i = 8
                elif mask_pred_macrophages.item(pos) > 0:
                    i = 9

                if true_mask_stroma.item(pos) > 0:
                    j = 0
                elif true_mask_background.item(pos) > 0:
                    j = 1
                elif true_mask_adipose.item(pos) > 0:
                    j = 2
                elif true_mask_tumor.item(pos) > 0:
                    j = 3
                elif true_mask_vessel.item(pos) > 0:
                    j = 4
                elif true_mask_lymphocytes.item(pos) > 0:
                    j = 5
                elif true_mask_benign.item(pos) > 0:
                    j = 6
                elif true_mask_muscle.item(pos) > 0:
                    j = 7
                elif true_mask_nerve.item(pos) > 0:
                    j = 8
                elif true_mask_macrophages.item(pos) > 0:
                    j = 9

                confusion_matrix[j][i] += 1


    
    
    for x in range(0, 10):
        for y in range(0, 10):
            if x == 0: f.write("stroma,")
            if x == 1: f.write("background,")
            if x == 2: f.write("adipose,")
            if x == 3: f.write("tumor,")
            if x == 4: f.write("vessel,")
            if x == 5: f.write("lymphocytes,")
            if x == 6: f.write("benign,")
            if x == 7: f.write("muscle,")
            if x == 8: f.write("nerve,")
            if x == 9: f.write("leukocytes,")
            
            if y == 0: f.write("stroma,")
            if y == 1: f.write("background,")
            if y == 2: f.write("adipose,")
            if y == 3: f.write("tumor,")
            if y == 4: f.write("vessel,")
            if y == 5: f.write("lymphocytes,")
            if y == 6: f.write("benign,")
            if y == 7: f.write("muscle,")
            if y == 8: f.write("nerve,")
            if y == 9: f.write("leukocytes,")
            
            f.write(str(confusion_matrix[y][x]))
            f.write("\n")
    
    f.flush()
    os.fsync(f.fileno())
    
    f.close()
    
        
    
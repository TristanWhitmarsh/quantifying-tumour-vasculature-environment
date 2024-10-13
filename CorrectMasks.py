
from __future__ import print_function, division

import numpy as np
import cv2
from matplotlib import pyplot as plt
#import torchvision
#import torch
#import torch.nn as nn
from numpy import zeros
from numpy import ones
from PIL import Image
import math
from itertools import product
from scipy import ndimage
import os

import glob, os
from os.path import exists


    
if __name__ == '__main__':
    
    
    directory_in = 'masks/'
    directory_out = 'masks_corrected/'
    
    #directory_in = 'C:/U-Net/masksMISSION/'
    #directory_out = 'C:/U-Net/masksMISSION_corrected/'
    
    # directory_in = 'C:/U-Net/masksBEHOLD/'
    # directory_out = 'C:/U-Net/masksBEHOLD_corrected/'
    
    #os.chdir(directory_in)
    for file in glob.glob("masks/*.jpg"):
        
        
        imageID = os.path.basename(file[:-4])
        print(imageID)
        patientID = imageID.split("_")
        print(patientID[0])
        
        img = cv2.imread(directory_in + imageID + '.jpg')
        
        mask_vessel = cv2.imread(directory_in + patientID[0] + '_vessel_' + patientID[1] + '-mask.png', 0)//255
        mask_tumor = cv2.imread(directory_in + patientID[0] + '_Tumor_' + patientID[1] + '-mask.png', 0)//255
        mask_benign = cv2.imread(directory_in + patientID[0] + '_Benign_' + patientID[1] + '-mask.png', 0)//255
        mask_adipose = cv2.imread(directory_in + patientID[0] + '_Adipose_' + patientID[1] + '-mask.png', 0)//255
        mask_background = cv2.imread(directory_in + patientID[0] + '_Background_' + patientID[1] + '-mask.png', 0)//255
        mask_lymphocytes = cv2.imread(directory_in + patientID[0] + '_Lymphocytes_' + patientID[1] + '-mask.png', 0)//255
        mask_macrophages = cv2.imread(directory_in + patientID[0] + '_Macrophages_' + patientID[1] + '-mask.png', 0)//255
        mask_muscle = cv2.imread(directory_in + patientID[0] + '_Muscle_' + patientID[1] + '-mask.png', 0)//255
        mask_nerve = cv2.imread(directory_in + patientID[0] + '_Nerve_' + patientID[1] + '-mask.png', 0)//255


        mask_vessel2 = mask_vessel
        
        mask_muscle2 = cv2.subtract(mask_muscle, mask_vessel)
        
        mask_macrophages2 = cv2.subtract(mask_macrophages, mask_muscle)
        mask_macrophages2 = cv2.subtract(mask_macrophages2, mask_vessel)
        
        mask_adipose2 = cv2.subtract(mask_adipose, mask_vessel)
        mask_adipose2 = cv2.subtract(mask_adipose2, mask_muscle)
        mask_adipose2 = cv2.subtract(mask_adipose2, mask_macrophages)
        
        mask_nerve2 = cv2.subtract(mask_nerve, mask_vessel)
        mask_nerve2 = cv2.subtract(mask_nerve2, mask_muscle)
        mask_nerve2 = cv2.subtract(mask_nerve2, mask_macrophages)
        mask_nerve2 = cv2.subtract(mask_nerve2, mask_adipose)
        
        mask_tumor2 = cv2.subtract(mask_tumor, mask_vessel)
        mask_tumor2 = cv2.subtract(mask_tumor2, mask_muscle)
        mask_tumor2 = cv2.subtract(mask_tumor2, mask_macrophages)
        mask_tumor2 = cv2.subtract(mask_tumor2, mask_adipose)
        mask_tumor2 = cv2.subtract(mask_tumor2, mask_nerve)
        
        mask_background2 = cv2.subtract(mask_background, mask_vessel)
        mask_background2 = cv2.subtract(mask_background2, mask_muscle)
        mask_background2 = cv2.subtract(mask_background2, mask_macrophages)
        mask_background2 = cv2.subtract(mask_background2, mask_adipose)
        mask_background2 = cv2.subtract(mask_background2, mask_nerve)
        mask_background2 = cv2.subtract(mask_background2, mask_tumor)
        
        mask_benign2 = cv2.subtract(mask_benign, mask_vessel)
        mask_benign2 = cv2.subtract(mask_benign2, mask_muscle)
        mask_benign2 = cv2.subtract(mask_benign2, mask_macrophages)
        mask_benign2 = cv2.subtract(mask_benign2, mask_adipose)
        mask_benign2 = cv2.subtract(mask_benign2, mask_nerve)
        mask_benign2 = cv2.subtract(mask_benign2, mask_tumor)
        mask_benign2 = cv2.subtract(mask_benign2, mask_background)
        
        mask_lymphocytes2 = cv2.subtract(mask_lymphocytes, mask_vessel)
        mask_lymphocytes2 = cv2.subtract(mask_lymphocytes2, mask_muscle)
        mask_lymphocytes2 = cv2.subtract(mask_lymphocytes2, mask_macrophages)
        mask_lymphocytes2 = cv2.subtract(mask_lymphocytes2, mask_adipose)
        mask_lymphocytes2 = cv2.subtract(mask_lymphocytes2, mask_nerve)
        mask_lymphocytes2 = cv2.subtract(mask_lymphocytes2, mask_tumor)
        mask_lymphocytes2 = cv2.subtract(mask_lymphocytes2, mask_background)
        mask_lymphocytes2 = cv2.subtract(mask_lymphocytes2, mask_benign)
        
        mask_stroma2 = 1 - mask_lymphocytes2 - mask_benign2 - mask_tumor2 - mask_background2 - mask_adipose2 - mask_macrophages2 - mask_nerve2 - mask_muscle2 - mask_vessel2
        mask_stroma2 = np.maximum(mask_stroma2, 0)
        
        cv2.imwrite(directory_out + imageID + '.jpg', img)
        cv2.imwrite(directory_out + patientID[0] + '_vessel_' + patientID[1] + '-mask.png', 255*mask_vessel2)
        cv2.imwrite(directory_out + patientID[0] + '_Tumor_' + patientID[1] + '-mask.png', 255*mask_tumor2)
        cv2.imwrite(directory_out + patientID[0] + '_Benign_' + patientID[1] + '-mask.png', 255*mask_benign2)
        cv2.imwrite(directory_out + patientID[0] + '_Adipose_' + patientID[1] + '-mask.png', 255*mask_adipose2)
        cv2.imwrite(directory_out + patientID[0] + '_Background_' + patientID[1] + '-mask.png', 255*mask_background2)
        cv2.imwrite(directory_out + patientID[0] + '_Lymphocytes_' + patientID[1] + '-mask.png', 255*mask_lymphocytes2)
        cv2.imwrite(directory_out + patientID[0] + '_Macrophages_' + patientID[1] + '-mask.png', 255*mask_macrophages2)
        cv2.imwrite(directory_out + patientID[0] + '_Muscle_' + patientID[1] + '-mask.png', 255*mask_muscle2)
        cv2.imwrite(directory_out + patientID[0] + '_Nerve_' + patientID[1] + '-mask.png', 255*mask_nerve2)
        cv2.imwrite(directory_out + patientID[0] + '_Stroma_' + patientID[1] + '-mask.png', 255*mask_stroma2)
        





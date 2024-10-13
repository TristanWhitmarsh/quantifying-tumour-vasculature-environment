# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:05:53 2019

@author: whitma01
"""

import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
from numpy import zeros
import datetime
from tqdm import tqdm
import os
import glob, os
from PIL import Image
import gc

Image.MAX_IMAGE_PIXELS = None


def measure(fileTumor, fileTumorDilated, fileStroma, fileLymphocytes, doprint):
    
    a = datetime.datetime.now()
    
    pixel_length = 0.5034
    pixel_length_mm = 0.0005034
    pixel_size = (pixel_length * pixel_length) # microns
    pixel_size_mm = (pixel_length_mm * pixel_length_mm) # microns
    
    
    mask_stroma = Image.open(fileStroma)
    mask_stroma = np.array(mask_stroma)
    mask_stroma = np.uint8(mask_stroma)
    
    mask_lymphocytes = Image.open(fileLymphocytes)
    mask_lymphocytes = np.array(mask_lymphocytes)
    mask_lymphocytes = np.uint8(mask_lymphocytes)
    
    mask_tumor = Image.open(fileTumor)
    mask_tumor = np.array(mask_tumor)
    mask_tumor = np.uint8(mask_tumor)
    
    mask_tumor_dilated = Image.open(fileTumorDilated)
    mask_tumor_dilated = np.array(mask_tumor_dilated)
    mask_tumor_dilated = np.uint8(mask_tumor_dilated)
      
    
    mask_stroma_lymphocytes = mask_stroma
    mask_stroma_lymphocytes[mask_lymphocytes == 1] = 1
    
    
    mask_stroma_lymphocytes[mask_tumor_dilated == 0] = 0
    mask_lymphocytes[mask_tumor_dilated == 0] = 0

    
    tumor_area = np.count_nonzero(mask_tumor) * pixel_size_mm;
    stroma_area = np.count_nonzero(mask_stroma_lymphocytes) * pixel_size_mm;
    lymphocyte_area = np.count_nonzero(mask_lymphocytes) * pixel_size_mm;
    
    print("tumor_area {}".format(tumor_area))
    print("stroma_area {}".format(stroma_area))
    print("lymphocyte_area {}".format(lymphocyte_area))
    
    del mask_tumor
    del mask_stroma
    del mask_lymphocytes
    del mask_tumor_dilated
    gc.collect()
    
    return tumor_area, stroma_area, lymphocyte_area
    


names = [
#['684913'],
#['684914'],
#['684916'],
['685039','685040'],
#['685041'],
['685042'],
['685281'],
['688058'],
['688059','688060'],
['688061'],
['688062'],
['688063'],
['688064'],
['688065'],
['688066','688067'],
['688068','688069'],
['688070'],
['688071'],
['688072'],
['688073'],
['688074'],
['688075','688076'],
['688077'],
['688078'],
['688079','688080','688081','688135'],
['688082','688083'],
['688084'],
['745384'],
['745382'],
#['745352'],
['745353'],
#['745371'],
#['745372'],
['745373', '745374'],
#['745354', '745355'],
['745356'],
['745357'],
['745358', '745359'],
#['745381', '745383'],
['745375'],
#['745360'],
#['745361', '745362'],
['745363', '745364'],
['745365'],
['745366', '745367'],
['745368', '745369'],
['745370'],
['745380'],
#['745378', '745379'],
['745376', '745377']]


directory = "FullInferenceProcessed/"
directory_dilated = "FullInferenceDilatedTumorAdipose/"

data_writer = open("measurements/tumor_tissue_ratios.csv", "w")


data_writer.write("Name,tumor_stroma_ratio,lymphocyte_stroma_percentage\n")
data_writer.flush()
os.fsync(data_writer.fileno())


for i, patient in enumerate(names):
    print(patient)
    
    total_tumor_area = 0
    total_stroma_area = 0
    total_lymphocyte_area = 0
    
    for i, name in enumerate(patient):
        print(name)
        
        file_tumor = directory + name +"_tumor.png"
        file_stroma = directory + name +"_stroma.png"
        file_lymphocytes = directory + name +"_lymphocytes.png"
        file_tumor_dilated = directory_dilated + name +"_tumor_dilated.png"
        
        tumor_area, stroma_area, lymphocyte_area = measure(file_tumor, file_tumor_dilated, file_stroma, file_lymphocytes, False)

        total_tumor_area += tumor_area
        total_stroma_area += stroma_area
        total_lymphocyte_area += lymphocyte_area
                
    
    
    tumor_stroma_ratio = total_tumor_area / total_stroma_area
    lymphocyte_stroma_percentage = 100 * (total_lymphocyte_area / total_stroma_area)
    
    
    print()
    print('tumor stroma ratio: {:.5f}'.format(tumor_stroma_ratio))
    print('lymphocyte stroma percentage: {:.5f}'.format(lymphocyte_stroma_percentage))
    
    data_writer.write(name  
                      + "," + str(tumor_stroma_ratio) 
                      + "," + str(lymphocyte_stroma_percentage) + "\n")
    data_writer.flush()
    os.fsync(data_writer.fileno())
    
        
data_writer.close()
        
        
        
        
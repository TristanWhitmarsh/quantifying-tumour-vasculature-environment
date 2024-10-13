
#from __future__ import print_function, division

import numpy as np
#from matplotlib import pyplot as plt
#import torchvision
#import torch
#import torch.nn as nn
#from numpy import zeros
#from numpy import ones
from PIL import Image
#import math
#from itertools import product
from scipy import ndimage
#import os
import SimpleITK as sitk
import gc
import glob, os
# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,50).__str__()
import cv2
from os.path import exists
    
if __name__ == '__main__':
    
    
    directory_in = 'FullInference/'
    directory_out = 'FullInferenceProcessed/'
    
    # os.chdir(directory_in)
    for file in glob.glob(directory_in + "*.png"):
        
        
        imageID = os.path.basename(file[:-4])
        # print(imageID)
        patientID = imageID.split("_")
        # print(patientID[0])
        
        if patientID[1] == "vessel" and not exists(directory_out + patientID[0] + '_vessel.png'):
            print(patientID[0])
            Image.MAX_IMAGE_PIXELS = 36764830080
            
            mask_vessel = Image.open(directory_in + patientID[0] + '_vessel.png')
            mask_vessel = np.uint8(mask_vessel)
            mask_tumor = Image.open(directory_in + patientID[0] + '_tumor.png')
            mask_tumor = np.uint8(mask_tumor)
            mask_benign = Image.open(directory_in + patientID[0] + '_benign.png')
            mask_benign = np.uint8(mask_benign)
            mask_adipose = Image.open(directory_in + patientID[0] + '_adipose.png')
            mask_adipose = np.uint8(mask_adipose)
            mask_background = Image.open(directory_in + patientID[0] + '_background.png')
            mask_background = np.uint8(mask_background)
            mask_lymphocytes = Image.open(directory_in + patientID[0] + '_lymphocytes.png')
            mask_lymphocytes = np.uint8(mask_lymphocytes)
            mask_macrophages = Image.open(directory_in + patientID[0] + '_macrophages.png')
            mask_macrophages = np.uint8(mask_macrophages)
            mask_muscle = Image.open(directory_in + patientID[0] + '_muscle.png')
            mask_muscle = np.uint8(mask_muscle)
            mask_nerve = Image.open(directory_in + patientID[0] + '_nerve.png')
            mask_nerve = np.uint8(mask_nerve)
            mask_stroma = Image.open(directory_in + patientID[0] + '_stroma.png')
            mask_stroma = np.uint8(mask_stroma)


                
            def remove_small_regions(m_mask, threshold): 
                m_mask_inverse = (1-m_mask)
                img = sitk.GetImageFromArray(m_mask_inverse)
                cca = sitk.ConnectedComponentImageFilter()
                cca_image = cca.Execute(img)
                stats = sitk.LabelShapeStatisticsImageFilter()                                                                                                                                     
                stats.Execute(cca_image)
                
                relabelMap =  { i : 0 for i in stats.GetLabels() if stats.GetNumberOfPixels(i) < threshold }
                output = sitk.ChangeLabel(cca_image, changeMap=relabelMap)
                
                del img
                del cca
                del cca_image
                del stats
                
                m_mask_inverse = sitk.GetArrayFromImage(output)
                del output
                m_mask_inverse = m_mask_inverse != 0
                m_mask = (1-m_mask_inverse)
                
                del m_mask_inverse
                return m_mask.astype(bool)
            
            
                
            if True:        
                            
                print('remove small regions in background')
                mask_background = remove_small_regions(mask_background, 10000)

                mask_tumor = mask_tumor > mask_background
                mask_benign = mask_benign > mask_background
                mask_adipose = mask_adipose > mask_background
                mask_lymphocytes = mask_lymphocytes > mask_background
                mask_macrophages = mask_macrophages > mask_background
                mask_muscle = mask_muscle > mask_background
                mask_nerve = mask_nerve > mask_background
                mask_vessel = mask_vessel > mask_background
                mask_stroma = mask_stroma > mask_background
                gc.collect()
                   
                print('remove small regions in stroma')
                mask_stroma = remove_small_regions(mask_stroma, 1000)
            
                mask_tumor = mask_tumor > mask_stroma
                mask_benign = mask_benign > mask_stroma
                mask_adipose = mask_adipose > mask_stroma
                mask_background = mask_background > mask_stroma
                mask_lymphocytes = mask_lymphocytes > mask_stroma
                mask_muscle = mask_muscle > mask_stroma
                mask_nerve = mask_nerve > mask_stroma
                gc.collect()
            
                print('remove small regions in adipose')
                mask_adipose = remove_small_regions(mask_adipose, 10000)
            
                mask_tumor = mask_tumor > mask_adipose
                mask_benign = mask_benign > mask_adipose
                mask_background = mask_background > mask_adipose
                mask_lymphocytes = mask_lymphocytes > mask_adipose
                mask_muscle = mask_muscle > mask_adipose
                mask_nerve = mask_nerve > mask_adipose
                mask_stroma = mask_stroma > mask_adipose
                gc.collect()
                
                
                print('recover macrophages')
                mask_tumor = mask_tumor > mask_macrophages
                mask_benign = mask_benign > mask_macrophages
                mask_adipose = mask_adipose > mask_macrophages
                mask_background = mask_background > mask_macrophages
                mask_lymphocytes = mask_lymphocytes > mask_macrophages
                mask_muscle = mask_muscle > mask_macrophages
                mask_nerve = mask_nerve > mask_macrophages
                mask_stroma = mask_stroma > mask_macrophages
                gc.collect()
            
                
                print('fill vessel holes')
                mask_vessel = ndimage.binary_fill_holes(mask_vessel)
                
                mask_tumor = mask_tumor > mask_vessel
                mask_benign = mask_benign > mask_vessel
                mask_adipose = mask_adipose > mask_vessel
                mask_background = mask_background > mask_vessel
                mask_lymphocytes = mask_lymphocytes > mask_vessel
                mask_macrophages = mask_macrophages > mask_vessel
                mask_muscle = mask_muscle > mask_vessel
                mask_nerve = mask_nerve > mask_vessel
                mask_stroma = mask_stroma > mask_vessel
                gc.collect()
                    
                
            if False:
                print("distinguishing tumor/benign")
                mask_tumor_benign = np.maximum(mask_tumor, mask_benign)
                nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_tumor_benign.astype(np.uint8), connectivity=8)
            
                lookup_table = np.zeros((nlabels, 2)).astype(float)
                height, width = mask_tumor_benign.shape[:2]
                            
                for x in range(0, width):
                    for y in range(0, height):
                        i = labels[y,x]
                        lookup_table[i][0] += img_map[0,:,:][y,x]
                        lookup_table[i][1] += img_map[1,:,:][y,x]
                
                
                for x in range(0, width):
                    for y in range(0, height):
                        i = labels[y,x]
                        if i > 0:
                            if (lookup_table[i][0] / lookup_table[i][1]) >= 0.5:
                                mask_tumor[y,x] = 1
                                mask_benign[y,x] = 0
                            else:
                                mask_tumor[y,x] = 0
                                mask_benign[y,x] = 1
                        else:
                            mask_tumor[y,x] = 0
                            mask_benign[y,x] = 0
                
                b = datetime.datetime.now()
                delta = b - a
                print("distinguishing tumor/benign took {}".format(delta))
                a = datetime.datetime.now()
            

            im = Image.fromarray(mask_vessel.astype(bool))
            im.save(directory_out + patientID[0] + '_vessel.png')
            im = Image.fromarray(mask_tumor.astype(bool))
            im.save(directory_out + patientID[0] + '_tumor.png')
            im = Image.fromarray(mask_benign.astype(bool))
            im.save(directory_out + patientID[0] + '_benign.png')
            im = Image.fromarray(mask_adipose.astype(bool))
            im.save(directory_out + patientID[0] + '_adipose.png')
            im = Image.fromarray(mask_background.astype(bool))
            im.save(directory_out + patientID[0] + '_background.png')
            im = Image.fromarray(mask_lymphocytes.astype(bool))
            im.save(directory_out + patientID[0] + '_lymphocytes.png')
            im = Image.fromarray(mask_macrophages.astype(bool))
            im.save(directory_out + patientID[0] + '_macrophages.png')
            im = Image.fromarray(mask_muscle.astype(bool))
            im.save(directory_out + patientID[0] + '_muscle.png')
            im = Image.fromarray(mask_nerve.astype(bool))
            im.save(directory_out + patientID[0] + '_nerve.png')
            im = Image.fromarray(mask_stroma.astype(bool))
            im.save(directory_out + patientID[0] + '_stroma.png')



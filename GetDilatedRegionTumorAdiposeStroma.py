# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:16:57 2020

@author: whitma01
"""


from PIL import Image
import numpy as np
from skimage import morphology
import os
import datetime
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import gc

Image.MAX_IMAGE_PIXELS = None

def remove_small_regions(m_mask, threshold): 
    m_mask = sitk.GetImageFromArray(m_mask)
    m_mask = sitk.ConnectedComponentImageFilter().Execute(m_mask)
    stats = sitk.LabelShapeStatisticsImageFilter()                                                                                                                                     
    stats.Execute(m_mask)
    relabelMap =  { i : 0 for i in stats.GetLabels() if stats.GetNumberOfPixels(i) < threshold }
    m_mask = sitk.ChangeLabel(m_mask, changeMap=relabelMap)
    m_mask = sitk.GetArrayFromImage(m_mask)
    m_mask = m_mask != 0
    
    del stats
    gc.collect()
    return m_mask

# files = ['688058',
# '688059',
# '688060',
# '688061',
# '688062',
# '688063',
# '688064',
# '688065',
# '688066',
# '688067',
# '688068',
# '688069',
# '688070',
# '688071',
# '688072',
# '688073',
# '688074',
# '688075',
# '688076',
# '688077',
# '688078',
# '688079',
# '688080',
# '688081',
# '688082',
# '688083',
# '688084',
# '688135',
# '685039',
# '685040',
# '685042',]



files = [
'688079',
'688076',
'688071',
'688072',
'688073',
'688074',
'688075',
'688078',
'688080',
'688081',
'688083',
'688084',
'688085',
'688135',
'688058',
'688059',
'688060',
'688061',
'688062',
'688063',
'688064',
'688065',
'688066',
'688068',
'688069',
'685041',
'685042',
'685040',
'685281',
'684913',
'684914',
'684916',
'685039',
'745366',
'745367',
'745368',
'745369',
'745370',
'745372',
'745376',
'745373',
'745371',
'745374',
'745378',
'745377',
'745375',
'745381',
'745384',
'745352',
'745380',
'745382',
'745353',
'745356',
'745364',
'745359',
'745357',
'745383',
'745363',
'745355',
'745362',
'745354',
'745358',
'745360',
'745361',
'688067',
'688077',
'688082',
'688070',
'745379',
'745365',
]


directory_in = "FullInferenceProcessed/"
directory_out = "FullInferenceDilatedTumorAdipose/"


kernel_size = 25
kernel = np.zeros((2*kernel_size+1,2*kernel_size+1), np.uint8)
y,x = np.ogrid[-kernel_size: kernel_size+1, -kernel_size: kernel_size+1]
kernel_area = x**2+y**2 <= kernel_size**2
kernel[kernel_area] = 1


# for i, file in enumerate(files):
#     print(file)
def run(file_number):
    for index in range(7):
        file = files[(7*file_number)+index]
        print(file)

        mask_Adipose = Image.open(directory_in + file +"_adipose.png")
        mask_Adipose = np.array(mask_Adipose)
        mask_Adipose = np.uint8(mask_Adipose)
        print("removing small regions")
        mask_Adipose = remove_small_regions(mask_Adipose, 1000)
        mask_Adipose = np.uint8(mask_Adipose)  

        mask_Tumor = Image.open(directory_in + file +"_tumor.png")
        mask_Tumor = np.array(mask_Tumor)
        mask_Tumor = np.uint8(mask_Tumor)
        print("removing small regions")
        mask_Tumor = remove_small_regions(mask_Tumor, 1000)
        mask_Tumor = np.uint8(mask_Tumor)

        print("dilating")

    #         mask_tumor = cv2.dilate(mask_tumor,kernel,iterations = 1)
    #         mask_adipose = cv2.dilate(mask_adipose,kernel,iterations = 1)
    #         mask_adipose = cv2.subtract(mask_adipose,mask_tumor)

    #         mask_tumor = cv2.dilate(mask_tumor,kernel,iterations = 1)
    #         mask_adipose = cv2.dilate(mask_adipose,kernel,iterations = 1)
    #         mask_tumor = cv2.subtract(mask_tumor,mask_adipose)

        mask_Tumor = cv2.dilate(mask_Tumor,kernel,iterations = 1)
        mask_Tumor = cv2.subtract(mask_Tumor,mask_Adipose)
        mask_Adipose = cv2.dilate(mask_Adipose,kernel,iterations = 1)
        mask_Adipose = cv2.subtract(mask_Adipose,mask_Tumor)
        for i in range(11):
            print(i)
            mask_Tumor = cv2.dilate(mask_Tumor,kernel,iterations = 1)
            mask_Tumor = cv2.subtract(mask_Tumor,mask_Adipose)
            mask_Adipose = cv2.dilate(mask_Adipose,kernel,iterations = 1)
            mask_Adipose = cv2.subtract(mask_Adipose,mask_Tumor)


        mask_Tumor = mask_Tumor > 0
        mask_Tumor = Image.fromarray(mask_Tumor)   
        mask_Tumor.save(directory_out + file + "_tumor_dilated.png")

        mask_Adipose = mask_Adipose > 0
        mask_Adipose = Image.fromarray(mask_Adipose)   
        mask_Adipose.save(directory_out + file + "_adipose_dilated.png")


        del mask_Tumor
        del mask_Adipose

        gc.collect()

import threading
threads = list()
for index in range(10):
    x = threading.Thread(target=run, args=(index,))
    threads.append(x)
    x.start()
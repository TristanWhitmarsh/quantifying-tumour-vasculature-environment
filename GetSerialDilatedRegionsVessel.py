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
from scipy import ndimage
from skimage.morphology import disk, binary_dilation
import time

Image.MAX_IMAGE_PIXELS = None

def remove_small_regions(m_mask, threshold): 

    img = sitk.GetImageFromArray(m_mask)
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
    m_mask_inverse = m_mask_inverse != 0
    return m_mask_inverse


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
# '685042',
# ]



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
directory_out = "FullInferenceDilatedVessel/"


kernel_size = 50
kernel = np.zeros((2*kernel_size+1,2*kernel_size+1), np.uint8)
y,x = np.ogrid[-kernel_size: kernel_size+1, -kernel_size: kernel_size+1]
kernel_area = x**2+y**2 <= kernel_size**2
kernel[kernel_area] = 1

# for i, file in enumerate(files):
    
def run(file_number):
    

    for index in range(7):
        file = files[(7*file_number)+index]
        print(file)

        mask_vessel = Image.open(directory_in + file +"_vessel.png")
        mask_vessel = np.array(mask_vessel)
        mask_vessel = np.uint8(mask_vessel)
        print("removing small regions")
        mask_vessel = remove_small_regions(mask_vessel, 40)
        mask_vessel = np.uint8(mask_vessel) 

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated1.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated2.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated3.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated4.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated5.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated6.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated7.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated8.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated9.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated10.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated11.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated12.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated13.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated14.png")

        mask_vessel = cv2.dilate(mask_vessel,kernel,iterations = 1)   
        mask_vessel_binary = mask_vessel > 0
        mask_vessel_binary = Image.fromarray(mask_vessel_binary)   
        mask_vessel_binary.save(directory_out + file + "_vessel_dilated15.png")
        print("finished " + file)


import threading
threads = list()
for index in range(10):
    x = threading.Thread(target=run, args=(index,))
    threads.append(x)
    x.start()
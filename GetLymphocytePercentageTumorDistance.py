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
import cv2
import gc



Image.MAX_IMAGE_PIXELS = None


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
directory_dilated = "FullInferenceDilatedTumor/"

data_writer = open("regional_analysis/lymphocyte_tumor_percentage.csv", "w")

data_writer.write("Name, region1, region2, region3, region4, region5, region6, region7, region8, region9, region10, region11, region12, region13, region14, region15\n")
data_writer.flush()
os.fsync(data_writer.fileno())



for i, patient in enumerate(names):
    
    stroma_count_region1 = 0
    stroma_count_region2 = 0
    stroma_count_region3 = 0
    stroma_count_region4 = 0
    stroma_count_region5 = 0
    stroma_count_region6 = 0
    stroma_count_region7 = 0
    stroma_count_region8 = 0
    stroma_count_region9 = 0
    stroma_count_region10 = 0
    stroma_count_region11 = 0
    stroma_count_region12 = 0
    stroma_count_region13 = 0
    stroma_count_region14 = 0
    stroma_count_region15 = 0
    
    lymphocyte_count_region1 = 0
    lymphocyte_count_region2 = 0
    lymphocyte_count_region3 = 0
    lymphocyte_count_region4 = 0
    lymphocyte_count_region5 = 0
    lymphocyte_count_region6 = 0
    lymphocyte_count_region7 = 0
    lymphocyte_count_region8 = 0
    lymphocyte_count_region9 = 0
    lymphocyte_count_region10 = 0
    lymphocyte_count_region11 = 0
    lymphocyte_count_region12 = 0
    lymphocyte_count_region13 = 0
    lymphocyte_count_region14 = 0
    lymphocyte_count_region15 = 0

    for i, file in enumerate(patient):
        print("file {}".format(file))
        
        mask_lymphocytes = Image.open(directory + file +"_lymphocytes.png")
        mask_stroma = Image.open(directory + file +"_stroma.png")

        mask_tumor_dilated1 = Image.open(directory_dilated + file +"_tumor_dilated1.png")


        mask_lymphocytes = np.array(mask_lymphocytes)
        mask_stroma = np.array(mask_stroma)
        mask_tumor_dilated1 = np.array(mask_tumor_dilated1)

        stroma_count_region1 += np.count_nonzero(mask_stroma & mask_tumor_dilated1)
        lymphocyte_count_region1 += np.count_nonzero(mask_lymphocytes & mask_tumor_dilated1)

        mask_tumor_dilated2 = Image.open(directory_dilated + file +"_tumor_dilated2.png")
        mask_tumor_dilated2 = np.array(mask_tumor_dilated2)

        mask_tumor_region = mask_tumor_dilated2 & ~mask_tumor_dilated1
        stroma_count_region2 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region2 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated1
        gc.collect()

        mask_tumor_dilated3 = Image.open(directory_dilated + file +"_tumor_dilated3.png")
        mask_tumor_dilated3 = np.array(mask_tumor_dilated3)

        mask_tumor_region = mask_tumor_dilated3 & ~mask_tumor_dilated2
        stroma_count_region3 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region3 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated2
        gc.collect()

        mask_tumor_dilated4 = Image.open(directory_dilated + file +"_tumor_dilated4.png")
        mask_tumor_dilated4 = np.array(mask_tumor_dilated4)

        mask_tumor_region = mask_tumor_dilated4 & ~mask_tumor_dilated3
        stroma_count_region4 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region4 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated3
        gc.collect()

        mask_tumor_dilated5 = Image.open(directory_dilated + file +"_tumor_dilated5.png")
        mask_tumor_dilated5 = np.array(mask_tumor_dilated5)

        mask_tumor_region = mask_tumor_dilated5 & ~mask_tumor_dilated4
        stroma_count_region5 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region5 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated4
        gc.collect()

        mask_tumor_dilated6 = Image.open(directory_dilated + file +"_tumor_dilated6.png")
        mask_tumor_dilated6 = np.array(mask_tumor_dilated6)

        mask_tumor_region = mask_tumor_dilated6 & ~mask_tumor_dilated5
        stroma_count_region6 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region6 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated5
        gc.collect()

        mask_tumor_dilated7 = Image.open(directory_dilated + file +"_tumor_dilated7.png")
        mask_tumor_dilated7 = np.array(mask_tumor_dilated7)

        mask_tumor_region = mask_tumor_dilated7 & ~mask_tumor_dilated6
        stroma_count_region7 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region7 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated6
        gc.collect()

        mask_tumor_dilated8 = Image.open(directory_dilated + file +"_tumor_dilated8.png")
        mask_tumor_dilated8 = np.array(mask_tumor_dilated8)

        mask_tumor_region = mask_tumor_dilated8 & ~mask_tumor_dilated7
        stroma_count_region8 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region8 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated7
        gc.collect()

        mask_tumor_dilated9 = Image.open(directory_dilated + file +"_tumor_dilated9.png")
        mask_tumor_dilated9 = np.array(mask_tumor_dilated9)

        mask_tumor_region = mask_tumor_dilated9 & ~mask_tumor_dilated8
        stroma_count_region9 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region9 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated8
        gc.collect()

        mask_tumor_dilated10 = Image.open(directory_dilated + file +"_tumor_dilated10.png")
        mask_tumor_dilated10 = np.array(mask_tumor_dilated10)

        mask_tumor_region = mask_tumor_dilated10 & ~mask_tumor_dilated9
        stroma_count_region10 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region10 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated9
        gc.collect()

        mask_tumor_dilated11 = Image.open(directory_dilated + file +"_tumor_dilated11.png")
        mask_tumor_dilated11 = np.array(mask_tumor_dilated11)

        mask_tumor_region = mask_tumor_dilated11 & ~mask_tumor_dilated10
        stroma_count_region11 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region11 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated10
        gc.collect()

        mask_tumor_dilated12 = Image.open(directory_dilated + file +"_tumor_dilated12.png")
        mask_tumor_dilated12 = np.array(mask_tumor_dilated12)

        mask_tumor_region = mask_tumor_dilated12 & ~mask_tumor_dilated11
        stroma_count_region12 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region12 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated11
        gc.collect()

        mask_tumor_dilated13 = Image.open(directory_dilated + file +"_tumor_dilated13.png")
        mask_tumor_dilated13 = np.array(mask_tumor_dilated13)

        mask_tumor_region = mask_tumor_dilated13 & ~mask_tumor_dilated12
        stroma_count_region13 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region13 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated12
        gc.collect()

        mask_tumor_dilated14 = Image.open(directory_dilated + file +"_tumor_dilated14.png")
        mask_tumor_dilated14 = np.array(mask_tumor_dilated14)

        mask_tumor_region = mask_tumor_dilated14 & ~mask_tumor_dilated13
        stroma_count_region14 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region14 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated13
        gc.collect()

        mask_tumor_dilated15 = Image.open(directory_dilated + file +"_tumor_dilated15.png")
        mask_tumor_dilated15 = np.array(mask_tumor_dilated15)

        mask_tumor_region = mask_tumor_dilated15 & ~mask_tumor_dilated14
        stroma_count_region15 += np.count_nonzero(mask_stroma & mask_tumor_region)
        lymphocyte_count_region15 += np.count_nonzero(mask_lymphocytes & mask_tumor_region)
        del mask_tumor_dilated14
        gc.collect()

        del mask_tumor_dilated15
        del mask_tumor_region
        del mask_lymphocytes
        del mask_stroma

        gc.collect()
    
    
    percentage1 = 100 * (lymphocyte_count_region1 / (stroma_count_region1 + lymphocyte_count_region1))
    percentage2 = 100 * (lymphocyte_count_region2 / (stroma_count_region2 + lymphocyte_count_region2))
    percentage3 = 100 * (lymphocyte_count_region3 / (stroma_count_region3 + lymphocyte_count_region3))
    percentage4 = 100 * (lymphocyte_count_region4 / (stroma_count_region4 + lymphocyte_count_region4))
    percentage5 = 100 * (lymphocyte_count_region5 / (stroma_count_region5 + lymphocyte_count_region5))
    percentage6 = 100 * (lymphocyte_count_region6 / (stroma_count_region6 + lymphocyte_count_region6))
    percentage7 = 100 * (lymphocyte_count_region7 / (stroma_count_region7 + lymphocyte_count_region7))
    percentage8 = 100 * (lymphocyte_count_region8 / (stroma_count_region8 + lymphocyte_count_region8))
    percentage9 = 100 * (lymphocyte_count_region9 / (stroma_count_region9 + lymphocyte_count_region9))
    percentage10 = 100 * (lymphocyte_count_region10 / (stroma_count_region10 + lymphocyte_count_region10))
    percentage11 = 100 * (lymphocyte_count_region11 / (stroma_count_region11 + lymphocyte_count_region11))
    percentage12 = 100 * (lymphocyte_count_region12 / (stroma_count_region12 + lymphocyte_count_region12))
    percentage13 = 100 * (lymphocyte_count_region13 / (stroma_count_region13 + lymphocyte_count_region13))
    percentage14 = 100 * (lymphocyte_count_region14 / (stroma_count_region14 + lymphocyte_count_region14))
    percentage15 = 100 * (lymphocyte_count_region15 / (stroma_count_region15 + lymphocyte_count_region15))
    
    
    data_writer.write(file + "," + str(percentage1) 
                      + "," + str(percentage2)
                      + "," + str(percentage3) 
                      + "," + str(percentage4) 
                      + "," + str(percentage5) 
                      + "," + str(percentage6) 
                      + "," + str(percentage7) 
                      + "," + str(percentage8) 
                      + "," + str(percentage9) 
                      + "," + str(percentage10) 
                      + "," + str(percentage11) 
                      + "," + str(percentage12) 
                      + "," + str(percentage13) 
                      + "," + str(percentage14) 
                      + "," + str(percentage15) + "\n")
    
    data_writer.flush()
    os.fsync(data_writer.fileno())
    
    
        
data_writer.close()


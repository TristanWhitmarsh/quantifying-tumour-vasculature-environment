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
# '685042',
# ]

# names = [['688058'],
#           ['688059','688060'],
#           ['688061'],
#           ['688062'],
#           ['688063'],
#           ['688064'],
#           ['688065'],
#           ['688066','688067'],
#           ['688068'],
#           ['688069'],
#           ['688070'],
#           ['688071'],
#           ['688072'],
#           ['688073'],
#           ['688074'],
#           ['688075'],
#           ['688076'],
#           ['688077'],
#           ['688078'],
#           ['688079','688080','688081','688135'],
#           ['688082','688083'],
#           ['688084'],
#           ['685039','685040'],
#           ['685042']]


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

data_writer = open("regional_analysis/vessel_size_tumor.csv", "w")
data_writer.write("Name, region1, region2, region3, region4, region5, region6, region7, region8, region9, region10, region11, region12, region13, region14, region15\n")
data_writer.flush()
os.fsync(data_writer.fileno())

data_writer_count = open("regional_analysis/vessel_count_tumor.csv", "w")
data_writer_count.write("Name, region1, region2, region3, region4, region5, region6, region7, region8, region9, region10, region11, region12, region13, region14, region15\n")
data_writer_count.flush()
os.fsync(data_writer_count.fileno())


pixel_length = 0.5034
pixel_length_mm = 0.0005034
pixel_size = (pixel_length * pixel_length) # microns
pixel_size_mm = (pixel_length_mm * pixel_length_mm) # microns


for i, patient in enumerate(names):
    print(patient)
    print() 
    
    vessel_size1 = 0
    vessel_count1 = 0
    vessel_size2 = 0
    vessel_count2 = 0
    vessel_size3 = 0
    vessel_count3 = 0
    vessel_size4 = 0
    vessel_count4 = 0
    vessel_size5 = 0
    vessel_count5 = 0
    vessel_size6 = 0
    vessel_count6 = 0
    vessel_size7 = 0
    vessel_count7 = 0
    vessel_size8 = 0
    vessel_count8 = 0
    vessel_size9 = 0
    vessel_count9 = 0
    vessel_size10 = 0
    vessel_count10 = 0
    vessel_size11 = 0
    vessel_count11 = 0
    vessel_size12 = 0
    vessel_count12 = 0
    vessel_size13 = 0
    vessel_count13 = 0
    vessel_size14 = 0
    vessel_count14 = 0
    vessel_size15 = 0
    vessel_count15 = 0
    
    tumor_count_region1 = 0
    tumor_count_region2 = 0
    tumor_count_region3 = 0
    tumor_count_region4 = 0
    tumor_count_region5 = 0
    tumor_count_region6 = 0
    tumor_count_region7 = 0
    tumor_count_region8 = 0
    tumor_count_region9 = 0
    tumor_count_region10 = 0
    tumor_count_region11 = 0
    tumor_count_region12 = 0
    tumor_count_region13 = 0
    tumor_count_region14 = 0
    tumor_count_region15 = 0
    
    for i, file in enumerate(patient):

        mask_vessels = Image.open(directory + file +"_vessel.png")
        mask_vessels = np.array(mask_vessels)

        mask_tumor = Image.open(directory + file +"_tumor.png")
        mask_tumor = np.array(mask_tumor)

        vessel_region_int = np.uint8(mask_vessels)
        numLabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(vessel_region_int, connectivity=8)

        del mask_vessels
        del vessel_region_int
        del labelImage
        gc.collect()

        sizes = stats[:, -1]




        mask_tumor_dilated1 = Image.open(directory_dilated + file +"_tumor_dilated1.png")
        mask_tumor_dilated1 = np.array(mask_tumor_dilated1)

        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated1[y,x] != 0:
                vessel_size1 += size
                vessel_count1 += 1


        mask_tumor_region = mask_tumor_dilated1 & ~mask_tumor
        del mask_tumor
        gc.collect()
        tumor_count_region1 = tumor_count_region1 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated2 = Image.open(directory_dilated + file +"_tumor_dilated2.png")
        mask_tumor_dilated2 = np.array(mask_tumor_dilated2)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated2[y,x] != 0 and mask_tumor_dilated1[y,x] == 0:
                vessel_size2 += size
                vessel_count2 += 1


        mask_tumor_region = mask_tumor_dilated2 & ~mask_tumor_dilated1
        del mask_tumor_dilated1
        gc.collect()
        tumor_count_region2 = tumor_count_region2 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated3 = Image.open(directory_dilated + file +"_tumor_dilated3.png")
        mask_tumor_dilated3 = np.array(mask_tumor_dilated3)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated3[y,x] != 0 and mask_tumor_dilated2[y,x] == 0:
                vessel_size3 += size
                vessel_count3 += 1


        mask_tumor_region = mask_tumor_dilated3 & ~mask_tumor_dilated2
        del mask_tumor_dilated2
        gc.collect()
        tumor_count_region3 = tumor_count_region3 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated4 = Image.open(directory_dilated + file +"_tumor_dilated4.png")
        mask_tumor_dilated4 = np.array(mask_tumor_dilated4)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated4[y,x] != 0 and mask_tumor_dilated3[y,x] == 0:
                vessel_size4 += size
                vessel_count4 += 1


        mask_tumor_region = mask_tumor_dilated4 & ~mask_tumor_dilated3
        del mask_tumor_dilated3
        gc.collect()
        tumor_count_region4 = tumor_count_region4 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated5 = Image.open(directory_dilated + file +"_tumor_dilated5.png")
        mask_tumor_dilated5 = np.array(mask_tumor_dilated5)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated5[y,x] != 0 and mask_tumor_dilated4[y,x] == 0:
                vessel_size5 += size
                vessel_count5 += 1


        mask_tumor_region = mask_tumor_dilated5 & ~mask_tumor_dilated4
        del mask_tumor_dilated4
        gc.collect()
        tumor_count_region5 = tumor_count_region5 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated6 = Image.open(directory_dilated + file +"_tumor_dilated6.png")
        mask_tumor_dilated6 = np.array(mask_tumor_dilated6)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated6[y,x] != 0 and mask_tumor_dilated5[y,x] == 0:
                vessel_size6 += size
                vessel_count6 += 1


        mask_tumor_region = mask_tumor_dilated6 & ~mask_tumor_dilated5
        del mask_tumor_dilated5
        gc.collect()
        tumor_count_region6 = tumor_count_region6 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated7 = Image.open(directory_dilated + file +"_tumor_dilated7.png")
        mask_tumor_dilated7 = np.array(mask_tumor_dilated7)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated7[y,x] != 0 and mask_tumor_dilated6[y,x] == 0:
                vessel_size7 += size
                vessel_count7 += 1


        mask_tumor_region = mask_tumor_dilated7 & ~mask_tumor_dilated6
        del mask_tumor_dilated6
        gc.collect()
        tumor_count_region7 = tumor_count_region7 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated8 = Image.open(directory_dilated + file +"_tumor_dilated8.png")
        mask_tumor_dilated8 = np.array(mask_tumor_dilated8)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated8[y,x] != 0 and mask_tumor_dilated7[y,x] == 0:
                vessel_size8 += size
                vessel_count8 += 1


        mask_tumor_region = mask_tumor_dilated8 & ~mask_tumor_dilated7
        del mask_tumor_dilated7
        gc.collect()
        tumor_count_region8 = tumor_count_region8 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated9 = Image.open(directory_dilated + file +"_tumor_dilated9.png")
        mask_tumor_dilated9 = np.array(mask_tumor_dilated9)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated9[y,x] != 0 and mask_tumor_dilated8[y,x] == 0:
                vessel_size9 += size
                vessel_count9 += 1


        mask_tumor_region = mask_tumor_dilated9 & ~mask_tumor_dilated8
        del mask_tumor_dilated8
        gc.collect()
        tumor_count_region9 = tumor_count_region9 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated10 = Image.open(directory_dilated + file +"_tumor_dilated10.png")
        mask_tumor_dilated10 = np.array(mask_tumor_dilated10)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated10[y,x] != 0 and mask_tumor_dilated9[y,x] == 0:
                vessel_size10 += size
                vessel_count10 += 1


        mask_tumor_region = mask_tumor_dilated10 & ~mask_tumor_dilated9
        del mask_tumor_dilated9
        gc.collect()
        tumor_count_region10 = tumor_count_region10 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated11 = Image.open(directory_dilated + file +"_tumor_dilated11.png")
        mask_tumor_dilated11 = np.array(mask_tumor_dilated11)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated11[y,x] != 0 and mask_tumor_dilated10[y,x] == 0:
                vessel_size11 += size
                vessel_count11 += 1


        mask_tumor_region = mask_tumor_dilated11 & ~mask_tumor_dilated10
        del mask_tumor_dilated10
        gc.collect()
        tumor_count_region11 = tumor_count_region11 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated12 = Image.open(directory_dilated + file +"_tumor_dilated12.png")
        mask_tumor_dilated12 = np.array(mask_tumor_dilated12)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated12[y,x] != 0 and mask_tumor_dilated11[y,x] == 0:
                vessel_size12 += size
                vessel_count12 += 1


        mask_tumor_region = mask_tumor_dilated12 & ~mask_tumor_dilated11
        del mask_tumor_dilated11
        gc.collect()
        tumor_count_region12 = tumor_count_region12 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated13 = Image.open(directory_dilated + file +"_tumor_dilated13.png")
        mask_tumor_dilated13 = np.array(mask_tumor_dilated13)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated13[y,x] != 0 and mask_tumor_dilated12[y,x] == 0:
                vessel_size13 += size
                vessel_count13 += 1


        mask_tumor_region = mask_tumor_dilated13 & ~mask_tumor_dilated12
        del mask_tumor_dilated12
        gc.collect()
        tumor_count_region13 = tumor_count_region13 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated14 = Image.open(directory_dilated + file +"_tumor_dilated14.png")
        mask_tumor_dilated14 = np.array(mask_tumor_dilated14)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated14[y,x] != 0 and mask_tumor_dilated13[y,x] == 0:
                vessel_size14 += size
                vessel_count14 += 1


        mask_tumor_region = mask_tumor_dilated14 & ~mask_tumor_dilated13
        del mask_tumor_dilated13
        gc.collect()
        tumor_count_region14 = tumor_count_region14 + np.count_nonzero(mask_tumor_region)


        mask_tumor_dilated15 = Image.open(directory_dilated + file +"_tumor_dilated15.png")
        mask_tumor_dilated15 = np.array(mask_tumor_dilated15)
        for i in range(1,numLabels):
            size = stats[i, -1]
            x = centroids[i][0].astype(np.int64)
            y = centroids[i][1].astype(np.int64)

            if mask_tumor_dilated15[y,x] != 0 and mask_tumor_dilated14[y,x] == 0:
                vessel_size15 += size
                vessel_count15 += 1

        mask_tumor_region = mask_tumor_dilated15 & ~mask_tumor_dilated14
        del mask_tumor_dilated14
        gc.collect()
        tumor_count_region15 = tumor_count_region15 + np.count_nonzero(mask_tumor_region)

        del mask_tumor_dilated15
        del mask_tumor_region
        gc.collect()


    
    data_writer.write(file + "," + str((vessel_size1 * pixel_size)/vessel_count1) 
                      + "," + str((vessel_size2 * pixel_size)/vessel_count2)
                      + "," + str((vessel_size3 * pixel_size)/vessel_count3) 
                      + "," + str((vessel_size4 * pixel_size)/vessel_count4) 
                      + "," + str((vessel_size5 * pixel_size)/vessel_count5) 
                      + "," + str((vessel_size6 * pixel_size)/vessel_count6) 
                      + "," + str((vessel_size7 * pixel_size)/vessel_count7) 
                      + "," + str((vessel_size8 * pixel_size)/vessel_count8) 
                      + "," + str((vessel_size9 * pixel_size)/vessel_count9) 
                      + "," + str((vessel_size10 * pixel_size)/vessel_count10) 
                      + "," + str((vessel_size11 * pixel_size)/vessel_count11) 
                      + "," + str((vessel_size12 * pixel_size)/vessel_count12) 
                      + "," + str((vessel_size13 * pixel_size)/vessel_count13) 
                      + "," + str((vessel_size14 * pixel_size)/vessel_count14) 
                      + "," + str((vessel_size15 * pixel_size)/vessel_count15) + "\n")
   
    data_writer.flush()
    os.fsync(data_writer.fileno())
    
    
    data_writer_count.write(file + "," + str(vessel_count1/(tumor_count_region1 * pixel_size_mm)) 
                      + "," + str(vessel_count2/(tumor_count_region2 * pixel_size_mm))
                      + "," + str(vessel_count3/(tumor_count_region3 * pixel_size_mm)) 
                      + "," + str(vessel_count4/(tumor_count_region4 * pixel_size_mm)) 
                      + "," + str(vessel_count5/(tumor_count_region5 * pixel_size_mm)) 
                      + "," + str(vessel_count6/(tumor_count_region6 * pixel_size_mm)) 
                      + "," + str(vessel_count7/(tumor_count_region7 * pixel_size_mm)) 
                      + "," + str(vessel_count8/(tumor_count_region8 * pixel_size_mm)) 
                      + "," + str(vessel_count9/(tumor_count_region9 * pixel_size_mm)) 
                      + "," + str(vessel_count10/(tumor_count_region10 * pixel_size_mm)) 
                      + "," + str(vessel_count11/(tumor_count_region11 * pixel_size_mm)) 
                      + "," + str(vessel_count12/(tumor_count_region12 * pixel_size_mm))
                      + "," + str(vessel_count13/(tumor_count_region13 * pixel_size_mm)) 
                      + "," + str(vessel_count14/(tumor_count_region14 * pixel_size_mm)) 
                      + "," + str(vessel_count15/(tumor_count_region15 * pixel_size_mm)) + "\n")
    data_writer_count.flush()
    os.fsync(data_writer_count.fileno())
    
        


        

data_writer.close()
data_writer_count.close()
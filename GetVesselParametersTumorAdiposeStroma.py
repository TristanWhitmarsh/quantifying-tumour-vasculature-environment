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
from scipy import ndimage
import gc

Image.MAX_IMAGE_PIXELS = None


# names = ['688058',
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


# names = [
#     ['688058'],
#     ['688059','688060'],
#     ['688061'],
#     ['688062'],
#     ['688063'],
#     ['688064'],
#     ['688065'],
#     ['688066','688067'],
#     ['688068'],
#     ['688069'],
#     ['688070'],
#     ['688071'],
#     ['688072'],
#     ['688073'],
#     ['688074'],
#     ['688075'],
#     ['688076'],
#     ['688077'],
#     ['688078'],
#     ['688079','688080','688081','688135'],
#     ['688082','688083'],
#     ['688084'],
#     ['685039','685040'],
#     ['685042'],
# ]


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




directory_dilated = "FullInferenceDilatedTumorAdipose/"
directory = "FullInferenceProcessed/"




def measure(name, doprint):
    
    a = datetime.datetime.now()
    
    pixel_length = 0.5034
    pixel_size = (pixel_length * pixel_length) # microns
    
    
    mask_vessel = Image.open(directory + name + "_vessel.png")
    mask_adipose_dilated = Image.open(directory_dilated + name + "_adipose_dilated.png")
    mask_tumor_dilated = Image.open(directory_dilated + name + "_tumor_dilated.png")
    mask_stroma = Image.open(directory + name + "_stroma.png")
    mask_lymphocytes = Image.open(directory + name + "_lymphocytes.png")
    
    mask_vessel = np.array(mask_vessel)
    mask_adipose_dilated = np.array(mask_adipose_dilated)
    mask_tumor_dilated = np.array(mask_tumor_dilated)
    mask_stroma = np.array(mask_stroma)
    mask_lymphocytes = np.array(mask_lymphocytes)
 
    mask_stroma_lymphocytes_all = np.logical_or(mask_lymphocytes, mask_stroma)
    mask_stroma_lymphocytes_all = np.logical_or(mask_stroma_lymphocytes_all, mask_vessel)

    mask_stroma = np.logical_and(mask_stroma_lymphocytes_all, np.logical_not(mask_tumor_dilated))
    mask_stroma = np.logical_and(mask_stroma, np.logical_not(mask_adipose_dilated))
    mask_tumor_stroma = np.logical_and(mask_stroma_lymphocytes_all, mask_tumor_dilated)
    mask_adipose_stroma = np.logical_and(mask_stroma_lymphocytes_all, mask_adipose_dilated)
    
    
#     mask_stroma_binary = mask_stroma > 0
#     mask_stroma_binary = Image.fromarray(mask_stroma_binary)   
#     mask_stroma_binary.save("mask_stroma_binary.png")
    
#     mask_tumor_stroma_binary = mask_tumor_stroma > 0
#     mask_tumor_stroma_binary = Image.fromarray(mask_tumor_stroma_binary)   
#     mask_tumor_stroma_binary.save("mask_tumor_stroma_binary.png")
    
#     mask_adipose_stroma_binary = mask_adipose_stroma > 0
#     mask_adipose_stroma_binary = Image.fromarray(mask_adipose_stroma_binary)   
#     mask_adipose_stroma_binary.save("mask_adipose_stroma_binary.png")

    
    full_area_tumor_stroma = np.count_nonzero(mask_tumor_stroma) * pixel_size
    full_area_adipose_stroma = np.count_nonzero(mask_adipose_stroma) * pixel_size
    full_area_stroma = np.count_nonzero(mask_stroma) * pixel_size

    print("full_area_tumor_stroma {}".format(full_area_tumor_stroma))
    print("full_area_adipose_stroma {}".format(full_area_adipose_stroma))
    print("full_area_stroma {}".format(full_area_stroma))
    
    mask_vessel_int = np.uint8(mask_vessel)
    
    nlabels, labels, stats, centroids  = cv2.connectedComponentsWithStats(mask_vessel_int, connectivity=8)
    statistics = stats[1:,cv2.CC_STAT_AREA]
    
    full_height, full_width = mask_vessel.shape
    
    count_vessels_tumor = 0
    list_ratios_tumor = []
    list_areas_tumor = []
    list_circularities_tumor = []    
    list_thicknesses_tumor = []
    
    count_vessels_adipose = 0
    list_ratios_adipose = []
    list_areas_adipose = []
    list_circularities_adipose = []    
    list_thicknesses_adipose = []
    
    count_vessels_stroma = 0
    list_ratios_stroma = []
    list_areas_stroma = []
    list_circularities_stroma = []    
    list_thicknesses_stroma = []
    
    # for i in range(1, nlabels):
    for i in tqdm(range(1, nlabels)):
        area = stats[i,cv2.CC_STAT_AREA]
        
        if(area > 40):
            
            minx = stats[i,cv2.CC_STAT_LEFT]-1
            miny = stats[i,cv2.CC_STAT_TOP]-1
            maxx = minx + stats[i,cv2.CC_STAT_WIDTH]+2
            maxy = miny + stats[i,cv2.CC_STAT_HEIGHT]+2

            if(minx >= 0 and miny >= 0 and maxx <= full_width and maxy <= full_height ): # remove border vessels
                
    
                mask_adipose_cropped = mask_adipose_dilated[miny:maxy, minx:maxx]
                mask_tumor_cropped = mask_tumor_dilated[miny:maxy, minx:maxx]
                mask_stroma_cropped = mask_stroma[miny:maxy, minx:maxx]
                
                labels_cropped = labels[miny:maxy, minx:maxx]
                mask_cropped = labels_cropped == i
                
                inside_adipose = np.any(mask_cropped & mask_adipose_cropped)
                inside_tumor = np.any(mask_cropped & mask_tumor_cropped)
                inside_stroma = np.any(mask_cropped & mask_stroma_cropped)
                
                # cv2.imwrite('E:/U-Net/mask_cropped.png', 255*mask_cropped)
                height, width = mask_cropped.shape
                # print("height, width {} {}".format(height, width))
                
                mask_cropped_int = np.uint8(mask_cropped)
                
                
                if True:
                    distance_map = cv2.distanceTransform(mask_cropped_int,cv2.DIST_L2, 5)
                    skeleton = skeletonize(mask_cropped)
                    
                    thickness_map = mask_cropped.copy()
                    thickness_map.fill(0)
                    thickness_map = thickness_map.astype(np.float64)
                    
                    tmp_map = mask_cropped.copy()
                    tmp_map.fill(0)
                    tmp_map = tmp_map.astype(np.float64)
                    
                    
                    for x in range(0, width):
                        for y in range(0, height):
                            if mask_cropped[y,x] != 0 and skeleton[y,x] != 0:
                                radius = np.round(distance_map[y,x]).astype("int")
                                diameter = int(2*radius)
                                cv2.circle(tmp_map, (x,y), radius, diameter, -1)
                                thickness_map = np.maximum(thickness_map, tmp_map)
                    
                    thickness_total = 0
                    thickness_count = 0
                    for x in range(0, width):
                        for y in range(0, height):
                            if(mask_cropped[y,x]>0):
                                t = thickness_map[y,x]
                                if(t>0):
                                    thickness_total += t
                                    thickness_count += 1

                    thickness = thickness_total / thickness_count
                    thickness *= pixel_length
                

                im2, contours, hierarchy = cv2.findContours(mask_cropped_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
                perimeter = cv2.arcLength(contours[0],True)
                contour_area = cv2.contourArea(contours[0])
                # this is not nessesary but added for completeness
                perimeter = perimeter * pixel_length # convert to um
                contour_area = contour_area * pixel_size # convert to um2
                
                area = area * pixel_size # convert to um2
                # print('area {:.2f}'.format(area))
                circularity = (4.0 * math.pi * contour_area) / pow(perimeter, 2)
            
                
                rect = cv2.minAreaRect(contours[0])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # cv2.drawContours(thickness_map_norm_color,[box],0,(0,0,255),2)
                (x, y), (w, h), angle = rect
                w += 1
                h += 1
                axis_ratio = max(w/h,h/w)
                
                
                if inside_adipose:
                    count_vessels_adipose = count_vessels_adipose + 1
                    list_ratios_adipose.append(axis_ratio)
                    list_areas_adipose.append(area)
                    list_circularities_adipose.append(circularity)
                    list_thicknesses_adipose.append(thickness)
                    
                if inside_tumor:
                    count_vessels_tumor = count_vessels_tumor + 1
                    list_ratios_tumor.append(axis_ratio)
                    list_areas_tumor.append(area)
                    list_circularities_tumor.append(circularity)
                    list_thicknesses_tumor.append(thickness)
                
                if inside_stroma:
                    count_vessels_stroma = count_vessels_stroma + 1
                    list_ratios_stroma.append(axis_ratio)
                    list_areas_stroma.append(area)
                    list_circularities_stroma.append(circularity)
                    list_thicknesses_stroma.append(thickness)
                    
    
    del mask_vessel
    del mask_adipose_stroma
    del mask_adipose_dilated
    del mask_tumor_stroma
    del mask_tumor_dilated
    del mask_stroma
    del mask_lymphocytes


    gc.collect()
    
    b = datetime.datetime.now()
    delta = b - a
    print("this took {}".format(delta))
    a = datetime.datetime.now()
    
    return count_vessels_tumor, full_area_tumor_stroma, list_areas_tumor, list_circularities_tumor, list_ratios_tumor, list_thicknesses_tumor,\
        count_vessels_adipose, full_area_adipose_stroma, list_areas_adipose, list_circularities_adipose, list_ratios_adipose, list_thicknesses_adipose,\
        count_vessels_stroma, full_area_stroma, list_areas_stroma, list_circularities_stroma, list_ratios_stroma, list_thicknesses_stroma



data_writer = open("measurements/measurements_vessel_full_slide.csv", "w")
data_writer.write("Name,region,density,mean_area,mean_circularity,mean_axis_ratio,mean_thickness\n")

data_writer.flush()
os.fsync(data_writer.fileno())



for i, patient in enumerate(names):
    print(patient)
    print()
    
    count_vessels_tumor = 0
    full_area_tumor_stroma = 0
    list_areas_tumor = []
    list_circularities_tumor = []   
    list_ratios_tumor = [] 
    list_thicknesses_tumor = []
    
        
    count_vessels_adipose = 0
    full_area_adipose_stroma = 0
    list_areas_adipose = []
    list_circularities_adipose = []    
    list_ratios_adipose = []
    list_thicknesses_adipose = []
    
        
    count_vessels_stroma = 0
    full_area_stroma = 0
    list_areas_stroma = []
    list_circularities_stroma = []    
    list_ratios_stroma = []
    list_thicknesses_stroma = []
    
    for i, name in enumerate(patient):

        count_vessels_tumor2, full_area_tumor_stroma2, list_areas_tumor2, list_circularities_tumor2, list_ratios_tumor2, list_thicknesses_tumor2,\
        count_vessels_adipose2, full_area_adipose_stroma2, list_areas_adipose2, list_circularities_adipose2, list_ratios_adipose2, list_thicknesses_adipose2,\
        count_vessels_stroma2, full_area_stroma2, list_areas_stroma2, list_circularities_stroma2, list_ratios_stroma2, list_thicknesses_stroma2 = measure(name, True)

        
        count_vessels_tumor = count_vessels_tumor + count_vessels_tumor2
        full_area_tumor_stroma = full_area_tumor_stroma + full_area_tumor_stroma2
        list_areas_tumor = list_areas_tumor + list_areas_tumor2
        list_circularities_tumor = list_circularities_tumor + list_circularities_tumor2 
        list_ratios_tumor = list_ratios_tumor + list_ratios_tumor2  
        list_thicknesses_tumor = list_thicknesses_tumor + list_thicknesses_tumor2
        
        count_vessels_stroma = count_vessels_stroma + count_vessels_stroma2
        full_area_stroma = full_area_stroma + full_area_stroma2
        list_areas_stroma = list_areas_stroma + list_areas_stroma2
        list_circularities_stroma = list_circularities_stroma + list_circularities_stroma2 
        list_ratios_stroma = list_ratios_stroma + list_ratios_stroma2  
        list_thicknesses_stroma = list_thicknesses_stroma + list_thicknesses_stroma2
        
        count_vessels_adipose = count_vessels_adipose + count_vessels_adipose2
        full_area_adipose_stroma = full_area_adipose_stroma + full_area_adipose_stroma2
        list_areas_adipose = list_areas_adipose + list_areas_adipose2
        list_circularities_adipose = list_circularities_adipose + list_circularities_adipose2 
        list_ratios_adipose = list_ratios_adipose + list_ratios_adipose2  
        list_thicknesses_adipose = list_thicknesses_adipose + list_thicknesses_adipose2


    density_tumor = 1000000*(count_vessels_tumor/full_area_tumor_stroma)
    mean_area_tumor = np.mean(list_areas_tumor)
    mean_circularity_tumor = np.mean(list_circularities_tumor)
    mean_axis_tumor = np.mean(list_ratios_tumor)
    mean_thickness_tumor = np.mean(list_thicknesses_tumor)

    density_adipose = 1000000*(count_vessels_adipose/full_area_adipose_stroma)
    mean_area_adipose = np.mean(list_areas_adipose)
    mean_circularity_adipose = np.mean(list_circularities_adipose)
    mean_axis_adipose = np.mean(list_ratios_adipose)
    mean_thickness_adipose = np.mean(list_thicknesses_adipose)

    density_stroma = 1000000*(count_vessels_stroma/full_area_stroma)
    mean_area_stroma = np.mean(list_areas_stroma)
    mean_circularity_stroma = np.mean(list_circularities_stroma)
    mean_axis_stroma = np.mean(list_ratios_stroma)
    mean_thickness_stroma = np.mean(list_thicknesses_stroma)
    
    print()
    print('density tumor: {:.5f} #/mm2'.format(density_tumor))
    print('mean area tumor: {:.5f} um'.format(mean_area_tumor))
    print('mean circularity tumor: {:.5f}'.format(mean_circularity_tumor))
    print('mean major/minor axis ratio tumor: {:.5f}'.format(mean_axis_tumor))
    print('mean thickness tumor: {:.5f} um'.format(mean_thickness_tumor))

    print('density adipose: {:.5f} #/mm2'.format(density_adipose))
    print('mean area adipose: {:.5f} um'.format(mean_area_adipose))
    print('mean circularity adipose: {:.5f}'.format(mean_circularity_adipose))
    print('mean major/minor axis ratio adipose: {:.5f}'.format(mean_axis_adipose))
    print('mean thickness adipose: {:.5f} um'.format(mean_thickness_adipose))
    
    print('density stroma: {:.5f} #/mm2'.format(density_stroma))
    print('mean area stroma: {:.5f} um'.format(mean_area_stroma))
    print('mean circularity stroma: {:.5f}'.format(mean_circularity_stroma))
    print('mean major/minor axis ratio stroma: {:.5f}'.format(mean_axis_stroma))
    print('mean thickness stroma: {:.5f} um'.format(mean_thickness_stroma))
    
    data_writer.write(name + ",tumor," + str(density_tumor) 
                      + "," + str(mean_area_tumor) 
                      + "," + str(mean_circularity_tumor) 
                      + "," + str(mean_axis_tumor) 
                      + "," + str(mean_thickness_tumor) + "\n")
    
    data_writer.write(name + ",adipose," + str(density_adipose) 
                      + "," + str(mean_area_adipose) 
                      + "," + str(mean_circularity_adipose) 
                      + "," + str(mean_axis_adipose) 
                      + "," + str(mean_thickness_adipose) + "\n")
    
    data_writer.write(name + ",stroma," + str(density_stroma) 
                      + "," + str(mean_area_stroma) 
                      + "," + str(mean_circularity_stroma) 
                      + "," + str(mean_axis_stroma) 
                      + "," + str(mean_thickness_stroma) + "\n")

    
    data_writer.flush()
    os.fsync(data_writer.fileno())
        
data_writer.close()
        
        
        
        
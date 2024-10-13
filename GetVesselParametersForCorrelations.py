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



def measure(imagefile, doprint):
    
    a = datetime.datetime.now()
    
    pixel_length = 0.5034
    pixel_length_mm = 0.0005034
    pixel_size = (pixel_length * pixel_length) # microns
    
    
    # imagefile = 'E:/U-Net/masksBEHOLD_inference/688066_vessel_(1.00,25923,25694,4608,6144)-mask.png'
    # imagefile = inputfile
    # imagefile = 'E:/U-Net/test.png'
    mask = cv2.imread(imagefile, 0)
    
    
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    nlabels, labels, stats, centroids  = cv2.connectedComponentsWithStats(mask, connectivity=8)
    statistics = stats[1:,cv2.CC_STAT_AREA]
    
    
    # def remove_small_zero(mask_pred, threshold):
    #     nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_pred.astype(np.uint8), connectivity=4)
    #     #print(nb_components)
    #     sizes = stats[1:, -1];
    #     nb_components = nb_components - 1
    #     min_size = threshold
    #     for i in range(0, nb_components):
    #         if sizes[i] < min_size:
    #             mask_pred[output == i + 1] = 0 
                
    # remove_small_zero(mask, 10)
    
    full_height, full_width = mask.shape
    count_vessels = 0
    list_ratios = []
    list_areas = []
    list_contourareas = []
    list_circularities = []    
    list_thicknesses = []
    
    # for i in range(1, nlabels):
    for i in tqdm(range(1, nlabels)):
        area = stats[i,cv2.CC_STAT_AREA]
        
        if(area >= 40):
            onelabel = labels == i
            onelabel = np.uint8(onelabel)
            
            minx = stats[i,cv2.CC_STAT_LEFT]-1
            miny = stats[i,cv2.CC_STAT_TOP]-1
            maxx = minx + stats[i,cv2.CC_STAT_WIDTH]+2
            maxy = miny + stats[i,cv2.CC_STAT_HEIGHT]+2
    
            if(minx >= 0 and miny >= 0 and maxx <= full_width and maxy <= full_height ): # remove border vessels
                count_vessels += 1
    
                mask_cropped = onelabel[miny:maxy, minx:maxx]
                # cv2.imwrite('E:/U-Net/mask_cropped.png', 255*mask_cropped)
                height, width = mask_cropped.shape
                
                distance_map = cv2.distanceTransform(mask_cropped,cv2.DIST_L2, 5)
                skeleton = skeletonize(mask_cropped)
                
                kernel = np.ones((3,3),np.uint8)
                skeleton = skeleton.astype(np.uint8)
                skeleton = cv2.dilate(skeleton, kernel)
                
                thickness_map = mask_cropped.copy()
                thickness_map.fill(0)
                
                tmp_map = mask_cropped.copy()
                tmp_map.fill(0)
                tmp_map = tmp_map.astype(np.float64)
                
                # for x in range(0, width):
                #     for y in range(0, height):
                #         if mask_cropped[y,x] != 0 and skeleton[y,x] != 0:
                #             radius = np.round(distance_map[y,x]).astype("int")
                #             cv2.circle(tmp_map, (x,y), radius, 2*(int(distance_map[y,x])), -1)
                #             thickness_map = np.maximum(thickness_map, tmp_map)
                
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
                
               
                # cv2.imwrite('E:/U-Net/thickness_map.png', 10*thickness_map)
        
                
                # onelabel = labels == i
                # onelabel = np.uint8(onelabel)
                im2, contours, hierarchy = cv2.findContours(mask_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
                # cv2.drawContours(thickness_map_norm_color, contours, -1, (0,255,0), 2)
                
                
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
                # print('w,h {} {}'.format(w,h))
                
                list_ratios.append(axis_ratio)
                list_areas.append(area)
                list_contourareas.append(contour_area)
                list_circularities.append(circularity)
                list_thicknesses.append(thickness)
                
            
                # print()
                # print('perimeter {:.5f}'.format(perimeter))
                # print('area {:.5f}'.format(area))
                # print('contour_area {:.5f}'.format(contour_area))
                # print('circularity {:.5f}'.format(circularity))
                # print('axis_ratio {:.5f}'.format(axis_ratio))
                # print('thickness {:.5f}'.format(thickness))
                # print()
        
    
    
    full_pixel_count = full_width * full_height;
    full_area = full_pixel_count * pixel_size
    
    if doprint:
        density = 1000000*(count_vessels/full_area)
        vessel_area = np.sum(list_areas)
        area_fraction = 100*(vessel_area/full_area)

        median_area = np.median(list_areas)
        median_circularity = np.median(list_circularities)
        median_axis = np.median(list_ratios)
        median_thickness = np.median(list_thicknesses)

        mean_area = np.mean(list_areas)
        mean_contourarea = np.mean(list_contourareas)
        mean_circularity = np.mean(list_circularities)
        mean_axis = np.mean(list_ratios)
        mean_thickness = np.mean(list_thicknesses)
    
        print()
        print('density: {:.5f} #/mm2'.format(density))
        print('area fraction: {:.5f} %'.format(area_fraction))
        print('median area: {:.5f} um'.format(median_area))
        print('median circularity: {:.5f}'.format(median_circularity))
        print('median major/minor axis ratio: {:.5f}'.format(median_axis))
        print('median thickness: {:.5f} um'.format(median_thickness))
        print('mean area: {:.5f} um'.format(mean_area))
        print('mean contour area: {:.5f} um'.format(mean_contourarea))
        print('mean circularity: {:.5f}'.format(mean_circularity))
        print('mean major/minor axis ratio: {:.5f}'.format(mean_axis))
        print('mean thickness: {:.5f} um'.format(mean_thickness))
        
    
    b = datetime.datetime.now()
    delta = b - a
    print("this took {}".format(delta))
    a = datetime.datetime.now()
    
    return count_vessels, full_area, list_areas, list_circularities, list_ratios, list_thicknesses
    
    # cv2.imwrite('E:/U-Net/result_measurements.png', thickness_map_norm_color)
    
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
    "688085_(1.00,13134,1089,14848,4096)"]

# MISSION
names = [['684913'],
          ['684914'],
          ['684916'],
          ['685039','685040'],
          ['685041'],
          ['685042'],
          ['685281']]

# BEHOLD
# names = [['688058'],
#           ['688059','688060'],
#           ['688061'],
#           ['688062'],
#           ['688063'],
#           ['688064'],
#           ['688065'],
#           ['688066','688067'],
#           ['688068','688069'],
#           ['688070'],
#           ['688071'],
#           ['688072'],
#           ['688073'],
#           ['688074'],
#           ['688075', '688076'],
#           ['688077'],
#           ['688078'],
#           ['688079','688080','688081','688135'],
#           ['688082','688083'],
#           ['688084']]



# directory = '/home/user/GitRepositories/Vessel/masksBEHOLD_corrected/'
# data_writer = open("/home/user/GitRepositories/Vessel/parameters/measurements_vessel_BEHOLD_true.csv", "w")

# directory = '/home/user/GitRepositories/Vessel/Inference_BEHOLD/'
# data_writer = open("/home/user/GitRepositories/Vessel/parameters/measurements_vessel_BEHOLD_predicted.csv", "w")

# directory = '/home/user/GitRepositories/Vessel/masksMISSION_corrected/'
# data_writer = open("/home/user/GitRepositories/Vessel/parameters/measurements_vessel_MISSION_true.csv", "w")

directory = '/home/user/GitRepositories/Vessel/Inference_MISSION/'
data_writer = open("/home/user/GitRepositories/Vessel/parameters/measurements_vessel_MISSION_predicted.csv", "w")



data_writer.write("Name, density, area fraction, median area, median circularity, median major/minor axis ratio, median thickness, mean area, mean circularity, mean major/minor axis ratio, mean thickness\n")
data_writer.flush()
os.fsync(data_writer.fileno())


for i, patient in enumerate(names):
    print(patient)
    print() 
    
    total_count_vessels = 0
    total_full_area = 0
    total_list_areas = []
    total_list_circularities = []
    total_list_ratios = []
    total_list_thicknesses = []
    
    for i, name in enumerate(patient):
        # print(name)
        # print() 
        
        os.chdir(directory)
        for file in glob.glob("*.png"):
            
            imageID = os.path.basename(file[:-4])
            
            imageID2 = imageID.split("-")
            imageID3 = imageID2[0].split("_")
            imageID4 = imageID3[0] + "_" + imageID3[2]
            
            imageID2 = imageID.split("_")
            imageID5 = imageID2[0] + "_" + imageID2[1]
            
            
            if name in file and ('Vessel' in file or 'vessel' in file) and (imageID4 in exclude or imageID5 in exclude):
                print("excluding " + file)
            
            if name in file and ('Vessel' in file or 'vessel' in file) and not (imageID4 in exclude or imageID5 in exclude):               
                  
                print("measuring " + file)
                count_vessels, full_area, list_areas, list_circularities, list_ratios, list_thicknesses = measure(file, True)
                
                total_count_vessels = total_count_vessels + count_vessels
                total_full_area = total_full_area + full_area
                
                total_list_areas.extend(list_areas)
                total_list_circularities.extend(list_circularities)
                total_list_ratios.extend(list_ratios)
                total_list_thicknesses.extend(list_thicknesses)
                print(len(total_list_areas))
    

    density = 1000000*total_count_vessels/total_full_area
    area_fraction = 100*np.sum(total_list_areas)/total_full_area
    
    median_area = np.median(total_list_areas)
    median_circularity = np.median(total_list_circularities)
    median_axis = np.median(total_list_ratios)
    median_thickness = np.median(total_list_thicknesses)
        
    mean_area = np.mean(total_list_areas)
    mean_circularity = np.mean(total_list_circularities)
    mean_axis = np.mean(total_list_ratios)
    mean_thickness = np.mean(total_list_thicknesses)
    
    print()
    print('density: {:.5f} #/mm2'.format(density))
    print('area fraction: {:.5f} %'.format(area_fraction))
    print('median area: {:.5f} um'.format(median_area))
    print('median circularity: {:.5f}'.format(median_circularity))
    print('median major/minor axis ratio: {:.5f}'.format(median_axis))
    print('median thickness: {:.5f} um'.format(median_thickness))
    print('mean area: {:.5f} um'.format(mean_area))
    print('mean circularity: {:.5f}'.format(mean_circularity))
    print('mean major/minor axis ratio: {:.5f}'.format(mean_axis))
    print('mean thickness: {:.5f} um'.format(mean_thickness))
    
    data_writer.write(name + "," + str(density) 
                      + "," + str(area_fraction)
                      + "," + str(median_area) 
                      + "," + str(median_circularity) 
                      + "," + str(median_axis) 
                      + "," + str(median_thickness)
                      + "," + str(mean_area) 
                      + "," + str(mean_circularity) 
                      + "," + str(mean_axis) 
                      + "," + str(mean_thickness) + "\n")
    data_writer.flush()
    os.fsync(data_writer.fileno())
        
data_writer.close()
        
        
        
        
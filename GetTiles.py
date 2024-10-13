
from __future__ import print_function, division

import cv2
import numpy as np
import shutil
import os
import math
from pathlib import Path
import glob, os

if __name__ == '__main__':
    
    delete_all = True
    
    
    dirName = 'data/'
    if delete_all:
        try:
           shutil.rmtree(dirName)
        except:
           print('Could not delete directory')
             
    
    
    
    #directory = 'C:/U-Net/masks/'
    directory = 'masks_corrected/'
    
    #os.chdir(directory)
    for file in glob.glob("masks/*.jpg"):
        # print(file[6:-4])
        imageID = os.path.basename(file[:-4])
        # filename2 = os.path.basename(file[:-4])
        
        # print(filename2)
    
    # for imageID in images:
        print(imageID)
        # patientID = imageID[:-2]
        patientID = imageID.split("_")
        print(patientID[0])
        
        img = cv2.imread(directory + imageID + '.jpg')
        img_copy = img.copy()
        mask_vessel = cv2.imread(directory + patientID[0] + '_vessel_' + patientID[1] + '-mask.png', 0)
        mask_tumor = cv2.imread(directory + patientID[0] + '_Tumor_' + patientID[1] + '-mask.png', 0)
        mask_benign = cv2.imread(directory + patientID[0] + '_Benign_' + patientID[1] + '-mask.png', 0)
        mask_adipose = cv2.imread(directory + patientID[0] + '_Adipose_' + patientID[1] + '-mask.png', 0)
        mask_background = cv2.imread(directory + patientID[0] + '_Background_' + patientID[1] + '-mask.png', 0)
        mask_lymphocytes = cv2.imread(directory + patientID[0] + '_Lymphocytes_' + patientID[1] + '-mask.png', 0)
        mask_macrophages = cv2.imread(directory + patientID[0] + '_Macrophages_' + patientID[1] + '-mask.png', 0)
        mask_muscle = cv2.imread(directory + patientID[0] + '_Muscle_' + patientID[1] + '-mask.png', 0)
        mask_nerve = cv2.imread(directory + patientID[0] + '_Nerve_' + patientID[1] + '-mask.png', 0)
        
        
        print('images read')
    
    
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        dirName2 = dirName + patientID[0]
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/image/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/mask_vessel/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/mask_tumor/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/mask_benign/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/mask_adipose/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/mask_background/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/mask_lymphocytes/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/mask_macrophages/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/mask_muscle/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
        dirName2 = dirName + patientID[0] + '/mask_nerve/'
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
            
        print('folders made')
        
        height, width = img.shape[:2]
        
        input_size = 512#640
        
        #tile_size = round(math.sqrt((input_size*input_size)+(input_size*input_size)))
        tile_size = 512#640 
        #stepsizeX = input_size//2
        #stepsizeY = input_size//2
        stepsizeX = tile_size
        stepsizeY = tile_size
     
        x_range = width//stepsizeX
        y_range = height//stepsizeY
        
        #border = (tile_size-input_size)//2
        count = 0
        #for k in range(2):
        #print(k)
        for i in range(0, x_range):
            for j in range(0, y_range):
                
                x = i*stepsizeX
                y = j*stepsizeY
                
                img_cropped = img[y:y+tile_size, x:x+tile_size]
                
                
                tile_height, tile_width = img_cropped.shape[:2]
                if tile_height == tile_size and tile_width == tile_size:
                        
                    mask_vessel_cropped = mask_vessel[y:y+tile_size, x:x+tile_size]
                    mask_tumor_cropped = mask_tumor[y:y+tile_size, x:x+tile_size]
                    mask_benign_cropped = mask_benign[y:y+tile_size, x:x+tile_size]
                    mask_adipose_cropped = mask_adipose[y:y+tile_size, x:x+tile_size]
                    mask_background_cropped = mask_background[y:y+tile_size, x:x+tile_size]
                    mask_lymphocytes_cropped = mask_lymphocytes[y:y+tile_size, x:x+tile_size]
                    mask_macrophages_cropped = mask_macrophages[y:y+tile_size, x:x+tile_size]
                    mask_muscle_cropped = mask_muscle[y:y+tile_size, x:x+tile_size]
                    mask_nerve_cropped = mask_nerve[y:y+tile_size, x:x+tile_size]

                    
                    # mask_vessel_cropped2 = mask_vessel_cropped
                    
                    # mask_macrophages_cropped2 = cv2.subtract(mask_macrophages_cropped, mask_vessel_cropped)
                    
                    # mask_muscle_cropped2 = cv2.subtract(mask_muscle_cropped, mask_macrophages_cropped)
                    # mask_muscle_cropped2 = cv2.subtract(mask_muscle_cropped2, mask_vessel_cropped)
                    
                    # mask_nerve_cropped2 = cv2.subtract(mask_nerve_cropped, mask_muscle_cropped)
                    # mask_nerve_cropped2 = cv2.subtract(mask_nerve_cropped2, mask_macrophages_cropped)
                    # mask_nerve_cropped2 = cv2.subtract(mask_nerve_cropped2, mask_vessel_cropped)
                    
                    # mask_adipose_cropped2 = cv2.subtract(mask_adipose_cropped, mask_nerve_cropped)
                    # mask_adipose_cropped2 = cv2.subtract(mask_adipose_cropped2, mask_muscle_cropped)
                    # mask_adipose_cropped2 = cv2.subtract(mask_adipose_cropped2, mask_macrophages_cropped)
                    # mask_adipose_cropped2 = cv2.subtract(mask_adipose_cropped2, mask_vessel_cropped)
                    
                    # mask_tumor_cropped2 = cv2.subtract(mask_tumor_cropped, mask_adipose_cropped)
                    # mask_tumor_cropped2 = cv2.subtract(mask_tumor_cropped2, mask_nerve_cropped)
                    # mask_tumor_cropped2 = cv2.subtract(mask_tumor_cropped2, mask_muscle_cropped)
                    # mask_tumor_cropped2 = cv2.subtract(mask_tumor_cropped2, mask_macrophages_cropped)
                    # mask_tumor_cropped2 = cv2.subtract(mask_tumor_cropped2, mask_vessel_cropped)
                    
                    # mask_benign_cropped2 = cv2.subtract(mask_benign_cropped, mask_tumor_cropped)
                    # mask_benign_cropped2 = cv2.subtract(mask_benign_cropped2, mask_adipose_cropped)
                    # mask_benign_cropped2 = cv2.subtract(mask_benign_cropped2, mask_nerve_cropped)
                    # mask_benign_cropped2 = cv2.subtract(mask_benign_cropped2, mask_muscle_cropped)
                    # mask_benign_cropped2 = cv2.subtract(mask_benign_cropped2, mask_macrophages_cropped)
                    # mask_benign_cropped2 = cv2.subtract(mask_benign_cropped2, mask_vessel_cropped)
                    
                    # mask_lymphocytes_cropped2 = cv2.subtract(mask_lymphocytes_cropped, mask_benign_cropped)
                    # mask_lymphocytes_cropped2 = cv2.subtract(mask_lymphocytes_cropped2, mask_tumor_cropped)
                    # mask_lymphocytes_cropped2 = cv2.subtract(mask_lymphocytes_cropped2, mask_adipose_cropped)
                    # mask_lymphocytes_cropped2 = cv2.subtract(mask_lymphocytes_cropped2, mask_nerve_cropped)
                    # mask_lymphocytes_cropped2 = cv2.subtract(mask_lymphocytes_cropped2, mask_muscle_cropped)
                    # mask_lymphocytes_cropped2 = cv2.subtract(mask_lymphocytes_cropped2, mask_macrophages_cropped)
                    # mask_lymphocytes_cropped2 = cv2.subtract(mask_lymphocytes_cropped2, mask_vessel_cropped)
                    
                    # mask_background_cropped2 = cv2.subtract(mask_background_cropped, mask_lymphocytes_cropped)
                    # mask_background_cropped2 = cv2.subtract(mask_background_cropped2, mask_benign_cropped)
                    # mask_background_cropped2 = cv2.subtract(mask_background_cropped2, mask_tumor_cropped)
                    # mask_background_cropped2 = cv2.subtract(mask_background_cropped2, mask_adipose_cropped)
                    # mask_background_cropped2 = cv2.subtract(mask_background_cropped2, mask_nerve_cropped)
                    # mask_background_cropped2 = cv2.subtract(mask_background_cropped2, mask_muscle_cropped)
                    # mask_background_cropped2 = cv2.subtract(mask_background_cropped2, mask_macrophages_cropped)
                    # mask_background_cropped2 = cv2.subtract(mask_background_cropped2, mask_vessel_cropped)
                    
                    #if cv2.countNonZero(mask_vessel_cropped) > 0:
                    if True:
                    
                        cv2.imwrite(dirName + patientID[0] + '/image/' + imageID + '_' + str(i) + '-' + str(j) + '.jpg', img_cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        cv2.imwrite(dirName + patientID[0] + '/mask_vessel/' + imageID + '_' + str(i) + '-' + str(j) + '.png', mask_vessel_cropped)
                        cv2.imwrite(dirName + patientID[0] + '/mask_tumor/' + imageID + '_' + str(i) + '-' + str(j) + '.png', mask_tumor_cropped)
                        cv2.imwrite(dirName + patientID[0] + '/mask_benign/' + imageID + '_' + str(i) + '-' + str(j) + '.png', mask_benign_cropped)
                        cv2.imwrite(dirName + patientID[0] + '/mask_adipose/' + imageID + '_' + str(i) + '-' + str(j) + '.png', mask_adipose_cropped)
                        cv2.imwrite(dirName + patientID[0] + '/mask_background/' + imageID + '_' + str(i) + '-' + str(j) + '.png', mask_background_cropped)
                        cv2.imwrite(dirName + patientID[0] + '/mask_lymphocytes/' + imageID + '_' + str(i) + '-' + str(j) + '.png', mask_lymphocytes_cropped)
                        cv2.imwrite(dirName + patientID[0] + '/mask_macrophages/' + imageID + '_' + str(i) + '-' + str(j) + '.png', mask_macrophages_cropped)
                        cv2.imwrite(dirName + patientID[0] + '/mask_muscle/' + imageID + '_' + str(i) + '-' + str(j) + '.png', mask_muscle_cropped)
                        cv2.imwrite(dirName + patientID[0] + '/mask_nerve/' + imageID + '_' + str(i) + '-' + str(j) + '.png', mask_nerve_cropped)

                        #cv2.circle(img_copy,(x+int(tile_size/2),y+int(tile_size/2)), 256, (0,0,255), 4)
                        #cv2.rectangle(img_copy,(x,y),(x+tile_size,y+tile_size),(64,128,0),4)
                        #cv2.rectangle(img_copy,(x+50,y+50),(x+tile_size-50,y+tile_size-50),(0,128,64),4)
                        #cv2.rectangle(img_copy,(x+border,y+border),(x+tile_size-border,y+tile_size-border),(255,0,0),4)
                
                        count = count + 1
                        
        #cv2.imwrite('C:/U-Net/tiling/' + 'tiling_'+imageID+'.jpg', img_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])
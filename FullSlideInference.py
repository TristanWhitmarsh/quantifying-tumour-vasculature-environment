import slideio
from pathlib import Path
import matplotlib.pyplot as plt
import os


import numpy as np
from keras.layers import Conv2D as _Conv2D
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    LeakyReLU,
    BatchNormalization
)
from keras.models import Model
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import backend as K

def Conv2D(filters, size, activation=LeakyReLU(alpha=0.1), kernel_initializer="he_normal"):
    return _Conv2D(
        filters, 
        size, 
        activation=activation, 
        padding="same", 
        kernel_initializer=kernel_initializer,
    )


def get_standard_model(input_size, num_classes):
    
    inputs = keras.Input(shape=input_size + (3,))
    #inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3)(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3)(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3)(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2D(512, 2)(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3)(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2)(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3)(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2)(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3)(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2)(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3)(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3)(conv9)
    conv9 = BatchNormalization()(conv9)
    
    #conv10 = Conv2D(num_classes, 3)(conv9)
    conv10 = _Conv2D(num_classes, 1, activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=conv10, name='Unet')

    return model





def infer(img, filename, model):

    img_size_x = 512
    img_size_y = 512
    img_size = (img_size_x, img_size_y)
    num_classes = 10
    padding = 384
    stepsize = 384
    border = 0

    print('make distance map')
    weights = np.ones((512,512))
    weights[256,256] = 0
    weights[255,256] = 0
    weights[256,255] = 0
    weights[256,255] = 0
    
    distance_map = distance_transform_edt(weights)
    max_value = np.amax(distance_map)
    distance_map = distance_map / (max_value+1)
    distance_map = (1-distance_map)
    print(distance_map.shape)

    distance_map2 = np.expand_dims(distance_map, axis=0)
    distance_map2 = np.repeat(distance_map2[:,:], 10, axis=0)
    print(distance_map2.shape)


    # import matplotlib.pyplot as plt
    # plt.rcParams["figure.figsize"] = (25,25)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(5,5,1)
    # ax1.imshow(distance_map2[9,:,:])

    #model = keras.models.load_model("standard.h5", compile=True, options=None, custom_objects={ 'softmax_cross_entropy_with_logits_v2': softmax_cross_entropy_with_logits_v2, 'soft_dice_loss': soft_dice_loss, 'binary_crossentropy_Unet': binary_crossentropy_Unet, 'LeakyReLU': LeakyReLU})
    #model = get_standard_model(img_size, num_classes)
    #model.load_weights('chk/model100.h5')


    directory_out = './FullInference/'

    
    print('do padding')
    img = np.pad(img, ((padding,padding),(padding,padding),(0,0)), mode='reflect')
    height, width = img.shape[:2]
    
    print('make image buffers')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_map = zeros([10,height,width])
    img_map_count = zeros([height,width])
    
    x_range = math.floor(width/stepsize)
    y_range = math.floor(height/stepsize)
    

    #print("predicting")
    #for i in range(0, x_range):
    
    #single tile prediction
    if False:
        for i in tqdm(range(0, x_range)):
            for j in range(0, y_range):

                x = i*stepsize
                y = j*stepsize
                h = 512
                w = 512
                img_cropped_orig = img[y:y+h, x:x+w]

                if img_cropped_orig.shape[0] == h and img_cropped_orig.shape[1] == w:

                    img_cropped_orig = np.expand_dims(img_cropped_orig, axis=0)
                    #print(img_cropped_orig.shape)

                    mask_pred_orig = model(img_cropped_orig)

                    #mask_pred_orig = np.ones((1,512,512,10))
                    mask_pred_orig = mask_pred_orig[0,:,:,:]
                    mask_pred_orig = np.transpose(mask_pred_orig, (2, 0, 1))

                    #print(mask_pred_orig.shape)

                    #mask_pred_orig = mask_pred.squeeze().cpu().detach().numpy()
                    #img_map[:,y+border:y+h-border, x+border:x+w-border] += mask_pred_orig[:,0+border:h-border, 0+border:w-border]
                    img_map[:,y:y+h, x:x+w] += mask_pred_orig * distance_map2

                    #ones_matrix = ones([512,512])
                    #img_map_count[y+border:y+h-border, x+border:x+w-border] += ones_matrix[0+border:h-border, 0+border:w-border]
                    img_map_count[y:y+h, x:x+w] += distance_map

    else:
        for i in tqdm(range(0, x_range)):
            tile_list = []
            for j in range(0, y_range):
                #print("i {} j {}".format(i, j))
                x = i*stepsize
                y = j*stepsize
                h = 512
                w = 512
                img_cropped_orig = img[y:y+h, x:x+w]
                #print("img_cropped_orig.shape {}".format(img_cropped_orig.shape))

                if img_cropped_orig.shape[0] == h and img_cropped_orig.shape[1] == w:
                    #img_cropped_orig = np.expand_dims(img_cropped_orig, axis=0)
                    tile_list.append(img_cropped_orig)
                    #print(img_cropped_orig.shape)
                    
            if len(tile_list) > 0:

                tile_list = np.asarray(tile_list)
                #print("tile_list.shape")
                #print(tile_list.shape)
                #tile_list = tile_list.transpose(1, 2, 0)
                y_tiles = tile_list.shape[0]

                mask_pred_orig = model(tile_list)
                #print("mask_pred_orig.shape")
                #print(mask_pred_orig.shape)

                for j in range(0, y_tiles):

                    x = i*stepsize
                    y = j*stepsize
                    h = 512
                    w = 512

                    #mask_pred_orig = np.ones((1,512,512,10))
                    #print("j")
                    #print(j)
                    #print(mask_pred_orig.shape)
                    mask_pred_orig2 = mask_pred_orig[j,:,:,:]
                    mask_pred_orig2 = np.transpose(mask_pred_orig2, (2, 0, 1))

                    #print(mask_pred_orig.shape)

                    #mask_pred_orig = mask_pred.squeeze().cpu().detach().numpy()
                    #img_map[:,y+border:y+h-border, x+border:x+w-border] += mask_pred_orig[:,0+border:h-border, 0+border:w-border]
                    img_map[:,y:y+h, x:x+w] += mask_pred_orig2 * distance_map2

                    #ones_matrix = ones([512,512])
                    #img_map_count[y+border:y+h-border, x+border:x+w-border] += ones_matrix[0+border:h-border, 0+border:w-border]
                    img_map_count[y:y+h, x:x+w] += distance_map

    print('done predicting')

    img = img[padding:height-padding, padding:width-padding]
    
    print('divide by distance map')
    img_map_count = img_map_count[padding:height-padding, padding:width-padding]
    img_map = img_map[:,padding:height-padding, padding:width-padding] / img_map_count[:,:]

    print('smooth')
    img_map[0,:,:] = ndimage.gaussian_filter(img_map[0,:,:], sigma=5)
    img_map[1,:,:] = ndimage.gaussian_filter(img_map[1,:,:], sigma=5)
    img_map[2,:,:] = ndimage.gaussian_filter(img_map[2,:,:], sigma=5)
    img_map[3,:,:] = ndimage.gaussian_filter(img_map[3,:,:], sigma=5)
    #img_map[4,:,:] = ndimage.gaussian_filter(img_map[4,:,:], sigma=0)
    img_map[5,:,:] = ndimage.gaussian_filter(img_map[5,:,:], sigma=5)
    img_map[6,:,:] = ndimage.gaussian_filter(img_map[6,:,:], sigma=5)
    img_map[7,:,:] = ndimage.gaussian_filter(img_map[7,:,:], sigma=5)
    #img_map[8,:,:] = ndimage.gaussian_filter(img_map[8,:,:], sigma=1)
    img_map[9,:,:] = ndimage.gaussian_filter(img_map[9,:,:], sigma=5)

    mask_pred = np.argmax(img_map, axis=0)

    mask_pred_tumor = (mask_pred == 7).astype(int)
    mask_pred_benign = (mask_pred == 2).astype(int)
    mask_pred_adipose = (mask_pred == 0).astype(int)
    mask_pred_background = (mask_pred == 1).astype(int)
    mask_pred_lymphocytes = (mask_pred == 3).astype(int)
    mask_pred_macrophages = (mask_pred == 4).astype(int)
    mask_pred_muscle = (mask_pred == 5).astype(int)
    mask_pred_nerve = (mask_pred == 6).astype(int)
    mask_pred_vessel = (mask_pred == 8).astype(int)
    mask_pred_stroma = (mask_pred == 9).astype(int)


    if False:
        def remove_small_regions(m_mask, m_img, threshold): 
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(m_mask.astype(np.uint8), connectivity=4)
            sizes = stats[1:, -1];
            for i in range(0, nb_components - 1):
                if sizes[i] < threshold:
                    m_img[output == i + 1] = 1


        print('remove small regions in background')
        mask_pred_background_inverse = (1-mask_pred_background)
        remove_small_regions(mask_pred_background_inverse, mask_pred_background, 10000)

        mask_pred_tumor = mask_pred_tumor > mask_pred_background
        mask_pred_benign = mask_pred_benign > mask_pred_background
        mask_pred_adipose = mask_pred_adipose > mask_pred_background
        mask_pred_lymphocytes = mask_pred_lymphocytes > mask_pred_background
        mask_pred_macrophages = mask_pred_macrophages > mask_pred_background
        mask_pred_muscle = mask_pred_muscle > mask_pred_background
        mask_pred_nerve = mask_pred_nerve > mask_pred_background
        mask_pred_vessel = mask_pred_vessel > mask_pred_background
        mask_pred_stroma = mask_pred_stroma > mask_pred_background


        print('remove small regions in stroma')
        mask_pred_stroma_inverse = (1-mask_pred_stroma)
        remove_small_regions(mask_pred_stroma_inverse, mask_pred_stroma, 1000)

        mask_pred_tumor = mask_pred_tumor > mask_pred_stroma
        mask_pred_benign = mask_pred_benign > mask_pred_stroma
        mask_pred_adipose = mask_pred_adipose > mask_pred_stroma
        mask_pred_background = mask_pred_background > mask_pred_stroma
        mask_pred_lymphocytes = mask_pred_lymphocytes > mask_pred_stroma
        mask_pred_muscle = mask_pred_muscle > mask_pred_stroma
        mask_pred_nerve = mask_pred_nerve > mask_pred_stroma

        print('remove small regions in adipose')
        mask_pred_adipose_inverse = (1-mask_pred_adipose)
        remove_small_regions(mask_pred_adipose_inverse, mask_pred_adipose, 10000)

        mask_pred_tumor = mask_pred_tumor > mask_pred_adipose
        mask_pred_benign = mask_pred_benign > mask_pred_adipose
        mask_pred_background = mask_pred_background > mask_pred_adipose
        mask_pred_lymphocytes = mask_pred_lymphocytes > mask_pred_adipose
        mask_pred_muscle = mask_pred_muscle > mask_pred_adipose
        mask_pred_nerve = mask_pred_nerve > mask_pred_adipose
        mask_pred_stroma = mask_pred_stroma > mask_pred_adipose


        print('recover macrophages')
        mask_pred_tumor = mask_pred_tumor > mask_pred_macrophages
        mask_pred_benign = mask_pred_benign > mask_pred_macrophages
        mask_pred_adipose = mask_pred_adipose > mask_pred_macrophages
        mask_pred_background = mask_pred_background > mask_pred_macrophages
        mask_pred_lymphocytes = mask_pred_lymphocytes > mask_pred_macrophages
        mask_pred_muscle = mask_pred_muscle > mask_pred_macrophages
        mask_pred_nerve = mask_pred_nerve > mask_pred_macrophages
        mask_pred_stroma = mask_pred_stroma > mask_pred_macrophages


        print('fill vessel holes with borders')
        mask_pred_vessel = mask_pred_vessel.astype(np.uint8)
        h, w = mask_pred_vessel.shape[:2]

        mask_pred_vessel = cv2.copyMakeBorder(mask_pred_vessel,100,100,0,0,cv2.BORDER_CONSTANT,value=1) # top bottom left right
        mask_pred_vessel = ndimage.binary_fill_holes(mask_pred_vessel)
        mask_pred_vessel = mask_pred_vessel[100:100+h, 0:w]
        mask_pred_vessel = np.array(mask_pred_vessel, dtype=np.uint8)

        mask_pred_vessel = cv2.copyMakeBorder(mask_pred_vessel,0,0,100,100,cv2.BORDER_CONSTANT,value=1) # top bottom left right
        mask_pred_vessel = ndimage.binary_fill_holes(mask_pred_vessel)
        mask_pred_vessel = mask_pred_vessel[0:h, 100:100+w]
        mask_pred_vessel = np.array(mask_pred_vessel, dtype=np.uint8)

        mask_pred_tumor = mask_pred_tumor > mask_pred_vessel
        mask_pred_benign = mask_pred_benign > mask_pred_vessel
        mask_pred_adipose = mask_pred_adipose > mask_pred_vessel
        mask_pred_background = mask_pred_background > mask_pred_vessel
        mask_pred_lymphocytes = mask_pred_lymphocytes > mask_pred_vessel
        mask_pred_macrophages = mask_pred_macrophages > mask_pred_vessel
        mask_pred_muscle = mask_pred_muscle > mask_pred_vessel
        mask_pred_nerve = mask_pred_nerve > mask_pred_vessel
        mask_pred_stroma = mask_pred_stroma > mask_pred_vessel



    if False:
        print("distinguishing tumor/benign")
        mask_pred_tumor_benign = np.maximum(mask_pred_tumor, mask_pred_benign)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_pred_tumor_benign.astype(np.uint8), connectivity=8)

        lookup_table = np.zeros((nlabels, 2)).astype(float)
        height, width = mask_pred_tumor_benign.shape[:2]

        for x in range(0, width):
            for y in range(0, height):
                i = labels[y,x]
                lookup_table[i][0] += img_map[7,:,:][y,x]
                lookup_table[i][1] += img_map[2,:,:][y,x]


        for x in range(0, width):
            for y in range(0, height):
                i = labels[y,x]
                if i > 0:
                    if (lookup_table[i][0] / lookup_table[i][1]) >= 0.5:
                        mask_pred_tumor[y,x] = 1
                        mask_pred_benign[y,x] = 0
                    else:
                        mask_pred_tumor[y,x] = 0
                        mask_pred_benign[y,x] = 1
                else:
                    mask_pred_tumor[y,x] = 0
                    mask_pred_benign[y,x] = 0

    return [mask_pred_tumor,
                mask_pred_benign,
                mask_pred_adipose,
                mask_pred_background,
                mask_pred_lymphocytes,
                mask_pred_macrophages,
                mask_pred_muscle,
                mask_pred_nerve,
                mask_pred_vessel,
                mask_pred_stroma]





import slideio
from pathlib import Path
import matplotlib.pyplot as plt
#https://towardsdatascience.com/slideio-a-new-python-library-for-reading-medical-images-11858a522059
from scipy import ndimage
from skimage.segmentation import watershed, random_walker
from skimage.feature import peak_local_max
import tensorflow as tf
from tensorflow.compat.v1.nn import softmax_cross_entropy_with_logits_v2
import matplotlib.pyplot as plt
import skimage.io
from numpy import zeros
import math
from scipy.ndimage.morphology import distance_transform_edt
import cv2
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

for file in glob.glob("./CD31/*.svs"):

    print(file)
    
    name = os.path.basename(file[:-4])
    print(name)
    
    #Slide = openslide.OpenSlide(file)
    
    slide = slideio.open_slide(file,'SVS')
    num_scenes = slide.num_scenes
    scene = slide.get_scene(0)
    print(num_scenes, scene.name, scene.rect, scene.num_channels)
    raw_string = slide.raw_metadata
    raw_string.split("|")

    img_size_x = 512
    img_size_y = 512
    img_size = (img_size_x, img_size_y)
    num_classes = 10
    model = get_standard_model(img_size, num_classes)
    model.load_weights('chk_ALL/model050.h5')

     
    directory = './FullInference/'
    
    image_section_size = 512*10
    width = scene.rect[2]
    height = scene.rect[3]
    
    
    print(width)
    print(height)
    
    full_map_tumor = np.zeros((height,width))
    full_map_tumor = full_map_tumor.astype(bool)
    full_map_benign = np.zeros((height,width))
    full_map_benign = full_map_benign.astype(bool)
    full_map_adipose = np.zeros((height,width))
    full_map_adipose = full_map_adipose.astype(bool)
    full_map_background = np.zeros((height,width))
    full_map_background = full_map_background.astype(bool)
    full_map_lymphocytes = np.zeros((height,width))
    full_map_lymphocytes = full_map_lymphocytes.astype(bool)
    full_map_macrophages = np.zeros((height,width))
    full_map_macrophages = full_map_macrophages.astype(bool)
    full_map_muscle = np.zeros((height,width))
    full_map_muscle = full_map_muscle.astype(bool)
    full_map_nerve = np.zeros((height,width))
    full_map_nerve = full_map_nerve.astype(bool)
    full_map_stroma = np.zeros((height,width))
    full_map_stroma = full_map_stroma.astype(bool)
    full_map_vessel = np.zeros((height,width))
    full_map_vessel = full_map_vessel.astype(bool)
    

    x_range = math.ceil(width / image_section_size)
    y_range = math.ceil(height / image_section_size)
    
    print(x_range)
    print(y_range)
    
    for j in tqdm(range(0, y_range)):
    #for j in range(0, y_range):
        for i in range(0, x_range):
            x = i * image_section_size
            y = j * image_section_size
            w = image_section_size + 512
            h = image_section_size + 512
            
            if x + w > width:
                x = width - w
                
            if y + h > height:
                y = height - h
                            
            print('{} {} {} {}'.format(x,y,w,h))    
            filename_original = directory + name + '.svs_(1.0,'+str(x)+','+str(y)+','+str(w)+','+str(h)+').jpg'
            filename_segmentations = directory + name + '.svs_Tumor_(1.0,'+str(x)+','+str(y)+','+str(w)+','+str(h)+').png'
            #img = Slide.read_region((x, y), 0, (w, h))
            
            print('reading block')    
            img = scene.read_block((x, y, w, h), size=(0,0))
            #img = img.convert("RGB")
            
            # img.save(filename_original, "JPEG", quality=95)
            
            print('infer') 
            masks = infer(img, filename_segmentations, model)
            print('finished infer')    
            
            
            mask_pred_tumor = masks[0]
            mask_pred_benign = masks[1]
            mask_pred_adipose = masks[2]
            mask_pred_background = masks[3]
            mask_pred_lymphocytes = masks[4]
            mask_pred_macrophages = masks[5]
            mask_pred_muscle = masks[6]
            mask_pred_nerve = masks[7]
            mask_pred_vessel = masks[8]
            mask_pred_stroma = masks[9]

            mask_pred_tumor = mask_pred_tumor.astype(bool)
            mask_pred_benign = mask_pred_benign.astype(bool)
            mask_pred_adipose = mask_pred_adipose.astype(bool)
            mask_pred_background = mask_pred_background.astype(bool)
            mask_pred_lymphocytes = mask_pred_lymphocytes.astype(bool)
            mask_pred_macrophages = mask_pred_macrophages.astype(bool)
            mask_pred_muscle = mask_pred_muscle.astype(bool)
            mask_pred_nerve = mask_pred_nerve.astype(bool)
            mask_pred_stroma = mask_pred_stroma.astype(bool)
            mask_pred_vessel = mask_pred_vessel.astype(bool)
            
            full_map_tumor[y:y+h, x:x+w] = mask_pred_tumor
            full_map_benign[y:y+h, x:x+w] = mask_pred_benign
            full_map_adipose[y:y+h, x:x+w] = mask_pred_adipose
            full_map_background[y:y+h, x:x+w] = mask_pred_background
            full_map_lymphocytes[y:y+h, x:x+w] = mask_pred_lymphocytes
            full_map_macrophages[y:y+h, x:x+w] = mask_pred_macrophages
            full_map_muscle[y:y+h, x:x+w] = mask_pred_muscle
            full_map_nerve[y:y+h, x:x+w] = mask_pred_nerve
            full_map_stroma[y:y+h, x:x+w] = mask_pred_stroma
            full_map_vessel[y:y+h, x:x+w] = mask_pred_vessel
            

    im = Image.fromarray(full_map_tumor)
    im.save(directory + name + '_tumor.png')
    im = Image.fromarray(full_map_benign)
    im.save(directory + name + '_benign.png')
    im = Image.fromarray(full_map_adipose)
    im.save(directory + name + '_adipose.png')
    im = Image.fromarray(full_map_background)
    im.save(directory + name + '_background.png')
    im = Image.fromarray(full_map_lymphocytes)
    im.save(directory + name + '_lymphocytes.png')
    im = Image.fromarray(full_map_macrophages)
    im.save(directory + name + '_macrophages.png')
    im = Image.fromarray(full_map_muscle)
    im.save(directory + name + '_muscle.png')
    im = Image.fromarray(full_map_nerve)
    im.save(directory + name + '_nerve.png')
    im = Image.fromarray(full_map_stroma)
    im.save(directory + name + '_stroma.png')
    im = Image.fromarray(full_map_vessel)
    im.save(directory + name + '_vessel.png')

        
    
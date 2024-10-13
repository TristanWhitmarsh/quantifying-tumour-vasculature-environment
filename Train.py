#from IPython.display import Image, display
import os
import glob
import numpy as np
from scipy import ndimage
import skimage.io
import random
import numpy as np
import tifffile as tiff
import math
import cv2
import multiprocessing
import imageio
import random
import PIL
from PIL import Image
from PIL import ImageOps

import tensorflow
import tensorflow.compat.v2 as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.nn import softmax_cross_entropy_with_logits
from tensorflow.compat.v1.nn import softmax_cross_entropy_with_logits_v2

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
from keras import optimizers



tf.keras.mixed_precision.set_global_policy("mixed_float16")

#input_dir = "images/"
#input_dir = "data/684913/image/"
#target_dir = "annotations/trimaps/"
#target_dir = "data/684913/mask_vessel/"

img_size_x = 512
img_size_y = 512
img_size = (img_size_x, img_size_y)
num_classes = 10
batch_size = 23

#MISSION
dataset_MISSION = ['684913',
'684914',
'684916',
'685039',
'685040',
'685041',
'685042',
'685281']

#BEHOLD
dataset_BEHOLD = ['688058',
'688059',
'688060',
'688061',
'688062',
'688063',
'688064',
'688065',
'688066',
'688067',
'688068',
'688069',
'688070',
'688071',
'688072',
'688073',
'688074',
'688075',
'688076',
'688077',
'688078',
'688079',
'688080',
'688081',
'688082',
'688083',
'688084',
#'688085',
'688135']

dataset = ['684913',
'684914',
'684916',
'685039',
'685040',
'685041',
'685042',
'685281',
'688058',
'688059',
'688060',
'688061',
'688062',
'688063',
'688064',
'688065',
'688066',
'688067',
'688068',
'688069',
'688070',
'688071',
'688072',
'688073',
'688074',
'688075',
'688076',#mucus
'688077',
'688078',
'688079',
'688080',
'688081',
'688082',
'688083',
'688084',
#'688085',
'688135']

input_img_paths = []
target_mask_paths_adipose = []
target_mask_paths_background = []
target_mask_paths_benign = []
target_mask_paths_lymphocytes = []
target_mask_paths_macrophages = []
target_mask_paths_muscle = []
target_mask_paths_nerve = []
target_mask_paths_tumor = []
target_mask_paths_vessel = []

for item in dataset:
    for fname in glob.glob('data/'+item+'/image/*.jpg'):
        input_img_paths.append(fname)
    for fname in glob.glob('data/'+item+'/mask_adipose/*.png'):
        target_mask_paths_adipose.append(fname)
    for fname in glob.glob('data/'+item+'/mask_background/*.png'):
        target_mask_paths_background.append(fname)
    for fname in glob.glob('data/'+item+'/mask_benign/*.png'):
        target_mask_paths_benign.append(fname)
    for fname in glob.glob('data/'+item+'/mask_lymphocytes/*.png'):
        target_mask_paths_lymphocytes.append(fname)
    for fname in glob.glob('data/'+item+'/mask_macrophages/*.png'):
        target_mask_paths_macrophages.append(fname)
    for fname in glob.glob('data/'+item+'/mask_muscle/*.png'):
        target_mask_paths_muscle.append(fname)
    for fname in glob.glob('data/'+item+'/mask_nerve/*.png'):
        target_mask_paths_nerve.append(fname)
    for fname in glob.glob('data/'+item+'/mask_tumor/*.png'):
        target_mask_paths_tumor.append(fname)
    for fname in glob.glob('data/'+item+'/mask_vessel/*.png'):
        target_mask_paths_vessel.append(fname)

input_img_paths = sorted(input_img_paths)
target_mask_paths_adipose = sorted(target_mask_paths_adipose)
target_mask_paths_background = sorted(target_mask_paths_background)
target_mask_paths_benign = sorted(target_mask_paths_benign)
target_mask_paths_lymphocytes = sorted(target_mask_paths_lymphocytes)
target_mask_paths_macrophages = sorted(target_mask_paths_macrophages)
target_mask_paths_muscle = sorted(target_mask_paths_muscle)
target_mask_paths_nerve = sorted(target_mask_paths_nerve)
target_mask_paths_tumor = sorted(target_mask_paths_tumor)
target_mask_paths_vessel = sorted(target_mask_paths_vessel)

print("Number of samples:", len(input_img_paths))


def crop_center(img,cropx,cropy):
    if img.ndim == 3:
        y,x,z = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx,:]
    else:
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]
    
    
def augment(img,orientation):
    # orientation == 0 does nothing
    if orientation == 1:
        img = np.rot90(img)
    elif orientation == 2:
        img = np.rot90(np.rot90(img))
    elif orientation == 3:
        img = np.rot90(np.rot90(np.rot90(img)))
    elif orientation == 4:
        img = np.fliplr(img)
    elif orientation == 5:
        img = np.fliplr(np.rot90(img))
    elif orientation == 6:
        img = np.fliplr(np.rot90(np.rot90(img)))
    elif orientation == 7:
        img = np.fliplr(np.rot90(np.rot90(np.rot90(img))))
        
    return img


class PathologyData(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    
    def __init__(self, batch_size, img_size, input_img_paths,
                 target_mask_paths_adipose,
                 target_mask_paths_background,
                 target_mask_paths_benign,
                 target_mask_paths_lymphocytes,
                 target_mask_paths_macrophages,
                 target_mask_paths_muscle,
                 target_mask_paths_nerve,
                 target_mask_paths_tumor,
                 target_mask_paths_vessel):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        
        self.target_mask_paths_adipose = target_mask_paths_adipose
        self.target_mask_paths_background = target_mask_paths_background
        self.target_mask_paths_benign = target_mask_paths_benign
        self.target_mask_paths_lymphocytes = target_mask_paths_lymphocytes
        self.target_mask_paths_macrophages = target_mask_paths_macrophages
        self.target_mask_paths_muscle = target_mask_paths_muscle
        self.target_mask_paths_nerve = target_mask_paths_nerve
        self.target_mask_paths_tumor = target_mask_paths_tumor
        self.target_mask_paths_vessel = target_mask_paths_vessel
        #self.on_epoch_end()
        
        c = list(zip(self.input_img_paths,
                 self.target_mask_paths_adipose,
                 self.target_mask_paths_background,
                 self.target_mask_paths_benign,
                 self.target_mask_paths_lymphocytes,
                 self.target_mask_paths_macrophages,
                 self.target_mask_paths_muscle,
                 self.target_mask_paths_nerve,
                 self.target_mask_paths_tumor,
                 self.target_mask_paths_vessel))
        random.shuffle(c)
        (
            self.input_img_paths, 
            self.target_mask_paths_adipose,
            self.target_mask_paths_background,
            self.target_mask_paths_benign,
            self.target_mask_paths_lymphocytes,
            self.target_mask_paths_macrophages,
            self.target_mask_paths_muscle,
            self.target_mask_paths_nerve,
            self.target_mask_paths_tumor,
            self.target_mask_paths_vessel
        ) = zip(*c)

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size
    

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        
        batch_target_img_paths = self.target_mask_paths_vessel[i : i + self.batch_size]
        batch_target_mask_paths_adipose = self.target_mask_paths_adipose[i : i + self.batch_size]
        batch_target_mask_paths_background = self.target_mask_paths_background[i : i + self.batch_size]
        batch_target_mask_paths_benign = self.target_mask_paths_benign[i : i + self.batch_size]
        batch_target_mask_paths_lymphocytes = self.target_mask_paths_lymphocytes[i : i + self.batch_size]
        batch_target_mask_paths_macrophages = self.target_mask_paths_macrophages[i : i + self.batch_size]
        batch_target_mask_paths_muscle = self.target_mask_paths_muscle[i : i + self.batch_size]
        batch_target_mask_paths_nerve = self.target_mask_paths_nerve[i : i + self.batch_size]
        batch_target_mask_paths_tumor = self.target_mask_paths_tumor[i : i + self.batch_size]
        batch_target_mask_paths_vessel = self.target_mask_paths_vessel[i : i + self.batch_size]


        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (num_classes,), dtype="uint8")
        
        for j, path in enumerate(batch_input_img_paths):
            img = skimage.io.imread(path)#.astype(np.float32)
            
            mask_adipose = skimage.io.imread(batch_target_mask_paths_adipose[j])
            mask_background = skimage.io.imread(batch_target_mask_paths_background[j])
            mask_benign = skimage.io.imread(batch_target_mask_paths_benign[j])
            mask_lymphocytes = skimage.io.imread(batch_target_mask_paths_lymphocytes[j])
            mask_macrophages = skimage.io.imread(batch_target_mask_paths_macrophages[j])
            mask_muscle = skimage.io.imread(batch_target_mask_paths_muscle[j])
            mask_nerve = skimage.io.imread(batch_target_mask_paths_nerve[j])
            mask_tumor = skimage.io.imread(batch_target_mask_paths_tumor[j])
            mask_vessel = skimage.io.imread(batch_target_mask_paths_vessel[j])
            
            # Fast random transformation
            orientation = random.randint(0, 7)
            img = augment(img,orientation)
            mask_adipose = augment(mask_adipose,orientation)
            mask_background = augment(mask_background,orientation)
            mask_benign = augment(mask_benign,orientation)
            mask_lymphocytes = augment(mask_lymphocytes,orientation)
            mask_macrophages = augment(mask_macrophages,orientation)
            mask_muscle = augment(mask_muscle,orientation)
            mask_nerve = augment(mask_nerve,orientation)
            mask_tumor = augment(mask_tumor,orientation)
            mask_vessel = augment(mask_vessel,orientation)
            
            # img = ndimage.rotate(img, rotation, reshape=False, mode='reflect')
            # mask_adipose = ndimage.rotate(mask_adipose, rotation, reshape=False, mode='reflect')
            # mask_background = ndimage.rotate(mask_background, rotation, reshape=False, mode='reflect')
            # mask_benign = ndimage.rotate(mask_benign, rotation, reshape=False, mode='reflect')
            # mask_lymphocytes = ndimage.rotate(mask_lymphocytes, rotation, reshape=False, mode='reflect')
            # mask_macrophages = ndimage.rotate(mask_macrophages, rotation, reshape=False, mode='reflect')
            # mask_muscle = ndimage.rotate(mask_muscle, rotation, reshape=False, mode='reflect')
            # mask_nerve = ndimage.rotate(mask_nerve, rotation, reshape=False, mode='reflect')
            # mask_tumor = ndimage.rotate(mask_tumor, rotation, reshape=False, mode='reflect')
            # mask_vessel = ndimage.rotate(mask_vessel, rotation, reshape=False, mode='reflect')
            
            do_rotate = False # random.choice([True, False])
            do_scale = False # random.choice([True, False])
            
            if do_rotate or do_scale:
                padding = 256

                img = np.pad(img, [(padding, padding),(padding, padding),(0, 0)], mode='reflect')
                mask_adipose = np.pad(mask_adipose, [(padding, padding),(padding, padding)], mode='reflect')
                mask_background = np.pad(mask_background, [(padding, padding),(padding, padding)], mode='reflect')
                mask_benign = np.pad(mask_benign, [(padding, padding),(padding, padding)], mode='reflect')
                mask_lymphocytes = np.pad(mask_lymphocytes, [(padding, padding),(padding, padding)], mode='reflect')
                mask_macrophages = np.pad(mask_macrophages, [(padding, padding),(padding, padding)], mode='reflect')
                mask_muscle = np.pad(mask_muscle, [(padding, padding),(padding, padding)], mode='reflect')
                mask_nerve = np.pad(mask_nerve, [(padding, padding),(padding, padding)], mode='reflect')
                mask_tumor = np.pad(mask_tumor, [(padding, padding),(padding, padding)], mode='reflect')
                mask_vessel = np.pad(mask_vessel, [(padding, padding),(padding, padding)], mode='reflect')
            
            # Rotate
            if do_rotate:
                rotation = random.uniform(0, 360)

                img = Image.fromarray(img, "RGB")
                mask_adipose = Image.fromarray(mask_adipose)
                mask_background = Image.fromarray(mask_background)
                mask_benign = Image.fromarray(mask_benign)
                mask_lymphocytes = Image.fromarray(mask_lymphocytes)
                mask_macrophages = Image.fromarray(mask_macrophages)
                mask_muscle = Image.fromarray(mask_muscle)
                mask_nerve = Image.fromarray(mask_nerve)
                mask_tumor = Image.fromarray(mask_tumor)
                mask_vessel = Image.fromarray(mask_vessel)

                img = img.rotate(rotation)
                mask_adipose = mask_adipose.rotate(rotation)
                mask_background = mask_background.rotate(rotation)
                mask_benign = mask_benign.rotate(rotation)
                mask_lymphocytes = mask_lymphocytes.rotate(rotation)
                mask_macrophages = mask_macrophages.rotate(rotation)
                mask_muscle = mask_muscle.rotate(rotation)
                mask_nerve = mask_nerve.rotate(rotation)
                mask_tumor = mask_tumor.rotate(rotation)
                mask_vessel = mask_vessel.rotate(rotation)

                img = np.array(img)
                mask_adipose = np.array(mask_adipose)
                mask_background = np.array(mask_background)
                mask_benign = np.array(mask_benign)
                mask_lymphocytes = np.array(mask_lymphocytes)
                mask_macrophages = np.array(mask_macrophages)
                mask_muscle = np.array(mask_muscle)
                mask_nerve = np.array(mask_nerve)
                mask_tumor = np.array(mask_tumor)
                mask_vessel = np.array(mask_vessel)
                
            
            #Scale
            if do_scale:
                scale = random.uniform(0.8, 1.2)
                
                mask_adipose = mask_adipose.astype(np.uint8)
                mask_background = mask_background.astype(np.uint8)
                mask_benign = mask_benign.astype(np.uint8)
                mask_lymphocytes = mask_lymphocytes.astype(np.uint8)
                mask_macrophages = mask_macrophages.astype(np.uint8)
                mask_muscle = mask_muscle.astype(np.uint8)
                mask_nerve = mask_nerve.astype(np.uint8)
                mask_tumor = mask_tumor.astype(np.uint8)
                mask_vessel = mask_vessel.astype(np.uint8)
            
                dim = (int(1024*scale), int(1024*scale))
                img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
                mask_adipose = cv2.resize(mask_adipose, dim, interpolation = cv2.INTER_NEAREST)
                mask_background = cv2.resize(mask_background, dim, interpolation = cv2.INTER_NEAREST)
                mask_benign = cv2.resize(mask_benign, dim, interpolation = cv2.INTER_NEAREST)
                mask_lymphocytes = cv2.resize(mask_lymphocytes, dim, interpolation = cv2.INTER_NEAREST)
                mask_macrophages = cv2.resize(mask_macrophages, dim, interpolation = cv2.INTER_NEAREST)
                mask_muscle = cv2.resize(mask_muscle, dim, interpolation = cv2.INTER_NEAREST)
                mask_nerve = cv2.resize(mask_nerve, dim, interpolation = cv2.INTER_NEAREST)
                mask_tumor = cv2.resize(mask_tumor, dim, interpolation = cv2.INTER_NEAREST)
                mask_vessel = cv2.resize(mask_vessel, dim, interpolation = cv2.INTER_NEAREST)

                
            if do_rotate or do_scale:
                img = crop_center(img,512,512)
                mask_adipose = crop_center(mask_adipose,512,512)
                mask_background = crop_center(mask_background,512,512)
                mask_benign = crop_center(mask_benign,512,512)
                mask_lymphocytes = crop_center(mask_lymphocytes,512,512)
                mask_macrophages = crop_center(mask_macrophages,512,512)
                mask_muscle = crop_center(mask_muscle,512,512)
                mask_nerve = crop_center(mask_nerve,512,512)
                mask_tumor = crop_center(mask_tumor,512,512)
                mask_vessel = crop_center(mask_vessel,512,512)
            
            mask_adipose = mask_adipose > 126
            mask_background = mask_background > 126
            mask_benign = mask_benign > 126
            mask_lymphocytes = mask_lymphocytes > 126
            mask_macrophages = mask_macrophages > 126
            mask_muscle = mask_muscle > 126
            mask_nerve = mask_nerve > 126
            mask_tumor = mask_tumor > 126
            mask_vessel = mask_vessel > 126
            
            mask_adipose = mask_adipose.astype(np.uint8)
            mask_background = mask_background.astype(np.uint8)
            mask_benign = mask_benign.astype(np.uint8)
            mask_lymphocytes = mask_lymphocytes.astype(np.uint8)
            mask_macrophages = mask_macrophages.astype(np.uint8)
            mask_muscle = mask_muscle.astype(np.uint8)
            mask_nerve = mask_nerve.astype(np.uint8)
            mask_tumor = mask_tumor.astype(np.uint8)
            mask_vessel = mask_vessel.astype(np.uint8)
            
            mask_stroma = np.ones(mask_adipose.shape) - mask_adipose \
            - mask_background \
            - mask_benign \
            - mask_lymphocytes \
            - mask_macrophages \
            - mask_muscle \
            - mask_nerve \
            - mask_tumor \
            - mask_vessel
            
            mask = np.dstack((mask_adipose,
                              mask_background,
                              mask_benign,
                              mask_lymphocytes,
                              mask_macrophages,
                              mask_muscle,
                              mask_nerve,
                              mask_tumor,
                              mask_vessel,
                              mask_stroma))
        
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
            img = tf.image.random_hue(img, 0.1)
            
            x[j] = img
            y[j] = mask
            
            
        return x, y
    
    


def Conv2D(filters, size, activation=LeakyReLU(alpha=0.1), kernel_initializer="he_normal"):
    return _Conv2D(
        filters, 
        size, 
        activation=activation, 
        padding="same", 
        kernel_initializer=kernel_initializer,
    )


def get_standard_model(input_size, num_classes):
    
    inputs = keras.Input(shape=img_size + (3,))
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



# Split our img paths into a training and a validation set
val_samples = 100

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_mask_paths_adipose)
random.Random(1337).shuffle(target_mask_paths_background)
random.Random(1337).shuffle(target_mask_paths_benign)
random.Random(1337).shuffle(target_mask_paths_lymphocytes)
random.Random(1337).shuffle(target_mask_paths_macrophages)
random.Random(1337).shuffle(target_mask_paths_muscle)
random.Random(1337).shuffle(target_mask_paths_nerve)
random.Random(1337).shuffle(target_mask_paths_tumor)
random.Random(1337).shuffle(target_mask_paths_vessel)

train_input_img_paths = input_img_paths[:-val_samples]
train_target_mask_paths_adipose = target_mask_paths_adipose[:-val_samples]
train_target_mask_paths_background = target_mask_paths_background[:-val_samples]
train_target_mask_paths_benign = target_mask_paths_benign[:-val_samples]
train_target_mask_paths_lymphocytes = target_mask_paths_lymphocytes[:-val_samples]
train_target_mask_paths_macrophages = target_mask_paths_macrophages[:-val_samples]
train_target_mask_paths_muscle = target_mask_paths_muscle[:-val_samples]
train_target_mask_paths_nerve = target_mask_paths_nerve[:-val_samples]
train_target_mask_paths_tumor = target_mask_paths_tumor[:-val_samples]
train_target_mask_paths_vessel = target_mask_paths_vessel[:-val_samples]

val_input_img_paths = input_img_paths[-val_samples:]
val_target_mask_paths_adipose = target_mask_paths_adipose[-val_samples:]
val_target_mask_paths_background = target_mask_paths_background[-val_samples:]
val_target_mask_paths_benign = target_mask_paths_benign[-val_samples:]
val_target_mask_paths_lymphocytes = target_mask_paths_lymphocytes[-val_samples:]
val_target_mask_paths_macrophages = target_mask_paths_macrophages[-val_samples:]
val_target_mask_paths_muscle = target_mask_paths_muscle[-val_samples:]
val_target_mask_paths_nerve = target_mask_paths_nerve[-val_samples:]
val_target_mask_paths_tumor = target_mask_paths_tumor[-val_samples:]
val_target_mask_paths_vessel = target_mask_paths_vessel[-val_samples:]

# Instantiate data Sequences for each split
train_gen = PathologyData(batch_size, img_size, train_input_img_paths,
    train_target_mask_paths_adipose,
    train_target_mask_paths_background,
    train_target_mask_paths_benign,
    train_target_mask_paths_lymphocytes,
    train_target_mask_paths_macrophages,
    train_target_mask_paths_muscle,
    train_target_mask_paths_nerve,
    train_target_mask_paths_tumor,
    train_target_mask_paths_vessel)
train_dataset = tensorflow.data.Dataset.from_generator(lambda: train_gen,
                                           output_types=(tensorflow.float32, tensorflow.float32),
                                           output_shapes=([batch_size, img_size_x, img_size_y, 3], [batch_size, img_size_x, img_size_y, num_classes]))

val_gen = PathologyData(batch_size, img_size, val_input_img_paths,
    val_target_mask_paths_adipose,
    val_target_mask_paths_background,
    val_target_mask_paths_benign,
    val_target_mask_paths_lymphocytes,
    val_target_mask_paths_macrophages,
    val_target_mask_paths_muscle,
    val_target_mask_paths_nerve,
    val_target_mask_paths_tumor,
    val_target_mask_paths_vessel)
val_dataset = tensorflow.data.Dataset.from_generator(lambda: val_gen,
                                           output_types=(tensorflow.float32, tensorflow.float32),
                                           output_shapes=([batch_size, img_size_x, img_size_y, 3], [batch_size, img_size_x, img_size_y, num_classes]))



def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 

    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')   
    numerator = 2. * K.sum(y_pred_f * y_true_f, axis=(0,1,2))
    denominator = K.sum(K.square(y_pred_f) + K.square(y_true_f), axis=(0,1,2))
    values = numerator / (denominator + epsilon)
    
    #adipose, background, benign, lymphocytes, macrophages, muscle, nerve, tumor, vessel, stroma
    #tf.print("denominator", output_stream=sys.stdout)
    #tf.print(tf.shape(denominator), output_stream=sys.stdout)
    #weights = [1.09, 1.02, 2.15, 1.94, 2.33, 2.80, 7.16, 1.08, 10, 1]
    #weights = [1., 1., 2., 2., 3., 3., 5., 1., 10., 1.]
    weights = [1., 1., 2., 2., 5., 2., 5., 2., 10., 1.]
    
    values = values * weights
    
    return 1 - (K.sum(values)/K.sum(weights))



# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_standard_model(img_size, num_classes)


lr_schedule = tensorflow.keras.experimental.CosineDecayRestarts(0.001, 200, t_mul=1.0, m_mul=1.0, alpha=0.0, name=None)

callbacks = [
    keras.callbacks.ModelCheckpoint('chk/model{epoch:03d}.h5', save_best_only=False, save_weights_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 50

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

opt = Adam(lr=0.0001, decay=0.001)
lr_metric = get_lr_metric(opt)

model.compile(optimizer=opt, metrics=['accuracy', lr_metric], loss=soft_dice_loss)
#model = keras.models.load_model("/checkpoints/model02.h5", compile=True, options=None, custom_objects={'softmax_cross_entropy_with_logits_v2': softmax_cross_entropy_with_logits_v2, 'soft_dice_loss': soft_dice_loss, 'binary_crossentropy_Unet': binary_crossentropy_Unet, 'LeakyReLU': LeakyReLU})
model.load_weights('chk/model050.h5')
#model.compile(optimizer=opt, loss=soft_dice_loss)
#model.compile(optimizer=opt, loss=softmax_cross_entropy_with_logits)
#model.compile(optimizer='SGD', loss=binary_crossentropy_Unet)
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, initial_epoch=0) 

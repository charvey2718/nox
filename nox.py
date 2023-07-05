import sys
import pathlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable debugging logs
import numpy as np
import cv2
import skimage
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import pickle
import gc
import csv
import time
import tensorflow as tf
import pandas as pd

# enable dynamic memory allocation
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# global variables
epochs = 1500
n_channels = 3 # 3 for RGB or 1 for grayscale
patch_size = 512
stride = 128
border = int((patch_size - stride)/2)
generator_fname = os.getcwd() + os.sep + ('generator_gray.h5', 'generator_color.h5')[n_channels == 3]
generator_ckpt = os.getcwd() + os.sep + ('generator_gray_ckpt', 'generator_color_ckpt')[n_channels == 3]
discriminator_fname = os.getcwd() + os.sep + ('discriminator_gray.h5', 'discriminator_color.h5')[n_channels == 3]
discriminator_ckpt = os.getcwd() + os.sep + ('discriminator_gray_ckpt', 'discriminator_color_ckpt')[n_channels == 3]

# training parameters
BATCH_SIZE = 6 # number of training samples to work through before the model’s parameters are updated
ema = 0.9995 # exponential moving average: keep ema % of the existing state and (1 - ema) % of the new state
ema_val = ema
lr = 1e-4 # learning rate
patience = 2 # number of epochs with no improvement after which learning rate will be reduced
cooldown = 4 # number of epochs to wait before resuming normal operation after lr has been reduced
validation = True
save_freq = 1

# START: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
def generator():
    tf.keras.backend.clear_session() # release global state of old models and layers
    layers = []
    filters = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64] # filter layers 0 - 14
    input = tf.keras.layers.Input(shape = (patch_size, patch_size, n_channels), name = "gen_input_image")
    for i in range(1 + len(filters)):
        if i == 0: # layer 0 convolution
            convolved = tf.keras.layers.Conv2D(filters[0], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(input)
            layers.append(convolved)
        elif 1 <= i <= 7: # convolution layers
            rectified = tf.keras.layers.LeakyReLU(alpha = 0.2)(layers[-1])
            convolved = tf.keras.layers.Conv2D(filters[i], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
            normalized = tf.keras.layers.LayerNormalization()(convolved)
            layers.append(normalized)
        elif 8 <= i <= 14: # deconvolution layers
            if i == 8:
                rectified = tf.keras.layers.ReLU()(layers[-1])
            else:
                concatenated = tf.concat([layers[-1], layers[15 - i]], axis = 3)
                rectified = tf.keras.layers.ReLU()(concatenated)
            deconvolved = tf.keras.layers.Conv2DTranspose(filters[i], kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
            normalized = tf.keras.layers.LayerNormalization()(deconvolved)
            layers.append(normalized)
        else: # layer 15
            concatenated = tf.concat([layers[-1], layers[0]], axis = 3)
            rectified = tf.keras.layers.ReLU()(concatenated)
            deconvolved = tf.keras.layers.Conv2DTranspose(n_channels, kernel_size = 4, strides = (2, 2), padding = "same", kernel_initializer = tf.initializers.GlorotUniform())(rectified)
            rectified = tf.keras.layers.ReLU()(deconvolved)
            output = tf.math.subtract(input, rectified)
    return tf.keras.Model(inputs = input, outputs = output, name = "generator")
# END: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License

# START: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
def discriminator():
    layers = []
    filters = [32, 64, 64, 128, 128, 256, 256, 256, 8]
    input = tf.keras.layers.Input(shape = (patch_size, patch_size, n_channels), name = "dis_input_image")
    for i in range(1 + len(filters)):
        if i % 2 == 1 or i == 8:
            padding = "valid"
            strides = (2, 2)
        else:
            padding = "same"
            strides = (1, 1)
        if i == 0: # layer 0 convolution
            convolved = tf.keras.layers.Conv2D(filters[i], kernel_size = 3, strides = strides, padding = padding)(input)
            rectified = tf.keras.layers.LeakyReLU(alpha = 0.2)(convolved)
            layers.append(rectified)
        elif 1 <= i <= 8: # convolution layers
            convolved = tf.keras.layers.Conv2D(filters[i], kernel_size = 3, strides = strides, padding = padding)(layers[-1])
            normalized = tf.keras.layers.LayerNormalization()(convolved)
            rectified = tf.keras.layers.LeakyReLU(alpha = 0.2)(normalized)
            layers.append(rectified)
        else: # layer 9
            dense = tf.keras.layers.Dense(1)(layers[-1])
            sigmoid = tf.nn.sigmoid(dense)
            layers.append(sigmoid)
    output = [layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[-1]]
    return tf.keras.Model(inputs = input, outputs = output, name = "discriminator")
# END: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
    
def get_images_paths(root_dir):
    starless_lst = []
    original_lst = []
    
    root = pathlib.Path(root_dir)
    x_paths = sorted(list(root.rglob("x*.png*")))
    y_paths = sorted(list(root.rglob("y*.png*")))
    
    x_paths_lst = [str(path) for path in x_paths]
    for p in x_paths_lst:
        original_lst.append(p)
        
    y_paths_lst = [str(path) for path in y_paths]
    for p in y_paths_lst:
        starless_lst.append(p)

    original_array = np.asarray(original_lst)
    starless_array = np.asarray(starless_lst)
    return original_array, starless_array
    
def up_down_flip(image, label):
    if tf.random.uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32) == 1:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    return image, label

def left_right_flip(image, label):
    if tf.random.uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32) == 1:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    return image, label

def rotate_90(image, label):
    rand_value = tf.random.uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32)
    image = tf.image.rot90(image, rand_value)
    label = tf.image.rot90(label, rand_value)
    return image, label

# START: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
def adjust_color(image, label):
    if n_channels == 3: # RGB model only
        
        # tweak colors
        if tf.random.uniform(shape = []) < 0.70:
            ch = tf.random.uniform(shape = [], minval = 0, maxval = 3, dtype = tf.int32)
            m = tf.reduce_min((image, label))
            offset = tf.random.uniform(shape = [])*0.25 - tf.random.uniform(shape = [])*m
            image_r, image_g, image_b = tf.split(image, axis = 2, num_or_size_splits = 3)
            label_r, label_g, label_b = tf.split(label, axis = 2, num_or_size_splits = 3)
            if ch == 0: # messy but I can't get tensor_scatter_nd_update to work
                image_r = image_r + offset*(1. - image_r)
                label_r = label_r + offset*(1. - label_r)
            elif ch == 1:
                image_g = image_g + offset*(1. - image_g)
                label_g = label_g + offset*(1. - label_g)
            else:
                image_b = image_b + offset*(1. - image_b)
                label_b = label_b + offset*(1. - label_b)
            
            image = tf.concat([image_r, image_g, image_b], axis = 2)
            label = tf.concat([label_r, label_g, label_b], axis = 2)
        
        # flip channels
        if tf.random.uniform(shape = []) < 0.70:
            image_ch = tf.split(image, axis = 2, num_or_size_splits = 3)
            label_ch = tf.split(label, axis = 2, num_or_size_splits = 3)
            indices = tf.range(start = 0, limit = tf.shape(image_ch)[0], dtype = tf.int32)
            indices = tf.random.shuffle(indices)
            image = tf.gather(image, indices, axis = 2)
            label = tf.gather(label, indices, axis = 2)
        
        image = tf.clip_by_value(image, 0. , 1.)
        label = tf.clip_by_value(label, 0. , 1.)
    
    return image, label
# END: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License

# START: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
def adjust_brightness(image, label):
    if n_channels == 1: # grayscale model only
        if tf.random.uniform(shape = []) < 0.7:
            m = tf.reduce_min((image, label))
            offset = tf.random.uniform(shape = [])*0.25 - tf.random.uniform(shape = [])*m
            image = image + offset*(1. - image)
            label = label + offset*(1. - label)
            image = tf.clip_by_value(image, 0. , 1.)
            label = tf.clip_by_value(label, 0. , 1.)
    return image, label
# END: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License

def choose_channel(image, label):
    image_r, image_g, image_b = tf.split(image, axis = 2, num_or_size_splits = 3)
    label_r, label_g, label_b = tf.split(label, axis = 2, num_or_size_splits = 3)
    ch = tf.random.uniform(shape = [], minval = 0, maxval = 3, dtype = tf.int32)
    if ch == 0: return image_r, label_r
    elif ch == 1: return image_g, label_g
    else: return image_b, label_b

def to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    label = tf.image.rgb_to_grayscale(label)
    return image, label

def process_path(path_original, path_starless):
    img_original = tf.io.read_file(path_original)
    img_original = tf.io.decode_image(img_original, dtype = tf.dtypes.float32)
    img_starless = tf.io.read_file(path_starless)
    img_starless = tf.io.decode_image(img_starless, dtype = tf.dtypes.float32)
    return img_original, img_starless

def random_crop(image, label):
    combined = tf.concat([image, label], axis=2)
    combined_crop = tf.image.random_crop(combined, [patch_size, patch_size, n_channels*2])
    return (combined_crop[:, :, :n_channels], combined_crop[:, :, n_channels:])

def data_generator(X, y, batch_size, augmentations = None):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size = 100, reshuffle_each_iteration = True)
    dataset = dataset.map(process_path, num_parallel_calls = tf.data.AUTOTUNE)
    # dataset = dataset.map(random_crop, num_parallel_calls = tf.data.AUTOTUNE) # uncomment if dataset tiles are oversized
    
    if augmentations:
        for f in augmentations:
            dataset = dataset.map(f, num_parallel_calls = tf.data.AUTOTUNE)
            # dataset = dataset.map(lambda x, y: tf.py_function(f, [x, y], [tf.float32, tf.float32]), num_parallel_calls = 1) # execute eagerly (for debugging only)
        if n_channels == 1: dataset = dataset.map(choose_channel, num_parallel_calls = tf.data.AUTOTUNE)
    elif n_channels == 1: dataset = dataset.map(to_grayscale, num_parallel_calls = tf.data.AUTOTUNE)
    
    dataset = dataset.repeat() # re-initialize the dataset as soon as all the entries have been read
    dataset = dataset.batch(batch_size = batch_size, drop_remainder = True) # make batches out of dataset
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # allows later elements to be prepared while the current element is being processed, improving latency and throughput, at the cost of using additional memory to store prefetched elements
    return dataset
    
def PSNR(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val = 1.)

def inference_single_tile(model, original_image):
    input_image = np.expand_dims(original_image, axis = 0)
    predicted_image = (model.predict(input_image*2. - 1.) + 1.)/2.
    return predicted_image[0]
    
def inference_batch_tiles(model, original_images):
    predicted_images = (model.predict(original_images*2. - 1., batch_size = BATCH_SIZE) + 1.)/2.
    return predicted_images
    
def infer_image(model, original_image, border = 0):
    predicted_image = np.zeros(original_image.shape)
    sizeX = original_image.shape[1]
    sizeY = original_image.shape[0]
    
    # add border around original image
    original_image = cv2.copyMakeBorder(original_image, border, border, border, border, cv2.BORDER_REFLECT)
    sizeX = original_image.shape[1]
    sizeY = original_image.shape[0]
    
    # split original image into tiles
    fromRow = 0 # row pixel index
    fromCol = 0 # col pixel index
    nRows = 0
    nCols = 0
    original_lst = []
    finalRow = False
    while True: # loop rows
        nRows += 1
        toRow = fromRow + patch_size
        if toRow >= sizeY:
            fromRow = sizeY - patch_size
            toRow = sizeY
            finalRow = True
        
        finalCol = False
        while True: # loop cols
            if finalRow:
                nCols += 1 # count columns on final row only
            
            toCol = fromCol + patch_size
            if toCol >= sizeX:
                fromCol = sizeX - patch_size
                toCol = sizeX
                finalCol = True
            
            original_roi = original_image[fromRow:toRow, fromCol:toCol]
            original_lst.append(original_roi)
            
            if finalCol:
                fromCol = 0
                break
            else:
                fromCol += (patch_size - 2*border)
                
        if finalRow:
            break
        else:
            fromRow += (patch_size - 2*border)
    
    # infer batch of tiles
    original_rois = np.array(original_lst)
    predicted_rois = inference_batch_tiles(model, original_rois)
    
    # build up predicted image from predicted tiles
    fromRow = 0 # row pixel index
    fromCol = 0 # col pixel index
    count = 0
    for i in range(0, nRows):
        toRow = fromRow + patch_size - 2*border
        if i == nRows - 1:
            fromRow = (sizeY - 2*border) - (patch_size - 2*border)
            toRow = (sizeY - 2*border)
            finalRow = True
        
        for j in range(0, nCols):
            toCol = fromCol + patch_size - 2*border
            
            if j == nCols - 1:
                fromCol = (sizeX - 2*border) - (patch_size - 2*border)
                toCol = (sizeX - 2*border)
                finalCol = True
            
            predicted_image[fromRow:toRow, fromCol:toCol] = predicted_rois[count][border:patch_size - border, border:patch_size - border]
            count = count + 1
            
            if j == nCols - 1:
                fromCol = 0
                break
            else:
                fromCol += (patch_size - 2*border)
        
        fromRow += (patch_size - 2*border)
    
    np.clip(predicted_image, 0., 1., predicted_image) # clip 0 to 1
    
    return predicted_image
    
def infer(file = ''):
    tf.keras.backend.clear_session()
    print('building generator...')
    model = generator()
    # model.summary()
    
    if os.path.exists(generator_fname):
        print('loading weights...')
        model.load_weights(generator_fname)
    else: return
    
    # process image from disk
    if len(file) > 0 and os.path.exists(file):
        original_image = cv2.imread(file)
        original_image = prepare_image(original_image)
        save_image('starry.png', original_image)
        start_time = time.time()
        starless_image = infer_image(model, original_image, border = border)
        print("--- %s seconds ---" % (time.time() - start_time))
        save_image('starless.png', starless_image)
        
    else: # process image from dataset
        original_array_paths, starless_array_paths = get_images_paths(os.getcwd() + os.sep + "nox data") # training and test image paths
        i = np.random.randint(0, original_array_paths.shape[0]) # get a random noisy image index
        original_image = cv2.imread(original_array_paths[i])
        original_image = prepare_image(original_image)
        save_image('input.png', original_image)
        gt_image = cv2.imread(starless_array_paths[i])
        gt_image = prepare_image(gt_image)
        save_image('gt.png', gt_image)
        starless_image = infer_image(model, original_image, border = border)
        save_image('output.png', starless_image)
        psnr1 = PSNR(original_image, gt_image).numpy()
        psnr2 = PSNR(starless_image, gt_image).numpy()
        print('PSNR(input, gt) = %f'%psnr1)
        print('PSNR(output, gt) = %f'%psnr2)

def train():
    tf.keras.backend.clear_session()
    
    # training and test image paths
    original_array_paths, starless_array_paths = get_images_paths(os.getcwd() + os.sep + "nox data")
    
    if validation:
        original_train_paths, original_test_paths, starless_train_paths, starless_test_paths = train_test_split(original_array_paths, starless_array_paths, test_size = 0.2, random_state = 42)
    else:
        original_train_paths = original_array_paths
        starless_train_paths = starless_array_paths
    
    # nets
    print('building generator...')
    G = generator()
    # G.summary()
    print('building discriminator...')
    D = discriminator()
    # D.summary()
    
    # optimizers
    global lr
    on_lr = 0 # number of epochs at present lr
    gen_optimizer = tf.optimizers.Adam(learning_rate = lr)
    dis_optimizer = tf.optimizers.Adam(learning_rate = lr/4)
    
    hist_train = {}
    hist_val = {}
    
    # load weights
    e0 = -1
    if os.path.exists(generator_fname) \
        and os.path.exists(discriminator_fname) \
        and os.path.exists(os.getcwd() + os.sep + 'history.pkl'):
        print('loading weights...')
        G.load_weights(generator_fname)
        D.load_weights(discriminator_fname)
        G_checkpoint = tf.train.Checkpoint(model = G, optim = gen_optimizer)
        G_checkpoint.restore(generator_ckpt + '-1')
        D_checkpoint = tf.train.Checkpoint(model = D, optim = dis_optimizer)
        D_checkpoint.restore(discriminator_ckpt + '-1')
        with open('history.pkl', 'rb') as f:
            print('loading training history...')
            if validation:
                e0, on_lr, lr, hist_train, hist_val, original_train_paths, original_test_paths, starless_train_paths, starless_test_paths = pickle.load(f)
                print("\repoch %d: loss %f PSNR %f    val: loss %f PSNR %f                    \n"%(e0, hist_train['total'][-1], hist_train['psnr'][-1], hist_val['total'][-1], hist_val['psnr'][-1]), end = '')
            else: 
                e0, on_lr, lr, hist_train, original_train_paths, starless_train_paths = pickle.load(f)
                print("\repoch %d: loss %f PSNR %f                    \n"%(e0, hist_train['total'][-1], hist_train['psnr'][-1]), end = '')
    
    if validation:
        print('number of original images for training = %i'%original_train_paths.shape)
        print('number of original images for testing = %i'%original_test_paths.shape)
        print('number of starless images for training = %i'%starless_train_paths.shape)
        print('number of starless images for testing = %i'%starless_test_paths.shape)
    else:
        print('number of original images for training = %i'%original_train_paths.shape)
        print('number of starless images for training = %i'%starless_train_paths.shape)
        
    # number of epochs and steps
    print('number of training epochs = %i'%epochs)
    steps_per_epoch_train = int(len(original_train_paths)/BATCH_SIZE)
    print('steps per training epoch = %i'%steps_per_epoch_train)
    steps_per_epoch_validation = 1
    if validation:
        steps_per_epoch_validation = int(len(original_test_paths)/BATCH_SIZE)
        ema_val = 1. - (1. - ema)*steps_per_epoch_train/steps_per_epoch_validation
        print('steps per validation epoch = %i'%steps_per_epoch_validation)

    # data generators
    augmentation_lst = [left_right_flip, up_down_flip, rotate_90, adjust_color, adjust_brightness]
    image_generator_train = data_generator(X = original_train_paths, y = starless_train_paths, batch_size = BATCH_SIZE, augmentations = augmentation_lst)
    image_generator_test = []
    if validation:
        image_generator_test = data_generator(X = original_test_paths, y = starless_test_paths, batch_size = BATCH_SIZE)
    
    @tf.function
    def train_step(e, x, y):
        # START: This portion of code is Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            gen_output = G(x)
            
            p1_real, p2_real, p3_real, p4_real, p5_real, p6_real, p7_real, p8_real, predict_real = D(y)
            p1_fake, p2_fake, p3_fake, p4_fake, p5_fake, p6_fake, p7_fake, p8_fake, predict_fake = D(gen_output)
            
            d = {}
            
            dis_loss = tf.reduce_mean(-(tf.math.log(predict_real + 1E-8)+tf.math.log(1. - predict_fake + 1E-8)))
            d['dis_loss'] = dis_loss
            
            gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake + 1E-8))
            d['gen_loss_GAN'] = gen_loss_GAN
            
            gen_p1 = tf.reduce_mean(tf.abs(p1_fake - p1_real))
            d['gen_p1'] = gen_p1
            
            gen_p2 = tf.reduce_mean(tf.abs(p2_fake - p2_real))
            d['gen_p2'] = gen_p2
            
            gen_p3 = tf.reduce_mean(tf.abs(p3_fake - p3_real))
            d['gen_p3'] = gen_p3
            
            gen_p4 = tf.reduce_mean(tf.abs(p4_fake - p4_real))
            d['gen_p4'] = gen_p4
            
            gen_p5 = tf.reduce_mean(tf.abs(p5_fake - p5_real))
            d['gen_p5'] = gen_p5
            
            gen_p6 = tf.reduce_mean(tf.abs(p6_fake - p6_real))
            d['gen_p6'] = gen_p6
            
            gen_p7 = tf.reduce_mean(tf.abs(p7_fake - p7_real))
            d['gen_p7'] = gen_p7
            
            gen_p8 = tf.reduce_mean(tf.abs(p8_fake - p8_real))
            d['gen_p8'] = gen_p8
            
            gen_L1 = tf.reduce_mean(tf.abs(y - gen_output))
            d['gen_L1'] = gen_L1*100.
            
            d['psnr'] = tf.reduce_mean(PSNR(y, gen_output))
            
            gen_loss = gen_loss_GAN*0.1 + gen_p1 *0.1 + gen_p2*10. + gen_p3*10. + gen_p4*10. + gen_p5*10. + gen_p6*10. + gen_p7*10. + gen_p8*10. + gen_L1*100.
            d['total'] = gen_loss
            
        if e > 0: # first 'epoch' (e = 0) is just to get starting losses for moving average as otherwise the validation loss starts much lower
            gen_grads = gen_tape.gradient(gen_loss, G.trainable_variables)
            gen_optimizer.apply_gradients(zip(gen_grads, G.trainable_variables))
            dis_grads = dis_tape.gradient(dis_loss, D.trainable_variables)
            dis_optimizer.apply_gradients(zip(dis_grads, D.trainable_variables))
        
        # END: This portion of code is Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
        return d
        
    @tf.function
    def test_step(x, y):
        # START: This portion of code is Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
        gen_output = G(x)
        
        p1_real, p2_real, p3_real, p4_real, p5_real, p6_real, p7_real, p8_real, predict_real = D(y)
        p1_fake, p2_fake, p3_fake, p4_fake, p5_fake, p6_fake, p7_fake, p8_fake, predict_fake = D(gen_output)
        
        d = {}
        
        dis_loss = tf.reduce_mean(-(tf.math.log(predict_real + 1E-8) + tf.math.log(1. - predict_fake + 1E-8)))
        d['dis_loss'] = dis_loss
        
        gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake + 1E-8))
        d['gen_loss_GAN'] = gen_loss_GAN
        
        gen_p1 = tf.reduce_mean(tf.abs(p1_fake - p1_real))
        d['gen_p1'] = gen_p1
        
        gen_p2 = tf.reduce_mean(tf.abs(p2_fake - p2_real))
        d['gen_p2'] = gen_p2
        
        gen_p3 = tf.reduce_mean(tf.abs(p3_fake - p3_real))
        d['gen_p3'] = gen_p3
        
        gen_p4 = tf.reduce_mean(tf.abs(p4_fake - p4_real))
        d['gen_p4'] = gen_p4
        
        gen_p5 = tf.reduce_mean(tf.abs(p5_fake - p5_real))
        d['gen_p5'] = gen_p5
        
        gen_p6 = tf.reduce_mean(tf.abs(p6_fake - p6_real))
        d['gen_p6'] = gen_p6
        
        gen_p7 = tf.reduce_mean(tf.abs(p7_fake - p7_real))
        d['gen_p7'] = gen_p7
        
        gen_p8 = tf.reduce_mean(tf.abs(p8_fake - p8_real))
        d['gen_p8'] = gen_p8
        
        gen_L1 = tf.reduce_mean(tf.abs(y - gen_output))
        d['gen_L1'] = gen_L1*100.
        
        d['psnr'] = tf.reduce_mean(PSNR(y, gen_output))
        
        gen_loss = gen_loss_GAN*0.1 + gen_p1*0.1 + gen_p2*10. + gen_p3*10. + gen_p4*10. + gen_p5*10. + gen_p6*10. + gen_p7*10. + gen_p8*10. + gen_L1*100.
        d['total'] = gen_loss
        
        # END: This portion of code is Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
        return d

    # training epochs with validation
    for e in range(e0 + 1, epochs + 1, 1):
        # training batches
        for i in range(steps_per_epoch_train):
            x, y = image_generator_train.take(1).as_numpy_iterator().next()
            x = x*2. - 1. # rescale fom -1 to 1
            y = y*2. - 1.
            
            if i > 0:
                print("\repoch %d: it %d/%d loss %f PSNR %f    "%(e, i + 1, steps_per_epoch_train, hist_train['total'][-1], hist_train['psnr'][-1]), end = '')
            else:
                print("\repoch %d: it %d/%d "%(e, i + 1, steps_per_epoch_train), end = '')
            
            d = train_step(tf.constant(e), x, y)
            for k in d:
                if k in hist_train.keys():
                    hist_train[k].append((d[k]*(1. - ema) + hist_train[k][-1]*ema).numpy())
                else:
                    hist_train[k] = [d[k].numpy()]
            
            if 'loss_us' in hist_train.keys(): # include unsmoothed loss and psnr in logs
                hist_train['loss_us'].append(d['total'].numpy())
                hist_train['psnr_us'].append(d['psnr'].numpy())
            else:
                hist_train['loss_us'] = [d['total'].numpy()]
                hist_train['psnr_us'] = [d['psnr'].numpy()]
            
        if e == 0: # force retracing after initial pass (so that gradients will be applied in future)
            tf.config.run_functions_eagerly(True)
            tf.config.run_functions_eagerly(False)
        
        # validation batches
        if validation:
            for i in range(steps_per_epoch_validation):
                x, y = image_generator_test.take(1).as_numpy_iterator().next()
                x = x*2. - 1. # rescale fom -1 to 1
                y = y*2. - 1.
                
                if i > 0:
                    print("\repoch %d: loss %f PSNR %f    val: it %d/%d loss %f PSNR %f    "%(e, hist_train['total'][-1], hist_train['psnr'][-1], i+1, steps_per_epoch_validation, hist_val['total'][-1], hist_val['psnr'][-1]), end = '')
                else:
                    print("\repoch %d: loss %f PSNR %f    val: it %d/%d    "%(e, hist_train['total'][-1], hist_train['psnr'][-1], i+1, steps_per_epoch_validation), end = '')
                
                d = test_step(x, y)
                for k in d:
                    if k in hist_val.keys():
                        hist_val[k].append((d[k]*(1. - ema_val) + hist_val[k][-1]*ema_val).numpy())
                    else:
                        hist_val[k] = [d[k].numpy()]
                        
                if 'val_loss_us' in hist_val.keys(): # include unsmoothed loss and psnr in logs
                    hist_val['val_loss_us'].append(d['total'].numpy())
                    hist_val['val_psnr_us'].append(d['psnr'].numpy())
                else:
                    hist_val['val_loss_us'] = [d['total'].numpy()]
                    hist_val['val_psnr_us'] = [d['psnr'].numpy()]
            
            print("\repoch %d: loss %f PSNR %f    val: loss %f PSNR %f                    \n"%(e, hist_train['total'][-1], hist_train['psnr'][-1], hist_val['total'][-1], hist_val['psnr'][-1]), end = '')
        else:
            print("")
        
        if e % save_freq == 0:
            G.save_weights(generator_fname)
            D.save_weights(discriminator_fname)
            G_checkpoint = tf.train.Checkpoint(model = G, optim = gen_optimizer)
            G_checkpoint.save(generator_ckpt)
            D_checkpoint = tf.train.Checkpoint(model = D, optim = dis_optimizer)
            D_checkpoint.save(discriminator_ckpt)
            if validation:
                with open('history.pkl', 'wb') as f:
                    pickle.dump([e, on_lr, lr, hist_train, hist_val, original_train_paths, original_test_paths, starless_train_paths, starless_test_paths], f)
            else:
                with open('history.pkl', 'wb') as f:
                    pickle.dump([e, on_lr, lr, hist_train, original_train_paths, starless_train_paths], f)
        
            # CSV logger
            with open('history.csv', 'a', newline = '') as csv:
                for i in reversed(range((save_freq, 1)[e == 0])):
                    if validation:
                        if e == 0: csv.write('epoch,loss_ema,psnr_ema,val_loss_ema,val_psnr_ema,loss_us,psnr_us,val_loss_us,val_psnr_us,lr\n')
                        csv.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'%
                            (e - i,
                            hist_train['total'][-1 - i*steps_per_epoch_train],
                            hist_train['psnr'][-1 - i*steps_per_epoch_train],
                            hist_val['total'][-1 - i*steps_per_epoch_validation],
                            hist_val['psnr'][-1 - i*steps_per_epoch_validation],
                            hist_train['loss_us'][-1 - i*steps_per_epoch_train],
                            hist_train['psnr_us'][-1 - i*steps_per_epoch_train],
                            hist_val['val_loss_us'][-1 - i*steps_per_epoch_validation],
                            hist_val['val_psnr_us'][-1 - i*steps_per_epoch_validation],
                            lr))
                    else:
                        if e == 0: csv.write('epoch,loss_ema,psnr_ema,lr\n')
                        csv.write('%d,%f,%f,%f,%f,%f\n'%
                            (e - i,
                            hist_train['total'][-1 - i*steps_per_epoch_train],
                            hist_train['psnr'][-1 - i*steps_per_epoch_train],
                            hist_train['loss_us'][-1 - i*steps_per_epoch_train],
                            hist_train['psnr_us'][-1 - i*steps_per_epoch_train],
                            lr))
            
            # plotting
            if validation and os.path.exists('history.csv'):
                df = pd.read_csv('history.csv')
                fig, ax = plt.subplots(1, 2, sharex = True)
                line1, = ax[0].plot(df.epoch, df.loss_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
                line2, = ax[0].plot(df.epoch, df.val_loss_us, 'r', label = 'validation unsmoothed', linewidth = 0.5)
                line3, = ax[0].plot(df.epoch, df.loss_ema, 'b', label = 'training ema decay = %.4f'%ema)
                line4, = ax[0].plot(df.epoch, df.val_loss_ema, 'r', label = 'validation ema decay = %.4f'%ema_val)
                ax[0].set_xlabel('epoch', fontsize = 12)
                ax[0].set_ylabel('loss', fontsize = 12)
                line1, = ax[1].plot(df.epoch, df.psnr_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
                line2, ax[1].plot(df.epoch, df.val_psnr_us, 'r', label = 'validation unsmoothed', linewidth = 0.5)
                line3, = ax[1].plot(df.epoch, df.psnr_ema, 'b', label = 'training ema decay = %.4f'%ema)
                line4, ax[1].plot(df.epoch, df.val_psnr_ema, 'r', label = 'validation ema decay = %.4f'%ema_val)
                ax[1].set_xlabel('epoch', fontsize = 12)
                ax[1].set_ylabel('psnr (dB)', fontsize = 12)
                fig.legend(handles = [line1, line2, line3, line4], loc = 'upper center', bbox_to_anchor = (0.5, 0), fancybox = True, ncol = 2)
                fig.tight_layout()
                plt.savefig(os.getcwd() + os.sep + 'training.png', bbox_inches = 'tight')
            elif ~validation and os.path.exists('history.csv'):
                df = pd.read_csv('history.csv')
                fig, ax = plt.subplots(1, 2, sharex = True)
                line1, = ax[0].plot(df.epoch, df.loss_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
                line2, = ax[0].plot(df.epoch, df.loss_ema, 'b', label = 'training ema decay = %.4f'%ema)
                ax[0].legend(handles=[line1, line2])
                ax[0].set_xlabel('epoch', fontsize = 12)
                ax[0].set_ylabel('loss', fontsize = 12)
                line1, = ax[1].plot(df.epoch, df.psnr_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
                line2, = ax[1].plot(df.epoch, df.psnr_ema, 'b', label = 'training ema decay = %.4f'%ema)
                ax[1].legend(handles = [line1, line2])
                ax[1].set_xlabel('epoch', fontsize = 12)
                ax[1].set_ylabel('psnr (dB)', fontsize = 12)
                fig.legend(handles = [line1, line2], loc = 'upper center', bbox_to_anchor = (0.5, 0), fancybox = True, ncol = 2)
                fig.tight_layout()
                plt.savefig(os.getcwd() + os.sep + 'training.png', bbox_inches = 'tight')
            
            if validation:
                starless_dir = os.getcwd() + os.sep + 'starless'
                if not os.path.exists(starless_dir):
                    os.makedirs(starless_dir)
                
                # process random image from data set
                i = np.random.randint(0, original_array_paths.shape[0]) # get a random image index
                original_image = cv2.imread(original_array_paths[i])
                original_image = prepare_image(original_image)
                save_image('input.png', original_image)
                gt_image = cv2.imread(starless_array_paths[i])
                gt_image = prepare_image(gt_image)
                save_image('gt.png', gt_image)
                starless_image = infer_image(G, original_image, border = border)
                save_image('output.png', starless_image)
                psnr1 = PSNR(original_image, gt_image).numpy()
                psnr2 = PSNR(starless_image, gt_image).numpy()
                print('PSNR(input, gt) = %f'%psnr1)
                print('PSNR(output, gt) = %f'%psnr2)
        
        # reduce learning rate on increasing loss
        if on_lr > cooldown:
            last_loss = []
            for i in hist_train['total'][::-steps_per_epoch_train]: # check training losses
                if len(last_loss) == 0 or i < last_loss[-1]: last_loss.append(i)
                else: break
                if len(last_loss) > patience:
                    lr/=2
                    gen_optimizer.learning_rate.assign(lr)
                    dis_optimizer.learning_rate.assign(lr/4)
                    on_lr = -1
                    break
                
            last_val_loss = []
            if on_lr != -1 and validation:
                for i in hist_val['total'][::-steps_per_epoch_validation]: # check validation losses
                    if len(last_val_loss) == 0 or i < last_val_loss[-1]: last_val_loss.append(i)
                    else: break
                    if len(last_val_loss) > patience:
                        lr/=2
                        gen_optimizer.learning_rate.assign(lr)
                        dis_optimizer.learning_rate.assign(lr/4)
                        on_lr = -1
                        break
        on_lr += 1

        # housekeeping
        gc.collect()
        tf.keras.backend.clear_session()
        
def prepare_image(image): # convert BGR image to RGB or gray 0..1 float image
    if image.dtype == 'uint8': image = image/255. # convert uint8 to 0..1 float
    if image.dtype == 'float64': image = image.astype(np.float32)
    if image.dtype == 'float32' and np.max(image) > 1.: image = image/255. # convert 0..255 float to 0..1 float
    if len(image.shape) == 3 and n_channels == 1: # if color but should be gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, :, np.newaxis]
    elif len(image.shape) == 3 and n_channels == 3: # if color and should be color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: # if gray
        image = image[:, :, np.newaxis]
    return image
    
def save_image(name, image): # save 0..1 float RGB or gray image
    if image.dtype == 'float64': image = image.astype(np.float32)
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # if color, convert to BGR
    if image.dtype == 'float32': image *= 255. # floats at this point will always be in range 0..1 so convert to 0..255 float
    cv2.imwrite(name, image)

def save_weights():
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    # create model
    tf.keras.backend.clear_session() # Release global state of old models and layers
    model = generator()
    model.summary()
    
    # load weights
    if os.path.exists(generator_fname):
        print('loading weights...')
        model.load_weights(generator_fname)
    else: return
    
    full_model = tf.function(lambda inputs: model(inputs))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def = frozen_func.graph,
        logdir = os.getcwd(),
        name = "noxGeneratorColor.pb",
        as_text = False)

if __name__ == "__main__":
    mode = 'train' # train mode by default
    if len(sys.argv) > 1: mode = sys.argv[1]
    
    if mode == 'train': train()
    elif mode == 'infer':
        if len(sys.argv) > 2: infer(sys.argv[2]) # file name
        else: infer() # draw randomly from dataset
    elif mode == 'save': save_weights()

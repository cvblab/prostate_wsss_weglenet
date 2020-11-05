
########################################################
# Imports
########################################################

import pandas as pd
import xlrd
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import random
import imutils
import os
import glob
from PIL import Image

from keras.metrics import *
from keras.utils import Sequence
from keras.applications import VGG19
from keras.layers import *
from keras.engine import Model
from keras import Sequential
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session

########################################################
# Data generator utils
########################################################


class DataGenerator(Sequence):

    def __init__(self, data_frame, path_images, input_shape=(224, 224, 3), batch_size=32, data_augmentation=False,
                 shuffle=False, hide_and_seek=False):

        # Inputs of the object
        self.path_images = path_images
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.shuffle = shuffle
        self.hide_and_seek = hide_and_seek

        # Data frame and indexes
        self.data_frame = data_frame
        self.indexes = np.arange(len(self.data_frame.index))
        self.n = len(self.data_frame.index)

    def __len__(self):
        ''' Returns the number of batches per epoch '''
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        ''' Returns a batch of data (the batches are indexed) '''
        # Take the id's of the batch number "index"
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Batch initialization
        X, Y = [], []

        # For each index,the sample and the label is taken. Then the batch is appended
        for idx in indexes:
            # Image and idx index tag is get
            x, y = self.get_sample(idx)
            # This image to the batch is added
            X.append(x)
            Y.append(y)
        # The created batch is returned
        return np.array(X), np.array(Y)

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes) # Shuffles the data

    def get_sample(self, idx):

        # Get the row from the dataframe corresponding to the index "idx"
        df_row = self.data_frame.iloc[idx]

        # Load label
        label = np.array(list(df_row[['G3', 'G4', 'G5']]))

        # Load image object
        image = Image.open(os.path.join(self.path_images, df_row['image_name']))

        # Resize image and convert into array
        image = image.resize(self.input_shape[:2])
        image = np.asarray(image)
        # Normalize intensity
        image = image / 255.0

        if self.data_augmentation:
            img_avg = np.average(image)
            image = image_transformation(image, bkg=img_avg)

        if self.hide_and_seek:
            img_avg = np.average(image)
            patch_list = get_random_patch_list(self.input_shape[0], self.input_shape[0]//10)
            image = random_hide(image, patch_list, hide_prob=0.25, mean=img_avg)

        image.astype(np.float32)
        return image, label

########################################################
# Image processing utils
########################################################


def random_hide(img, patch_list, hide_prob=0.5, mean=0.5):
    if type(img) is not np.ndarray:
        img = np.array(img)
    img = img.copy()
    np.random.seed()
    for patch in patch_list:
        (x, y, width, height) = patch
        if np.random.uniform() < hide_prob:
            img[x:x+width, y:y+height] = mean
    return img


def get_random_patch_list(img_size, patch_size):
    if img_size % patch_size != 0:
        raise Exception("patch_size cannot divide by img_size")
    patch_num = img_size//patch_size
    patch_list = [(x*patch_size, y*patch_size, patch_size, patch_size)
                  for y in range(patch_num)
                  for x in range(patch_num)]
    return patch_list


def image_transformation(im, bkg):

    random_index = np.clip(np.round(random.uniform(0, 1) * 10 / 2), 1, 4)

    if random_index == 1 or random_index == 3: # translation

        # Randomly obtain translation in pixels in certain bounds
        limit_translation = im.shape[0] / 4
        translation_X = np.round(random.uniform(-limit_translation, limit_translation))
        translation_Y = np.round(random.uniform(-limit_translation, limit_translation))
        # Get transformation function
        T = np.float32([[1, 0, translation_X], [0, 1, translation_Y]])
        # Apply transformation
        im_out = cv2.warpAffine(im, T, (im.shape[0], im.shape[1]), borderValue=(bkg, bkg, bkg))

    elif random_index == 2: # rotation

        # Get transformation function
        rotation_angle = np.round(random.uniform(0, 360))
        img_center = (im.shape[0] / 2, im.shape[0] / 2)
        T = cv2.getRotationMatrix2D(img_center, rotation_angle, 1)
        # Apply transformation
        im_out = cv2.warpAffine(im, T, (im.shape[0], im.shape[1]), borderValue=(bkg, bkg, bkg))

    elif random_index == 4: # mirroring

        rows, cols = im.shape[:2]
        src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        dst_points = np.float32([[cols - 1, 0], [0, 0], [cols - 1, rows - 1]])
        T = cv2.getAffineTransform(src_points, dst_points)

        im_out = cv2.warpAffine(im, T, (cols, rows), borderValue=(1, 1, 1))

    return im_out

########################################################
# Deep Learning utils
########################################################


def weglenet(input_shape, output_shape, aggregation='GMP', learning_rate=1*1e-3, freeze_up_to='', r=5):

    # Prepare model architecture
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze layers weights if required
    if freeze_up_to != '':
        for layer in base_model.layers[1:]:
            layer.trainable = False
            print('Capa ' + layer.name + ' congelada...')
            if freeze_up_to in layer.name:
                break

    # Segmentation layer
    base_model.layers.pop()
    x = base_model.layers[-1].output
    x = Conv2D(output_shape, (1, 1), padding="same")(x)
    x = Activation('softmax')(x)

    # Aggregation operator
    if aggregation == 'GAP':
        x = GlobalAveragePooling2D()(x)
    elif aggregation == 'GMP':
        x = GlobalMaxPool2D()(x)
    elif aggregation == 'LSE':
        x = global_log_sum_exp_pooling(x, r)

    # Slicing of non-cancerous class
    x = SliceLayer((0, 1), (-1, 3))(x)
    x._keras_shape = (None, 3)

    model = Model(base_model.input, x)

    # Loss, metrics and optimizer
    loss = binary_crossentropy
    metric = binary_accuracy
    optimizer = SGD(learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    print(model.summary())

    return model


def global_log_sum_exp_pooling(x, r=8):

    n = K.int_shape(x)[1]*K.int_shape(x)[2]
    return Lambda(lambda inputs: (1/r)*K.log((1/n) * tf.reduce_sum(K.exp(inputs*r), (1, 2))))(x)


class SliceLayer(Layer):
    def __init__(self, start, lengths):
        super(SliceLayer, self).__init__()
        self.start = start
        self.lengths = lengths
        self._keras_shape = (None, 3)

    def call(self, input):
        return tf.slice(input, self.start, self.lengths)


def scheduler(epoch):

  epoch_l = 380
  if epoch < epoch_l:
    return 1*1e-3
  else:
    return 1*1e-3 * math.exp(0.1 * (epoch_l - epoch))

########################################################
# Evaluation utils
########################################################


def learning_curve_plot_generalized(history, dir_out, name_out,
                                    metrics=['acc', 'val_acc'], losses=['loss', 'val_loss']):

    plt.figure()
    plt.subplot(211)
    for i in metrics:
        plt.plot(history.history[i])
    plt.axis([0, history.epoch[-1], 0, 1])
    plt.legend(metrics, loc='upper right')
    plt.title('learning-curve')
    plt.ylabel(metrics[0])
    plt.subplot(212)
    for i in losses:
        plt.plot(history.history[i])
    plt.axis([0, history.epoch[-1], 0, max(history.history[losses[0]])])
    plt.legend(losses, loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(dir_out + '/' + name_out)
    plt.close()


#!/usr/bin/env python
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import theano
import cv2
import numpy as np
import scipy as sp
import sys

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K
import h5py
import time
import os
import codecs


def get_features(model, layer, X_batch):
    get_features = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    features = get_features([X_batch, 0])
    return features


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    if weights_path:
        model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    dataPath = '../data/ner_img/' # root image path 
    tweet_data_path = '../data/tweet/all.txt' # all.txt store
    error = '../data/vgg_error_img_id' 
    store_img_feature = '../data/img_vgg_feature_224.h5' # image feature stored file    
    error_img_id = open(error, 'w') # store the image id which can not be resize 
    vgg_img_feature = h5py.File(store_img_feature, "w") # store the output -- img feature vector

    mean_pixel = [103.939, 116.779, 123.68]

    # load pretrained model
    model = VGG_16('../data/vgg16_weights_th_dim_ordering_th_kernels_notop.h5')

    img_id_list = [] # store image id 

    with codecs.open(tweet_data_path, 'r') as file:
        for line in file:
            rev = []
            rev = line.split('\t')
            img_id_list.append(rev[0])

    for item in img_id_list:
        print "process " + item + '.jpg'
        img_path = dataPath + item + '.jpg'
        print img_path
        try:
            im = cv2.resize(cv2.imread(img_path), (224,224))
        except:
            error_img_id.write(item + '\n')
            continue
        for c in range(3):
            im[:, :, c] = im[:, :, c] - mean_pixel[c]
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)

        start = time.time()

        features = get_features(model, 30, im)
        feat = features[0]

        print '%s feature extracted in %f  seconds.' % (img_path, time.time() - start)
        vgg_img_feature.create_dataset(name=item, data=feat) 

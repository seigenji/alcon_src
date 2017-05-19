#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from user_function import MyAlgorithm
from alcon_utils import AlconUtils
import numpy as np
import cv2

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model,Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from keras import backend as K
import keras
from keras.optimizers import SGD
import tensorflow as tf


def main(datasetdir,lv):    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    K.tensorflow_backend.set_session(tf.Session(config=config))
    
    # 初期化
    alcon = AlconUtils(datasetdir)

    # アノテーションの読み込み
    fn = "target_lv" + lv + ".csv"


    alcon.load_annotations_target(fn)

    fn = "groundtruth_lv" + lv + ".csv"
    alcon.load_annotations_ground(fn)

    
    # KNNモデルの作成
    dataset = {}
    print(len(alcon.targets.items()))
    for bb_id, target in alcon.targets.items():
        img_filename = alcon.get_filename_char( bb_id )
        code = alcon.ground_truth[bb_id][0]
        if code not in dataset:
            dataset[code] = []
        if len(dataset[code]) == 100:
            continue
        img = cv2.imread( img_filename )
        feature = MyAlgorithm.feature_extraction(img)
        dataset[code].append(feature)

    labels = []
    data = []
    classes = sorted(dataset.keys())
    print(len(dataset.items()))
    for label, values in dataset.items():
        labels += [classes.index(label)] * len(values)
        data += values

    data = np.asarray(data, dtype=np.float)
    labels = np.asarray(labels, dtype=np.int)

    print(data.shape)
    print(labels.shape)
#   classifier = KNeighborsClassifier()
#   classifier.fit(data, labels)

    batch_size = 128
    num_classes = 46
    epochs = 12
    img_rows, img_cols = 32, 32
#    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = data
    y_train = labels
    x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,3)
#    x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    input_shape = (img_rows, img_cols, 3)
    x_train = x_train.astype('float32')
#    x_test = x_test.astype('float32')
  
#    x_test /= 255
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
#    y_test = keras.utils.to_categorical(y_test, num_classes)
     
    
    
    
    classifier = Sequential()

    classifier.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape,padding='same'))
    classifier.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    classifier.add(Flatten())
    classifier.add(Dense(128,activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(num_classes, activation='softmax'))
    
    classifier.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


    classifier.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=None)

#    classifier.fit(data,labels)


    outputfile = "./model.pkl"
    outputfile2 = "./model2.pkl"
    joblib.dump(classes, outputfile)
    classifier.save(outputfile2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python train.py datasetdir lv", file=sys.stderr)
        quit()

    main(sys.argv[1], sys.argv[2])
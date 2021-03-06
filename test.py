import sys
import os
import os.path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from user_function import MyAlgorithm
from alcon_utils import AlconUtils
import numpy
import cv2

import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD
import tensorflow

def main(datasetdir,lv):    
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    backend.tensorflow_backend.set_session(tensorflow.Session(config=config))
    
    # 初期化
    alcon = AlconUtils(datasetdir)

    # アノテーションの読み込み
    alcon.load_annotations_target("target_lv"      + lv + ".csv")
    alcon.load_annotations_ground("groundtruth_lv" + lv + ".csv")
    
    # KNNモデルの作成
    dataset = {}
    for bb_id, target in alcon.targets.items(): # 49353 
        code = alcon.ground_truth[bb_id][0]
        if code not in dataset:
            dataset[code] = []
        if len(dataset[code]) == 75:
            continue
        img_filename = alcon.get_filename_char( bb_id )
        img = cv2.imread( img_filename )
        feature = MyAlgorithm.feature_extraction(img)
        dataset[code].append(feature)

    labels = []
    data = []
    classes = sorted(dataset.keys())
    len_values = len(list(dataset.values()))
    for label, values in dataset.items():
        labels += [classes.index(label)] * len_values
        data += values

    numpy_data = numpy.asarray(data, dtype=numpy.float)
    numpy_labels = numpy.asarray(labels, dtype=numpy.int)

    batch_size = 128
    num_classes = 46
    epochs = 12
    img_rows, img_cols = 32, 32
    channel = 1
    
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(img_rows, img_cols, channel),padding='same'))
    classifier.add(Conv2D(64,                 (3,3), activation='relu',                                  padding='same'))
    classifier.add(MaxPooling2D(pool_size=(6,6)))
    classifier.add(Dropout(0.25))
    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(num_classes, activation='softmax'))
    
    classifier.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Nadam(),metrics=['accuracy'])


    float_data = numpy_data.reshape(numpy_data.shape[0],img_rows,img_cols, channel )\
                           .astype('float32')
    y_train = keras.utils.to_categorical(numpy_labels, num_classes)
    classifier.fit(float_data,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=None)

    outputfile = "./model.pkl"
    outputfile2 = "./model2.pkl"
    joblib.dump(classes, outputfile)
    classifier.save(outputfile2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python train.py datasetdir lv", file=sys.stderr)
        quit()

    main(sys.argv[1], sys.argv[2])





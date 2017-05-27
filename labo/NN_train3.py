import sys
from sklearn.externals import joblib
from user_function import MyAlgorithm
from alcon_utils import AlconUtils
import numpy as np
import cv2

import keras
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import tensorflow
import cProfile, pstats

def main(datasetdir,lv, length):
    pr = cProfile.Profile()
    pr.enable()
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.93
    keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=config))
    
    # 初期化
    alcon = AlconUtils(datasetdir)

    # アノテーションの読み込み
    alcon.load_annotations_target("target_lv"      + lv + ".csv")
    alcon.load_annotations_ground("groundtruth_lv" + lv + ".csv")
    
    dataset = {}
    for bb_id, target in alcon.targets.items():
        code = alcon.ground_truth[bb_id][0]
        if code not in dataset:
            dataset[code] = []
        if len(dataset[code]) == int(length):
            continue
        img_filename = alcon.get_filename_char( bb_id )
        img = cv2.imread( img_filename )
        feature = MyAlgorithm.feature_extraction(img)
        dataset[code].append(feature)

    labels = []
    data = []
    classes = sorted(dataset.keys())

    for label, values in dataset.items():
        labels += [classes.index(label)] * len(values)
        data += values

    num_classes = 46
    input_shape = (32, 32, 1)# img_rows img_cols channel
    
    classifier = keras.models.Sequential()

    classifier.add(keras.layers.normalization.BatchNormalization(input_shape = input_shape))
    classifier.add(Conv2D(32, kernel_size=(3,3), activation='relu')) # 30*30
    classifier.add(Conv2D(64,                 (3,3), activation='relu')) # 28*28 
    classifier.add(Dropout(0.5))
    classifier.add(MaxPooling2D(pool_size=(4,4)))                      # 7*7
    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(num_classes, activation='softmax'))
    
    classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Nadam(),metrics=['accuracy'])
    x_data = np.asarray(data).reshape(len(data), *input_shape)
    y_train = keras.utils.to_categorical( labels, num_classes )
    classifier.fit(x_data, y_train, batch_size= 84, epochs=12)

    joblib.dump(classes, "./model.pkl")
    classifier.save("./model2.pkl")
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(5)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python train.py datasetdir lv", file=sys.stderr)
        quit()

    main(sys.argv[1], sys.argv[2], sys.argv[3])

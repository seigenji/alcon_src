# -*- coding: utf-8 -*-
"""
このMyAlgorithmを編集し，
自前のアルゴリズムを実装してください．

ただし，関数の入出力は変更しないでください．

1つのテストデータに対する処理は，
predict関数を編集してください．
他のメンバ関数の追加は自由です．

学習済みデータの読み込み等は自由に行って構いませんが，
読み込むファイルはuser_function.pyと同じディレクトリ
もしくはサブディレクトリに置いてください．
"""

import os.path
import cv2
from sklearn.externals import joblib
import keras
import numpy as np
import sys

class MyAlgorithm(object):
    """
    アルゴリズム
    """
    def __init__(self, datasetdir):
        self.datasetdir = datasetdir

        # 学習済み識別器等を読み込む
        trained_file = "./model2.pkl"
        self.classifier = keras.models.load_model(trained_file)
        self.classes = joblib.load("./model.pkl")
        
    def predict(self, full_img, bbox):
        """
        認識処理を記述する関数
        :param numpy.ndarray full_img: 1ページ全体の画像
        :param tuple[int] bbox: 対象となる領域のbounding box (x, y, w, h)
        :rtype list[str]
        :return ユニコードを表す文字列の配列 <-- レベル1は1つのunicode、レベル2は3つのunicode、レベル3は3つ以上のunicodeです
        """
        # 対象領域を切り出す
        x, y, w, h = bbox
        target_img = full_img[y:y+h, x:x+w, :]

        # 画像から特徴抽出
        channel = 1
        feature = self.feature_extraction(target_img).reshape((1,32,32, channel))
        
        # 識別器にかける
        recog_result = self.classifier.predict_classes(feature)
        codes = []
        for res in recog_result:
            codes.append(self.classes[res])

        return codes


    @classmethod
    def feature_extraction(cls, img):
        """
        特徴抽出
        """
        blur_image = cv2.bilateralFilter(img, 14, 12, 3)
        gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
        ret, binary_image = cv2.threshold(gray_image, 0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_ = cv2.resize(binary_image, (32, 32))
        thined_image = MyAlgorithm.thinning(img_)
        
        cv2.imshow("name", thined_image)
        sys.exit()
        
        return thined_image
    
    @staticmethod
    def thinning(img):
        kpw = np.array([[[ 0., 0., 0. ], [ 0., 1., 1. ], [ 0., 1., 0. ]],
                        [[ 0., 0., 0. ], [ 0., 1., 0. ], [ 1., 1., 0. ]],
                        [[ 0., 0., 0. ], [ 1., 1., 0. ], [ 0., 1., 0. ]],
                        [[ 1., 0., 0. ], [ 1., 1., 0. ], [ 0., 0., 0. ]],
                        [[ 0., 1., 0. ], [ 1., 1., 0. ], [ 0., 0., 0. ]],
                        [[ 0., 1., 1. ], [ 0., 1., 0. ], [ 0., 0., 0. ]],
                        [[ 0., 1., 0. ], [ 0., 1., 1. ], [ 0., 0., 0. ]],
                        [[ 0., 0., 0. ], [ 0., 1., 1. ], [ 0., 0., 1. ]]])
        kpb = np.array([[[ 1., 1., 0. ], [ 1., 0., 0. ], [ 0., 0., 0. ]],
                        [[ 1., 1., 1. ], [ 0., 0., 0. ], [ 0., 0., 0. ]],
                        [[ 0., 1., 1. ], [ 0., 0., 1. ], [ 0., 0., 0. ]],
                        [[ 0., 0., 1. ], [ 0., 0., 1. ], [ 0., 0., 1. ]],
                        [[ 0., 0., 0. ], [ 0., 0., 1. ], [ 0., 1., 1. ]],
                        [[ 0., 0., 0. ], [ 0., 0., 0. ], [ 1., 1., 1. ]],
                        [[ 0., 0., 0. ], [ 1., 0., 0. ], [ 1., 1., 0. ]],
                        [[ 1., 0., 0. ], [ 1., 0., 0. ], [ 1., 0., 0. ]]])
        src_w = np.array(img, dtype=np.float32)/255.
        thresh, src_b = cv2.threshold(src_w, 0.5, 1., cv2.THRESH_BINARY_INV)
        thresh, src_f = cv2.threshold(src_w, 0.5, 1., cv2.THRESH_BINARY)
        src_w = src_f.copy()
        th = 1.
        while th > 0:
            th = 0.
            for i in range(8):
                src_w = cv2.filter2D(src_w, cv2.CV_32F, kpw[i])
                src_b = cv2.filter2D(src_b, cv2.CV_32F, kpb[i])
                thresh, src_w = cv2.threshold(src_w, 2.99, 1, cv2.THRESH_BINARY)
                thresh, src_b = cv2.threshold(src_b, 2.99, 1, cv2.THRESH_BINARY)
                src_w = np.array(np.logical_and(src_w,src_b), dtype=np.float32)
                th += np.sum(src_w)
                src_f = np.array(np.logical_xor(src_f, src_w), dtype=np.float32)
                src_w = src_f.copy()
                thresh, src_b = cv2.threshold(src_f, 0.5, 1.0, cv2.THRESH_BINARY_INV)
        return src_f


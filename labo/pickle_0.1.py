from user_function import MyAlgorithm
from alcon_utils import AlconUtils
import cv2

import os.path
import pickle

datasetdir = "../../dataset/"
annotation_name = "iiyama_0.1"
"""
評価コードのメイン
:param datasetdir データセットを格納したディレクトリへのパス
"""

# 初期化
alcon = AlconUtils(datasetdir)
myalgorithm = MyAlgorithm(datasetdir)

# ターゲットの読み込み
file_name_last = "_lv1_" + annotation_name + ".csv"
alcon.load_annotations_target("target" + file_name_last)

imgs = {}
results_pre = {}
# １ターゲットずつ認識していくループ
print("coco")

def predict_preprocess(full_img, bbox):
# 対象領域を切り出す
    x, y, w, h = bbox
    target_img = full_img[y:y+h, x:x+w, :]

    # 画像から特徴抽出
    return MyAlgorithm.feature_extraction(target_img)

for bb_id, target in alcon.targets.items():
    img_file_id, *bb = target
    print(bb_id)
    # ページ全体の画像
    imgs[bb_id] = cv2.imread( os.path.join(datasetdir, "images", img_file_id+".jpg") )
    results_pre[bb_id] = predict_preprocess(imgs[bb_id], bb)
print("coco")

# 評価
alcon.load_annotations_ground("groundtruth" + file_name_last)
alcon.evaluation( results )
print("coco")


print(len(results_pre))
        
#with open('test0.1', 'wb') as f:
 #   pickle.dump(results_pre, f)
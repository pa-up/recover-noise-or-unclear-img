""" 
このファイルは静止画像に画像処理を実行する関数だけではなく、
動画・WEBカメラに対して、フレーム毎に画像処理を実行する関数も利用可
"""


import cv2
import numpy as np
from PIL import Image
import random
import time


def no_change(cv_img, func2):
    """ 無修正 """
    return cv_img



def gray(cv_img, func2):
    """ グレースケール化 """
    cv_calc_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    cv_calc_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv_calc_img = cv2.cvtColor(cv_calc_img, cv2.COLOR_GRAY2BGR)
    return cv_calc_img



def binar2pil(binary_img):
    """ バイナリ画像PIL画像に変換 """
    pil_img = Image.open(binary_img)
    return pil_img



def binar2opencv(binary_img):
    """ バイナリ画像をOpenCV画像に変換 """
    pil_img = Image.open(binary_img)
    cv_img = pil2opencv(pil_img)
    return cv_img



def pil2opencv(pil_img):
    """ PIL画像をOpenCV画像に変換 """
    cv_img = np.array(pil_img, dtype=np.uint8)

    if cv_img.ndim == 2:  # モノクロ
        pass
    elif cv_img.shape[2] == 3:  # カラー
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    elif cv_img.shape[2] == 4:  # 透過
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)
    return cv_img



def opencv2pil(cv_calc_img):
    """ OpenCV画像をPIL画像に変換 """
    pil_img = cv_calc_img.copy()
    
    if pil_img.ndim == 2:  # モノクロ
        pass
    elif pil_img.shape[2] == 3:  # カラー
        pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)
    elif pil_img.shape[2] == 4:  # 透過
        pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(pil_img)
    return pil_img



def max_size(cv_img , max_img_size):
    """ 
    縦横の倍率を保ちながら、画像の辺の長さの最大値を定義
    例）max_img_size = 1500 : 画像の縦または横サイズの最大値を1500に制限
    """
    rows = cv_img.shape[0]
    cols = cv_img.shape[1]
    new_row = rows
    new_col = cols
    if (rows >= cols)  and (rows > max_img_size) :
        new_row = max_img_size
        new_col = int( cols / (rows/max_img_size) )
    #
    if (cols > rows)  and (cols > max_img_size) :
        new_col = max_img_size
        new_row = int( rows / (cols/max_img_size) )
    #
    cv_img = cv2.resize( cv_img , dsize=(new_col, new_row) )
    return cv_img


def generate_input_img_path():
    # 現在時刻をシード値として使用
    random.seed(time.time())
    digits = [str(random.randint(0, 9)) for _ in range(7)]
    input_img_path = "".join(digits) + ".jpg"
    return input_img_path



def brightness(input_image , gamma):
  """ 
  画像の明るさ（輝度）を変える関数
  gamma > 1  =>  明るくなる
  gamma < 1  =>  暗くなる 
  """
  img2gamma = np.zeros((256,1),dtype=np.uint8)  # ガンマ変換初期値

  for i in range(256):
    # ガンマ補正の公式 : Y = 255(X/255)**(1/γ)
    img2gamma[i][0] = 255 * (float(i)/255) ** (1.0 /gamma)
  
  # 読込画像をガンマ変換
  gamma_img = cv2.LUT(input_image , img2gamma)
  return gamma_img


def mosaic(input_image , parameter):
  """ 
  モザイク処理（画像を縮小してモザイク効果を適用）
  parameter : リサイズにおける 縦＝横 サイズ（小さいほどモザイクが強くなる）
  例）一般的には parameter = 25 ~ 50 など
  """
  mosaic_img = cv2.resize(input_image, (parameter , parameter), interpolation=cv2.INTER_NEAREST)
  mosaic_img = cv2.resize(mosaic_img, input_image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
  return mosaic_img



def mask(input_image , threshold):
  """ 
   2値化（マスク）処理 
   threshold : しきい値（ 0 ~ 255 の 整数値）
  """
  # グレースケール化
  img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
  # 2値化
  ret, mask_img = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
  # 2値画像を3チャンネルに拡張する
  mask_img_3ch = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
  return mask_img_3ch


def back_convert_func(img_conversion):
    """ 入力した文字列に応じて、任意の画像処理関数とパラメータを返す関数 """
    """ WEB上でのユーザーからのform入力に対して、コマンドを返すことにも利用可 """
    convert_img_func = no_change
    convert_object_parameter = ""
    if img_conversion == "mosaic":
        convert_img_func = mosaic
        convert_object_parameter = 25
    if img_conversion == "mask":
        convert_img_func = mask
        convert_object_parameter = 120
    if img_conversion == "light":
        convert_img_func = brightness
        convert_object_parameter = 2
    if img_conversion == "dark":
        convert_img_func = brightness
        convert_object_parameter = 0.5
    if img_conversion == "gray":
        convert_img_func = gray
        convert_object_parameter = ""
    
    return convert_img_func , convert_object_parameter



def video_information(input_video_file , output_video_file):
  """
  総再生時間、総フレーム数の表示 : frame_number , total_time
  動画の書き込み形式の取得 ： fmt , writer
  """

  # 動画をフレームに分割
  cap = cv2.VideoCapture(input_video_file)

  #動画サイズ取得
  width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
  height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

  #フレームレート取得
  fps = cap.get(cv2.CAP_PROP_FPS)

  #フォーマット指定（動画の書き込み形式）
  fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  writer = cv2.VideoWriter( output_video_file , fmt, fps, (width, height) )

  # 表示
  print("合計フレーム数：")
  frame_number = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) )
  print(f"{frame_number} 枚 \n")

  print("合計再生時間（総フレーム数 / FPS）：")
  total_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
  total_time = round(total_time , 3)
  print(f"{total_time} 秒  \n \n")

  return cap , writer , fmt , fps , width , height



def extract_frames(cap):
    """ 動画のフレーム（画像）を一つのリストに格納する関数 """
    frames_list = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames_list.append(frame)
        else:
            break
    frames_np = np.array(frames_list)
    return frames_list , frames_np


def cascade_convert_detect_area(
    input_img , 
    cascade_information , 
    ):
    """
    OpenCVのカスケード分類器を用いて、特定の物体を検出し、その領域に任意の画像処理を実行する関数
    """
    # 分類器の読み込み
    cascade = cv2.CascadeClassifier(cascade_information[0])
    # 物体検出の実行
    detected_object = cascade.detectMultiScale(input_img , scaleFactor=1.1, minNeighbors=5)

    # 検出領域の内側に画像処理を実行
    if cascade_information[1] == "inside":
        for (x, y, w, h) in detected_object:
            detection_area = input_img[y:y+h, x:x+w]
            output_img = cascade_information[2](
                detection_area , 
                cascade_information[3] ,
            )
            input_img[y:y+h, x:x+w] = output_img
    # 検出領域の外側に画像処理を実行
    if cascade_information[1] == "background":
        for (x, y, w, h) in detected_object:
            detection_area = input_img[y:y+h, x:x+w]
            input_img = cascade_information[2](
                input_img , 
                cascade_information[3] ,
            )
            input_img[y:y+h, x:x+w] = detection_area

    return input_img


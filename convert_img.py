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


def add_gaussian_noise(image, mean=0, sigma=None):
    """ガウシアンノイズを加えた画像を生成する関数

    Args:
        image (numpy.ndarray): 入力画像
        mean (float): ガウス分布の平均値（デフォルトは0）
        sigma (float or None): ガウス分布の標準偏差（デフォルトはNone 、sigma ∝ ノイズの大きさ）

    Returns:
        numpy.ndarray: ガウシアンノイズを加えた画像
    """
    if sigma is None:
        np.random.seed(int(time.time() * 1000000 % 4294967296))  # 現在時刻のマイクロ秒以下を含める
        sigma = np.random.randint(50, 101)  # 50から100までのランダムな整数値を生成
        print(f"sigma : {sigma}")
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def generate_random_from_time(min , max):
  """ 
  ランダムにmin〜maxの範囲で数字を生成する関数 
  現在時刻（ナノ秒単位）をシード値
  """
  # 現在時刻を乱数のシード値に設定
  random.seed(time.time_ns() % 4294967296)
  # min 〜 max のランダムな整数を生成
  rand_num = random.randint(min , max)
  return rand_num


def marge_two_img(img1 , img2):
  """ 2枚の画像を左右に連結する関数（同じ高さの画像のみ）"""
  img1_height , img1_width = img1.shape[0] , img1.shape[1]
  img2_height , img2_width = img2.shape[0] , img2.shape[1]
  # 左右を分割する縦線の太さ
  line_weight = 5
  # 緑色の画像を作成
  marged_img = np.ones( (img1_height , img1_width + img2_width + line_weight , 3) )
  marged_img[: , :] = [0, 255, 0]   # 画像全体を緑色に変換
  marged_img[ : , : img1_width] = img1
  marged_img[ : , img1_width + line_weight : img1_width + img2_width + line_weight] = img2
  return marged_img



def generate_input_img_path():
  """ 入力画像の保存ファイル名を生成"""
  # 現在時刻をシード値として使用
  random.seed(time.time())
  digits = [str(random.randint(0, 9)) for _ in range(7)]
  input_img_path = "".join(digits) + ".jpg"
  return input_img_path



def resize_to_square(input_img , resized_length):
  """ 
  入力画像を正方形に収まるようにリサイズし、余白を黒で塗りつぶす関数
  （リサイズ後の画像を左 or 上に敷き詰め、画像を縦横の短い方向を黒で塗りつぶす）
  """
  input_height = input_img.shape[0]
  input_width = input_img.shape[1]
  
  # 入力画像が正方形の場合
  if input_width == input_height:
    resized_height , resized_width = resized_length , resized_length
    resized_input_img = cv2.resize( input_img, (resized_height , resized_width) )
    resized_square_img = resized_input_img

  # 入力画像が縦長の場合
  if  input_width < input_height:
    resized_height , resized_width = resized_length , int( input_width * resized_length / input_height )
    resized_input_img = cv2.resize( input_img, (resized_width , resized_height) )
    # 画像を正方形の左に敷き詰め、右の余白を黒で埋め尽くす
    resized_square_img = np.zeros( (resized_length , resized_length , 3) )
    resized_square_img[ : ,  : resized_width] = resized_input_img
  
  # 入力画像が横長の場合
  if  input_width > input_height:
    resized_height , resized_width = int( input_height * resized_length / input_width ) , resized_length
    resized_input_img = cv2.resize( input_img, (resized_width , resized_height) )
    # 画像を正方形の上に敷き詰め、下の余白を黒で埋め尽くす
    resized_square_img = np.zeros( (resized_length , resized_length , 3) )
    resized_square_img[ : resized_height ,  : ] = resized_input_img

  resized_square_img = resized_square_img.astype(np.float32)
  return resized_square_img , resized_input_img



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


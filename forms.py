import numpy as np

def get_form_name_list(choice):
    """ formに表示する文字列を取得 """
    choice_name_list = choice[: , 1]
    return choice_name_list

def form_name_to_variable_data(choice, form_name):
    """ formで得た文字列を変数のデータ """
    form_name_row = np.where(choice[:,1] == form_name)[0][0]
    variable_data = choice[form_name_row , 0]
    return variable_data


inside_or_background = np.array([
    ['inside', '内側'],
    ['background', '外側'],
])


convert_choice= np.array([
    ['normal', '無加工'],
    ['mosaic', 'モザイクをかける'],
    ['mask', '白黒画像にする'],
    ['light', '明るくする'],
    ['dark', '暗くする'],
    ['gray', 'グレースケール画像にする'],
])


person_cascade_path =  "./CascadeClassifier/person/haarcascade_fullbody.xml"
car_cascade_path =  "./CascadeClassifier/vehicle/cars.xml"
face_cascade_path =  "./CascadeClassifier/person/haarcascade_frontalface_default.xml"

all_img_label = np.array([
    ['all', '画像全体'],
])

# 物体毎のカスケード検出器とform入力情報を1対1で対応
cascade_label = np.array([
    [person_cascade_path, '人間（全身）'],
    [face_cascade_path, '人間（顔）'],
    [car_cascade_path, '車'],
])

# フォームのラベルの選択肢
label_choice = np.concatenate((all_img_label, cascade_label), axis=0)

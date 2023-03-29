import streamlit as st
import convert_img
import numpy as np
from tensorflow.keras.models import load_model

import streamlit as st
from PIL import Image
import numpy as np

def run_CAE(img , model):
    # 入力画像をモデルの入力サイズにリサイズ
    resized_length = 200
    resized_square_img , resized_input_img = convert_img.resize_to_square(img , resized_length)
    img = np.reshape(resized_square_img, (1, resized_length, resized_length, 3))
    # 入力画像の正規化
    model_input = img / 255.
    # 予測結果を出力
    pred_result = model.predict(model_input)
    # 予測結果を画像に変換
    pred_img = pred_result * 255
    output_img = pred_img[0][ : int(resized_input_img.shape[0] * 0.98) , : int(resized_input_img.shape[1] * 0.98) ]
    return output_img , resized_input_img


def upload_and_display_img(model_path, page_subtitle):
    st.write(page_subtitle, unsafe_allow_html=True)
    # 学習済みのCAEモデルを読み込む
    tf_CAE_model = load_model(model_path)

    # 画像のアップロード
    uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=['jpg', 'jpeg', 'png'])


    if uploaded_file is not None:
        # PILで画像を開く
        img_pil = Image.open(uploaded_file)
        
        # Numpyの配列に変換
        img_np = np.array(img_pil)
        # 透明度チャンネルがある場合はRGBに変換
        img_np = img_np[..., :3]
        
        # アップロード画像に画像認識を実行
        output_img , resized_input_img = run_CAE(img_np , tf_CAE_model)
        
        # PILのImageに戻す
        output_pil = Image.fromarray(np.uint8(output_img))
        input_pil = Image.fromarray(np.uint8(resized_input_img))

        col1, col2 = st.columns(2)
        # 元の画像を表示
        col1.image(input_pil, caption='元の画像', use_column_width=True)
        # 処理結果を表示
        col2.image(output_pil, caption='処理結果', use_column_width=True)

if __name__ == "__main__":
    upload_and_display_img()
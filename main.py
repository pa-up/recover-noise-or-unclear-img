import streamlit as st
import convert_unclear_img

st.title('モザイク・ノイズ画像の復元サイト')

col1, col2 = st.columns(2)

if col1.button('虫食い・モザイク画像の復元はこちら'):
    st.write("\n")
    # ボタンが押されたときに実行する関数
    model_path = "tf_CAE_ImgNet_mosaic.h5"
    convert_unclear_img.display(model_path)
    st.write("\n")
    st.write("\n")

if col2.button('ノイズが多い画像の復元はこちら'):
    st.write("\n")
    # ボタンが押されたときに実行する関数
    model_path = "tf_CAE_ImgNet_noise.h5"
    convert_unclear_img.display(model_path)
    st.write("\n")
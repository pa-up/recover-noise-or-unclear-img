import streamlit as st
import views

st.title('モザイク・ノイズ画像の復元')

def change_page(page):
    st.session_state["page"] = page
    
def mosaic_page():
    # 別ページへの遷移
    st.button("ノイズ画像の復元はこちら >", on_click=change_page, args=["page2"])
    st.write('\n')
    
    # 画像処理の実行
    model_path = "tf_CAE_ImgNet_mosaic.h5"
    page_subtitle = "<h3>虫食い・モザイク画像の復元</h3>"
    views.upload_and_display_img(model_path, page_subtitle)

def noise_page():
    # 別ページへの遷移
    st.button("虫食い・モザイク画像の復元はこちら >", on_click=change_page, args=["page1"])
    st.write('\n')

    # 画像処理の実行
    model_path = "tf_CAE_ImgNet_noise.h5"
    page_subtitle = "<h3>ノイズ画像の復元</h3>"
    views.upload_and_display_img(model_path, page_subtitle)


# メイン
def main():
    # セッション状態を取得
    session_state = st.session_state

    # セッション状態によってページを表示
    if "page" not in session_state:
        session_state["page"] = "page1"

    if session_state["page"] == "page1":
        mosaic_page()
    elif session_state["page"] == "page2":
        noise_page()

if __name__ == "__main__":
    main()

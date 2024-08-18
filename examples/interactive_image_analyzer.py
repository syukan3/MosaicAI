# $ streamlit run streamlit_sample.py

import streamlit as st
from PIL import Image
from mosaicai import MosaicAI
import tempfile
import os

# Streamlitアプリのタイトルを設定
st.title("AI画像分析ツール")

# ファイルアップローダーを作成し、画像ファイルのみを許可
uploaded_file = st.file_uploader(
    "画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # アップロードされた画像を表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # MosaicAIクライアントを初期化
    client = MosaicAI(model="gpt-4o")

    # アップロードされたファイルを一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 初期分析を実行
        initial_analysis = client.generate_with_image(
            "この画像を詳しく分析してください", tmp_file_path)
        st.write("初期分析:", initial_analysis)

        # ユーザーからの質問を受け付ける
        user_question = st.text_input("画像について質問してください")
        if user_question:
            # ユーザーの質問に基づいて画像分析を実行
            answer = client.generate_with_image(user_question, tmp_file_path)
            st.write("回答:", answer)
    finally:
        # 処理が終了したら一時ファイルを削除
        os.unlink(tmp_file_path)

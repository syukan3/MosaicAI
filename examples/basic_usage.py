from mosaicai import MosaicAI


def main():
    # 使用するモデルを指定してMosaicAIインスタンスを作成
    client = MosaicAI(model="gpt-4o")
    response = client.generate_text("AIの未来について教えてください")
    print("テキスト生成:", response)

    # 別のモデルを使用する場合
    client = MosaicAI(model="claude-3-5-sonnet-20240620")
    response = client.generate_text("AIの倫理について教えてください")
    print("テキスト生成 (Claude):", response)

    client = MosaicAI(model="gemini-1.5-pro")
    response = client.generate_text("AIと人間の共存について教えてください")
    print("テキスト生成 (Gemini):", response)

    client = MosaicAI(model="llama-3.1-sonar-large-128k-online")
    response = client.generate_text("AIの限界について教えてください")
    print("テキスト生成 (Perplexity):", response)

    # 画像を使用した生成の例
    client = MosaicAI(model="gpt-4o")
    image_path = "path/to/your/image.jpg"
    response = client.generate_with_image("この画像について説明してください", image_path)
    print("画像を使用した生成:", response)

    # JSON生成の例
    client = MosaicAI(model="gpt-4o")
    schema = {
        "title": "str",
        "main_points": "List[str]",
        "value": "int"
    }
    response = client.generate_json("AIの倫理的課題について3つのポイントを挙げて説明してください", schema)
    print("JSON生成:", response)

    # 画像を使用したJSON生成の例
    client = MosaicAI(model="gpt-4o")
    image_path = "path/to/your/image.jpg"
    schema = {
        "objects": "List[str]",
        "main_colors": "List[str]",
        "scene_description": "str"
    }
    response = client.generate_with_image_json("この画像の内容をJSONで説明してください", image_path, schema)
    print("画像を使用したJSON生成:", response)


if __name__ == "__main__":
    main()

from mosaicai import MosaicAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Type


def main():
    ####################################################################################################
    # 使用するモデルを指定してMosaicAIインスタンスを作成
    client = MosaicAI(model="gpt-4o")
    response = client.generate_text("AIの未来について教えてください")
    print("テキスト生成:", response)
    ####################################################################################################

    ####################################################################################################
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
    ####################################################################################################

    ####################################################################################################
    # 画像を使用した生成の例
    client = MosaicAI(model="gpt-4o")
    image_path = "path/to/your/image.jpg"
    response = client.generate_with_image("この画像について説明してください", image_path)
    print("画像を使用した生成:", response)
    ####################################################################################################

    ####################################################################################################
    # JSON生成の例
    class AIResponse(BaseModel):
        title: str = Field(description="タイトル")
        points: List[str] = Field(description="主要なポイント")
        summary: str = Field(description="要約")
        integer_value: int = Field(description="整数値")
        float_value: float = Field(description="浮動小数点値")
        boolean_flag: bool = Field(description="ブール値フラグ")
        nested_object: Dict[str, Any] = Field(description="ネストされたオブジェクト")
        array_of_numbers: List[float] = Field(description="数値の配列")

    client = MosaicAI(model="gpt-4o")
    schema = AIResponse
    response = client.generate_json("AIの倫理的課題について3つのポイントを挙げて説明してください", schema)
    print("JSON生成:", response)
    ####################################################################################################

    ####################################################################################################
    # 画像を使用したJSON生成の例
    class ImageDescription(BaseModel):
        product_name: str = Field(description="製品名")
        features: List[str] = Field(description="製品の特徴リスト")
        overall_impression: str = Field(description="全体的な印象")
        price: float = Field(description="価格")
        is_available: bool = Field(description="利用可能かどうか")
        release_date: str = Field(description="発売日")
        specifications: Dict[str, Union[str, int, float, bool]] = Field(description="製品の仕様")
        ratings: List[int] = Field(description="評価のリスト")

    client = MosaicAI(model="gpt-4o")
    image_path = "path/to/your/image.jpg"
    schema = ImageDescription
    response = client.generate_with_image_json("この画像の内容をJSONで説明してください", image_path, schema)
    print("画像を使用したJSON生成:", response)
    ####################################################################################################


if __name__ == "__main__":
    main()

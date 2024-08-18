import pytest
from pydantic import BaseModel, Field
from mosaicai import MosaicAI


@pytest.fixture(params=["gpt-4o", "claude-3-5-sonnet-20240620", "gemini-1.5-pro", "llama-3.1-sonar-large-128k-online"])
def mosaic_ai_client(request):
    """MosaicAIクライアントのフィクスチャを提供します。"""
    client = MosaicAI(model=request.param)
    return client


def test_text_generation(mosaic_ai_client):
    """
    各モデルでのテキスト生成をテストします。
    異なるモデルで正常にテキストが生成されることを確認します。
    """
    response = mosaic_ai_client.generate_text(f"{mosaic_ai_client.get_model()}について教えてください")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.parametrize("model", ["gpt-4o", "claude-3-5-sonnet-20240620"])
def test_image_generation(model):
    """
    画像を使用したテキスト生成をテストします。
    画像に基づいて適切な説明が生成されることを確認します。
    """
    client = MosaicAI(model=model)
    image_path = "./tests/test_image.jpg"  # 実際には適切な画像パスを指定してください
    response = client.generate_with_image("この画像について説明してください", image_path)
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.parametrize("model", ["gpt-4o", "claude-3-5-sonnet-20240620", "gemini-1.5-pro"])
def test_json_generation(model):
    """
    JSON形式でのレスポンス生成をテストします。
    指定されたスキーマに従ってJSONが正しく生成されることを確認します。
    """
    client = MosaicAI(model=model)

    class AIResponse(BaseModel):
        ethical_concerns: list[str] = Field(description="AIの倫理的な懸念")
        potential_benefits: list[str] = Field(description="AIの潜在的な利点")
        future_of_ai: str = Field(description="AIの未来")
        ai_adoption_rate: float = Field(description="AIの普及率")
        is_ai_safe: bool = Field(description="AIは安全か")

    response = client.generate_json("AIの倫理的課題、潜在的な利点、そして未来について説明してください。", AIResponse)
    assert isinstance(response, dict)
    assert "ethical_concerns" in response
    assert "potential_benefits" in response
    assert "future_of_ai" in response
    assert "ai_adoption_rate" in response
    assert "is_ai_safe" in response
    assert isinstance(response["ethical_concerns"], list)
    assert isinstance(response["potential_benefits"], list)
    assert isinstance(response["future_of_ai"], str)
    assert isinstance(response["ai_adoption_rate"], float)
    assert isinstance(response["is_ai_safe"], bool)


@pytest.mark.parametrize("model", ["gpt-4o", "claude-3-5-sonnet-20240620"])
def test_image_json_generation(model):
    """
    画像を使用したJSON形式でのレスポンス生成をテストします。
    画像に基づいて、指定されたスキーマに従ってJSONが正しく生成されることを確認します。
    """
    client = MosaicAI(model=model)
    image_path = "./tests/test_image.jpg"

    class ImageDescription(BaseModel):
        objects: list[str] = Field(description="画像に含まれるオブジェクトのリスト")
        main_colors: list[str] = Field(description="画像の主要な色のリスト")
        scene_description: str = Field(description="画像のシーンの説明")
        image_quality: float = Field(description="画像の品質")
        is_image_clear: bool = Field(description="画像は鮮明か")

    response = client.generate_with_image_json(
        "この画像の内容をJSONで説明してください。", image_path, ImageDescription)
    assert isinstance(response, dict)
    assert "objects" in response
    assert "main_colors" in response
    assert "scene_description" in response
    assert "image_quality" in response
    assert "is_image_clear" in response
    assert isinstance(response["objects"], list)
    assert isinstance(response["main_colors"], list)
    assert isinstance(response["scene_description"], str)
    assert isinstance(response["image_quality"], float)
    assert isinstance(response["is_image_clear"], bool)


def test_invalid_model():
    """
    無効なモデル名を指定した場合のエラー処理をテストします。
    適切な例外が発生することを確認します。
    """
    with pytest.raises(ValueError):
        MosaicAI(model="invalid_model")


@pytest.mark.parametrize("model", ["gpt-4o", "claude-3-5-sonnet-20240620", "gemini-1.5-pro", "llama-3.1-sonar-large-128k-online"])
def test_empty_prompt(model):
    """
    空のプロンプトを指定した場合のエラー処理をテストします。
    適切な例外が発生することを確認します。
    """
    client = MosaicAI(model=model)
    with pytest.raises(ValueError):
        client.generate_text("")


@pytest.mark.parametrize("model", ["gpt-4o", "claude-3-5-sonnet-20240620"])
def test_non_existent_image(model):
    """
    存在しない画像ファイルを指定した場合のエラー処理をテストします。
    適切な例外が発生することを確認します。
    """
    client = MosaicAI(model=model)
    non_existent_image = "non_existent_image.jpg"
    with pytest.raises(FileNotFoundError):
        client.generate_with_image("この画像について説明してください", non_existent_image)

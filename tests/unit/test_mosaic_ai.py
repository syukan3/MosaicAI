import pytest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from mosaicai import MosaicAI
from mosaicai.exceptions import ModelNotSupportedError


@pytest.fixture
def mosaicai():
    load_dotenv
    return MosaicAI("gpt-4o")


def test_initialize_model(mosaicai):
    # モデルが正しく初期化されていることを確認
    assert "gpt-4o" in mosaicai.models


def test_get_model(mosaicai):
    # 正しいモデルが返されることを確認
    assert mosaicai.get_model() == "gpt-4o"


def test_set_api_key(mosaicai):
    # APIキーが正しく設定されていることを確認
    mosaicai.set_api_key("openai", "test-api-key")
    assert mosaicai.api_key_manager.get_api_key("openai") == "test-api-key"


@pytest.mark.parametrize("model", ["gpt-4", "claude-2", "gemini-pro", "llama-2"])
def test_initialize_different_models(model):
    # 異なるモデルが正しく初期化されていることを確認
    ai = MosaicAI(model)
    assert model in ai.models


def test_initialize_unsupported_model():
    # サポートされていないモデルで初期化しようとするとエラーが発生することを確認
    with pytest.raises(ValueError):
        MosaicAI("unsupported-model")


@patch.object(MosaicAI, 'generate_text')
def test_generate_text(mock_generate_text, mosaicai):
    # generate_textメソッドが正しく呼び出され、結果が返されることを確認
    mock_generate_text.return_value = "Test text"
    result = mosaicai.generate_text("Test prompt. Please return 'Test text'.")
    assert result == "Test text"
    mock_generate_text.assert_called_once_with("Test prompt. Please return 'Test text'.")


@patch.object(MosaicAI, 'generate_with_image')
def test_generate_image_description(mock_generate_with_image, mosaicai):
    # generate_with_imageメソッドが正しく呼び出され、結果が返されることを確認
    mock_generate_with_image.return_value = "Image description"
    result = mosaicai.generate_image_description(
        "Test prompt. Please return 'Test image description'.", "./tests/test_image.jpg")
    assert result == "Test image description."


@patch.object(MosaicAI, 'generate_json')
def test_generate_json(mock_generate_json, mosaicai):
    # generate_jsonメソッドが正しく呼び出され、結果が返されることを確認
    schema = {"name": "string", "age": "integer"}
    mock_generate_json.return_value = {"name": "John", "age": 30}
    result = mosaicai.generate_json("Create a person", schema)
    assert result == {"name": "John", "age": 30}
    mock_generate_json.assert_called_once_with("Create a person", schema)


@patch.object(MosaicAI, 'generate_with_image')
def test_generate_with_image(mock_generate_with_image, mosaicai):
    # generate_with_imageメソッドが正しく呼び出され、結果が返されることを確認
    mock_generate_with_image.return_value = "Generated text with image"
    result = mosaicai.generate_with_image("Describe this", "image.jpg")
    assert result == "Generated text with image"
    mock_generate_with_image.assert_called_once_with(
        "Describe this", "image.jpg")


@patch.object(MosaicAI, 'generate_with_image_json')
def test_generate_with_image_json(mock_generate_with_image_json, mosaicai):
    # generate_with_image_jsonメソッドが正しく呼び出され、結果が返されることを確認
    schema = {"description": "string", "objects": "list"}
    mock_generate_with_image_json.return_value = {
        "description": "A cat", "objects": ["cat", "sofa"]}
    result = mosaicai.generate_with_image_json("Analyze this image", "image.jpg", schema)
    assert result == {"description": "A cat", "objects": ["cat", "sofa"]}
    mock_generate_with_image_json.assert_called_once_with(
        "Analyze this image", "image.jpg", schema)

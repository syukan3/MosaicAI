import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import json
import google.generativeai as genai
from mosaicai.models.gemini import Gemini
from mosaicai.utils.api_key_manager import APIKeyManager


# APIKeyManagerのモック
@pytest.fixture
def mock_api_key_manager():
    manager = MagicMock(spec=APIKeyManager)
    manager.get_api_key.return_value = "fake_api_key"
    return manager


# Geminiクラスのインスタンス化のテスト
def test_gemini_initialization(mock_api_key_manager):
    # Geminiクラスのインスタンス化をテスト
    gemini = Gemini(mock_api_key_manager)
    assert isinstance(gemini.model, genai.GenerativeModel)  # モデルがGenerativeModelオブジェクトであることを確認

    # APIキーが設定されていない場合のテスト
    mock_api_key_manager.get_api_key.return_value = None
    with pytest.raises(ValueError, match="Gemini APIキーが設定されていません。"):
        Gemini(mock_api_key_manager)


# get_modelメソッドのテスト
def test_get_model(mock_api_key_manager):
    gemini = Gemini(mock_api_key_manager)
    assert gemini.get_model() == gemini.model


# generateメソッドのテスト
@patch('google.generativeai.GenerativeModel')
def test_generate(mock_generative_model, mock_api_key_manager):
    # モックの設定
    mock_response = MagicMock()
    mock_response.text = "Generated response"
    mock_generative_model.return_value.generate_content.return_value = mock_response

    gemini = Gemini(mock_api_key_manager)
    result = gemini.generate("Test message")

    # メソッドが正しく呼び出されたか確認
    mock_generative_model.return_value.generate_content.assert_called_once_with("Test message")
    assert result == "Generated response"


# generate_with_imageメソッドのテスト
@patch('google.generativeai.GenerativeModel')
@patch('PIL.Image.open')
def test_generate_with_image(mock_image_open, mock_generative_model, mock_api_key_manager):
    # モックの設定
    mock_response = MagicMock()
    mock_response.text = "Generated response with image"
    mock_generative_model.return_value.generate_content.return_value = mock_response
    mock_image = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_image

    gemini = Gemini(mock_api_key_manager)
    result = gemini.generate_with_image("Test message with image", "fake_image_path")

    # メソッドが正しく呼び出されたか確認
    mock_image_open.assert_called_once_with("fake_image_path")
    mock_generative_model.return_value.generate_content.assert_called_once_with(
        ["Test message with image", mock_image])
    assert result == "Generated response with image"


# generate_jsonメソッドのテスト
@patch('google.generativeai.GenerativeModel')
def test_generate_json(mock_generative_model, mock_api_key_manager):
    # モックの設定
    mock_response = MagicMock()
    mock_response.text = '{"key": "value"}'
    mock_generative_model.return_value.generate_content.return_value = mock_response

    gemini = Gemini(mock_api_key_manager)
    output_schema = {"key": "str"}
    result = gemini.generate_json("Test JSON message", output_schema)

    # メソッドが正しく呼び出されたか確認
    mock_generative_model.return_value.generate_content.assert_called_once()
    assert isinstance(result, dict)
    assert result == {"key": "value"}


# generate_with_image_jsonメソッドのテスト
@patch('google.generativeai.GenerativeModel')
@patch('PIL.Image.open')
def test_generate_with_image_json(mock_image_open, mock_generative_model, mock_api_key_manager):
    # モックの設定
    mock_response = MagicMock()
    mock_response.text = '{"key": "value with image"}'
    mock_generative_model.return_value.generate_content.return_value = mock_response
    mock_image = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_image

    gemini = Gemini(mock_api_key_manager)
    output_schema = {"key": "str"}
    result = gemini.generate_with_image_json(
        "Test JSON message with image", "fake_image_path", output_schema)

    # メソッドが正しく呼び出されたか確認
    mock_image_open.assert_called_once_with("fake_image_path")
    mock_generative_model.return_value.generate_content.assert_called_once()
    assert isinstance(result, dict)
    assert result == {"key": "value with image"}


# エラーケースのテスト
def test_error_cases(mock_api_key_manager):
    gemini = Gemini(mock_api_key_manager)

    # 無効なJSONレスポンスのテスト
    with patch.object(gemini.model, 'generate_content', return_value=MagicMock(text='Invalid JSON')):
        with pytest.raises(ValueError, match="生成された応答が有効なJSONではありません。"):
            gemini.generate_json("Test message", {"key": "str"})

    # 存在しない画像ファイルのテスト
    with pytest.raises(FileNotFoundError):
        gemini.generate_with_image("Test message", "non_existent_image.jpg")


# 型変換のテスト
def test_type_conversion(mock_api_key_manager):
    gemini = Gemini(mock_api_key_manager)

    # 文字列から整数への変換をテスト
    with patch.object(gemini.model, 'generate_content', return_value=MagicMock(text='{"number": "42"}')):
        result = gemini.generate_json("Test message", {"number": "int"})
        assert result == {"number": 42}

    # 文字列からブール値への変換をテスト
    with patch.object(gemini.model, 'generate_content', return_value=MagicMock(text='{"flag": "true"}')):
        result = gemini.generate_json("Test message", {"flag": "bool"})
        assert result == {"flag": True}
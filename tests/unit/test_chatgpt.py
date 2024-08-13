import pytest
from unittest.mock import Mock, patch, mock_open
from mosaicai.models.chatgpt import ChatGPT
from mosaicai.utils.api_key_manager import APIKeyManager
from openai import OpenAI


@pytest.fixture
def mock_api_key_manager():
    """APIKeyManagerのモックを作成するフィクスチャ"""
    manager = Mock(spec=APIKeyManager)
    manager.get_api_key.return_value = "mock_api_key"
    return manager


@pytest.fixture
def chatgpt_instance(mock_api_key_manager):
    """ChatGPTインスタンスを作成するフィクスチャ"""
    with patch('openai.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create = Mock()
        mock_openai.return_value = mock_client
        return ChatGPT(mock_api_key_manager)


def test_init(mock_api_key_manager):
    """ChatGPTクラスの初期化をテスト"""
    with patch('openai.OpenAI'):
        chatgpt = ChatGPT(mock_api_key_manager)
    assert chatgpt.model == "gpt-4o"
    mock_api_key_manager.get_api_key.assert_called_once_with("openai")


def test_init_no_api_key(mock_api_key_manager):
    """APIキーが設定されていない場合のテスト"""
    mock_api_key_manager.get_api_key.return_value = None
    with pytest.raises(ValueError, match="OpenAI APIキーが設定されていません。"):
        ChatGPT(mock_api_key_manager)


def test_get_model(chatgpt_instance):
    """get_modelメソッドのテスト"""
    assert chatgpt_instance.get_model() == "gpt-4o"


def test_generate(chatgpt_instance):
    """generateメソッドのテスト"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Generated response"))]
    chatgpt_instance.client.chat.completions.create = Mock(return_value=mock_response)

    result = chatgpt_instance.generate("Test message")
    assert result == "Generated response"
    chatgpt_instance.client.chat.completions.create.assert_called_once()


@patch('builtins.open', new_callable=mock_open, read_data=b"image_data")
@patch('base64.b64encode')
def test_generate_with_image(mock_b64encode, mock_file, chatgpt_instance):
    """generate_with_imageメソッドのテスト"""
    mock_b64encode.return_value = b"encoded_image"
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Generated response with image"))]
    chatgpt_instance.client.chat.completions.create = Mock(return_value=mock_response)

    result = chatgpt_instance.generate_with_image(
        "Test message with image", "./tests/test_image.jpg")
    assert result == "Generated response with image"
    chatgpt_instance.client.chat.completions.create.assert_called_once()
    mock_file.assert_called_once_with("./tests/test_image.jpg", "rb")


def test_generate_json(chatgpt_instance):
    """generate_jsonメソッドのテスト"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
    chatgpt_instance.client.chat.completions.create = Mock(return_value=mock_response)

    output_schema = {"key": "str"}
    result = chatgpt_instance.generate_json("Test JSON message", output_schema)
    assert result == {"key": "value"}
    chatgpt_instance.client.chat.completions.create.assert_called_once()


@patch('builtins.open', new_callable=mock_open, read_data=b"image_data")
@patch('base64.b64encode')
def test_generate_with_image_json(mock_b64encode, mock_file, chatgpt_instance):
    """generate_with_image_jsonメソッドのテスト"""
    mock_b64encode.return_value = b"encoded_image"
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"key": "value with image"}'))]
    chatgpt_instance.client.chat.completions.create = Mock(return_value=mock_response)

    output_schema = {"key": "str"}
    result = chatgpt_instance.generate_with_image_json(
        "Test JSON message with image", "./tests/test_image.jpg", output_schema)
    assert result == {"key": "value with image"}
    chatgpt_instance.client.chat.completions.create.assert_called_once()
    mock_file.assert_called_once_with("./tests/test_image.jpg", "rb")

import pytest
from unittest.mock import Mock, patch, mock_open
from mosaicai.models.claude import Claude
from mosaicai.utils.api_key_manager import APIKeyManager
from anthropic import Anthropic


@pytest.fixture
def mock_api_key_manager():
    """APIKeyManagerのモックを作成するフィクスチャ"""
    manager = Mock(spec=APIKeyManager)
    manager.get_api_key.return_value = "mock_api_key"
    return manager


@pytest.fixture
def claude_instance(mock_api_key_manager):
    """Claudeインスタンスを作成するフィクスチャ"""
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.create = Mock()  # createメソッドをモック化
        mock_anthropic.return_value = mock_client
        return Claude(mock_api_key_manager)


def test_init(mock_api_key_manager):
    """Claudeクラスの初期化をテスト"""
    with patch('anthropic.Anthropic'):
        claude = Claude(mock_api_key_manager)
    assert claude.model == "claude-3-5-sonnet-20240620"
    mock_api_key_manager.get_api_key.assert_called_once_with("claude")


def test_init_no_api_key(mock_api_key_manager):
    """APIキーが設定されていない場合のテスト"""
    mock_api_key_manager.get_api_key.return_value = None
    with pytest.raises(ValueError, match="Claude APIキーが設定されていません。"):
        Claude(mock_api_key_manager)


def test_get_model(claude_instance):
    """get_modelメソッドのテスト"""
    assert claude_instance.get_model() == "claude-3-5-sonnet-20240620"


def test_generate(claude_instance):
    """generateメソッドのテスト"""
    mock_response = Mock()
    mock_response.content = [Mock(text="Generated response")]
    claude_instance.client.messages.create = Mock(return_value=mock_response)

    result = claude_instance.generate("Test message")
    assert result == "Generated response"
    claude_instance.client.messages.create.assert_called_once()


@patch('builtins.open', new_callable=mock_open, read_data=b"image_data")
@patch('base64.b64encode')
@patch('mimetypes.guess_type')
def test_generate_with_image(mock_guess_type, mock_b64encode, mock_file, claude_instance):
    """generate_with_imageメソッドのテスト"""
    mock_guess_type.return_value = ("image/jpeg", None)
    mock_b64encode.return_value = b"encoded_image"
    mock_response = Mock()
    mock_response.content = [Mock(text="Generated response with image")]
    claude_instance.client.messages.create = Mock(return_value=mock_response)

    result = claude_instance.generate_with_image(
        "Test message with image", "./tests/test_image.jpg")
    assert result == "Generated response with image"
    claude_instance.client.messages.create.assert_called_once()
    mock_file.assert_called_once_with("./tests/test_image.jpg", "rb")


def test_generate_json(claude_instance):
    """generate_jsonメソッドのテスト"""
    mock_response = Mock()
    mock_response.content = [Mock(text='{"key": "value"}')]
    claude_instance.client.messages.create = Mock(return_value=mock_response)

    output_schema = {"key": "str"}
    result = claude_instance.generate_json("Test JSON message", output_schema)
    assert result == {"key": "value"}
    claude_instance.client.messages.create.assert_called_once()


@patch('builtins.open', new_callable=mock_open, read_data=b"image_data")
@patch('base64.b64encode')
@patch('mimetypes.guess_type')
def test_generate_with_image_json(mock_guess_type, mock_b64encode, mock_file, claude_instance):
    """generate_with_image_jsonメソッドのテスト"""
    mock_guess_type.return_value = ("image/jpeg", None)
    mock_b64encode.return_value = b"encoded_image"
    mock_response = Mock()
    mock_response.content = [Mock(text='{"key": "value with image"}')]
    claude_instance.client.messages.create = Mock(return_value=mock_response)

    output_schema = {"key": "str"}
    result = claude_instance.generate_with_image_json(
        "Test JSON message with image", "./tests/test_image.jpg", output_schema)
    assert result == {"key": "value with image"}
    claude_instance.client.messages.create.assert_called_once()
    mock_file.assert_called_once_with("./tests/test_image.jpg", "rb")

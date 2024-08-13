import pytest
from unittest.mock import patch, MagicMock
from cryptography.fernet import Fernet
from mosaicai.utils.api_key_manager import APIKeyManager


@pytest.fixture
def api_key_manager():
    """APIKeyManagerインスタンスを返すフィクスチャ。テスト前に実行されます。
    環境変数`MOSAICAI_ENCRYPTION_KEY`にダミーの暗号化キーを設定します。
    """
    with patch.dict('os.environ', {'MOSAICAI_ENCRYPTION_KEY': Fernet.generate_key().decode()}):
        return APIKeyManager()


def test_set_and_get_api_key(api_key_manager):
    """APIキーが正しく設定され、取得できることをテストします。"""
    model = "test_model"
    api_key = "test_api_key"

    api_key_manager.set_api_key(model, api_key)
    retrieved_key = api_key_manager.get_api_key(model)

    assert retrieved_key == api_key


def test_get_nonexistent_api_key(api_key_manager):
    """存在しないモデルのAPIキーを取得しようとすると、Noneが返されることをテストします。"""
    nonexistent_model = "nonexistent_model"

    retrieved_key = api_key_manager.get_api_key(nonexistent_model)

    assert retrieved_key is None


@pytest.mark.parametrize("model, env_key", [
    ("openai", "OPENAI_API_KEY"),
    ("claude", "ANTHROPIC_API_KEY"),
    ("gemini", "GOOGLE_API_KEY"),
    ("perplexity", "PERPLEXITY_API_KEY")
])
def test_load_from_env(api_key_manager, model, env_key):
    """環境変数からAPIキーが正しく読み込まれることをテストします。"""
    test_api_key = f"test_{model}_api_key"

    with patch.dict('os.environ', {env_key: test_api_key}):
        api_key_manager.load_from_env()

    retrieved_key = api_key_manager.get_api_key(model)
    assert retrieved_key == test_api_key


def test_load_from_env_missing_key(api_key_manager):
    """環境変数にAPIキーが存在しない場合、APIキーがNoneになることをテストします。"""
    with patch.dict('os.environ', clear=True):
        api_key_manager.load_from_env()

    for model in ["openai", "claude", "gemini", "perplexity"]:
        assert api_key_manager.get_api_key(model) is None

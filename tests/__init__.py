import pytest
import asyncio
from unittest.mock import patch, MagicMock

# 共通のフィクスチャやヘルパー関数をここに定義できます
@pytest.fixture
def mock_api_response():
    return MagicMock()

# 非同期テスト用のフィクスチャ
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

import os
import json
import openai
from PIL import Image
from io import BytesIO
from typing import Dict, List, Union


class MosaicAI:
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def generate_text(self, prompt: str) -> str:
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return response.choices[0].text.strip()

    def generate_with_image(self, prompt: str, image_path: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("プロンプトが空です。有効なプロンプトを入力してください。")
        with open(image_path, "rb") as f:
            image = f.read()
        response = openai.Image.create(
            model=self.model,
            prompt=prompt,
            image=image,
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return response.choices[0].text.strip()

    def generate_json(self, prompt: str, schema: dict) -> dict:
        if not prompt or not prompt.strip():
            raise ValueError("プロンプトが空です。有効なプロンプトを入力してください。")
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            json_mode=True,
            schema=schema,
        )
        return response.choices[0].text.strip()

    def generate_with_image_json(self, prompt: str, image_path: str, schema: dict) -> dict:
        if not prompt or not prompt.strip():
            raise ValueError("プロンプトが空です。有効なプロンプトを入力してください。")
        with open(image_path, "rb") as f:
            image = f.read()
        response = openai.Image.create(
            model=self.model,
            prompt=prompt,
            image=image,
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            json_mode=True,
            schema=schema,
        )
        return response.choices[0].text.strip()


def test_mosaicai():
    client = MosaicAI(model="gpt-4o")
    prompt = "Hello, World!"
    response = client.generate_text(prompt)
    print(response)

    prompt = "This is a test prompt."
    image_path = "test.jpg"
    response = client.generate_with_image(prompt, image_path)
    print(response)

    prompt = "This is a test prompt."
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
        },
    }
    response = client.generate_json(prompt, schema)
    print(json.loads(response))

    prompt = "This is a test prompt."
    image_path = "test.jpg"
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
        },
    }
    response = client.generate_with_image_json(prompt, image_path, schema)
    print(json.loads(response))


if __name__ == "__main__":
    test_mosaicai()

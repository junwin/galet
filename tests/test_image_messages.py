from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from galet.imagegen_dto import ImageInput, ImageMessage, ImageGenResponse
from galet.imagegen_router import ImageGenRouter
from galet.openai_imagegen import OpenAIImageGenApi
from galet.gemini_imagegen import GeminiImageGenApi


def test_openai_references_use_edit_and_preserve_transcript():
    client = Mock()
    client.images.edit.return_value = SimpleNamespace(data=[SimpleNamespace(b64_json='abc')])
    api = OpenAIImageGenApi(client=client)
    image = ImageInput(b'jpeg-data', 'image/jpeg')
    messages = [ImageMessage('user', 'Keep the jacket', (image,)), ImageMessage('assistant', 'Previous version')]
    api.generate_image(model='openai/gpt-image-1', prompt='Change the background', messages=messages)
    client.images.generate.assert_not_called()
    args = client.images.edit.call_args.kwargs
    assert args['model'] == 'gpt-image-1'
    assert args['quality'] == 'medium'
    assert args['image'] == [('reference-0.jpg', b'jpeg-data', 'image/jpeg')]
    assert args['prompt'] == 'user: Keep the jacket\n\nassistant: Previous version\n\nuser: Change the background'


def test_openai_text_only_keeps_generate_endpoint():
    client = Mock()
    client.images.generate.return_value = SimpleNamespace(data=[])
    OpenAIImageGenApi(client=client).generate_image(model='gpt-image-1', prompt='A hill')
    client.images.edit.assert_not_called()
    assert client.images.generate.call_args.kwargs['prompt'] == 'A hill'


def test_unsupported_openai_edit_fails_before_call():
    client = Mock()
    with pytest.raises(ValueError, match='gpt-image'):
        OpenAIImageGenApi(client=client).generate_image(model='dall-e-3', prompt='Edit', messages=[ImageMessage('user', images=(ImageInput(b'x'),))])
    assert not client.mock_calls


def test_router_normalizes_provider_prefix_and_forwards_messages():
    backend = Mock()
    router = ImageGenRouter(openai_api=backend, gemini_api=Mock())
    messages = [ImageMessage('user', 'Reference', (ImageInput(b'x'),))]
    router.generate_image(model='openai/gpt-image-1', prompt='Edit', messages=messages)
    assert backend.generate_image.call_args.kwargs['messages'] is messages
    assert backend.generate_image.call_args.kwargs['model'] == 'gpt-image-1'


def test_gemini_preserves_roles_image_bytes_and_output_mime():
    pytest.importorskip('google.genai')
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b'webp-output', mime_type='image/webp'))]))])
    messages = [ImageMessage('user', 'Reference', (ImageInput(b'jpeg-input', 'image/jpeg'),)), ImageMessage('assistant', 'Previous version')]
    result = GeminiImageGenApi(client=client).generate_image(model='gemini-2.5-flash-image', prompt='Make it brighter', messages=messages)
    contents = client.models.generate_content.call_args.kwargs['contents']
    assert [c.role for c in contents] == ['user', 'model', 'user']
    assert contents[0].parts[1].inline_data.data == b'jpeg-input'
    assert contents[0].parts[1].inline_data.mime_type == 'image/jpeg'
    assert contents[-1].parts[0].text == 'Make it brighter'
    assert result.images[0].mime_type == 'image/webp'


@pytest.mark.parametrize('factory', [lambda: ImageInput(b''), lambda: ImageInput(b'x', 'text/plain'), lambda: ImageMessage('system', 'x'), lambda: ImageMessage('user')])
def test_invalid_inputs_fail(factory):
    with pytest.raises(ValueError):
        factory()

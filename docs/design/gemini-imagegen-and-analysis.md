# Gemini image generation and image analysis

Issue: https://github.com/junwin/galet/issues/2

## Goal

Enable Gemini backends for both:

- Image generation (new `GeminiImageGenApi`).
- Image analysis / vision (complete the existing partial wiring in `GeminiApi`).

## Current state

### Image analysis (vision)

`GeminiApi` already:

- Returns `True` from `supports_image_processing()`.
- Maps the canonical image part shape to a Gemini interactions `ImageContent`
  in `_content_to_parts()`.

The canonical image part shape used across the stack is:

```python
{"type": "image", "source": {"data": "<base64>", "mime_type": "image/png"}}
```

`GeminiApi._content_to_parts()` currently handles only the base64 `source.data`
form. It does not handle image URLs, and it has no direct test coverage.

### Image generation

Not implemented. `ImageGenRouter` only routes OpenAI/DALL-E models to
`OpenAIImageGenApi` (which itself is still a `NotImplementedError` stub). There
is no Gemini image generation backend.

## Scope

### In scope

1. New `GeminiImageGenApi` implementing the `ImageGenApi` protocol.
2. Route Gemini/Imagen image-generation models through `ImageGenRouter`.
3. Complete Gemini image analysis by adding URL support to
   `_content_to_parts()`.
4. Tests for all of the above.

### Out of scope

- Implementing `OpenAIImageGenApi` (still a stub; tracked separately).
- Native Gemini "flash" image generation via `generate_content` (uses the
  dedicated `generate_images` endpoint instead — see below).
- Changing the `ImageResult` DTO (no `mime_type` field today).

## Design

### `GeminiImageGenApi` — new module `src/galet/gemini_imagegen.py`

- Implements `ImageGenApi` (`generate_image(...)`).
- Uses the `google-genai` SDK (v2.19.0), same lazy-import fallback pattern as
  `gemini_api.py`.
- Reuses the Gemini credential resolution via `GeminiApi._build_default_client`
  (DRY; no new credential code).
- Constructor mirrors `GeminiApi`:
  `client`, `max_attempts`, `backoff_base`, `backoff_cap`, `settings`.

#### `generate_image(model, prompt, size="1024x1024", quality="standard", n=1)`

1. Build `google.genai.types.GenerateImagesConfig`:
   - `number_of_images=n`.
   - `aspect_ratio=_size_to_aspect_ratio(size)`.
   - `quality` is accepted for protocol compatibility but not mapped (Imagen has
     no equivalent); documented as ignored.
2. Call `client.models.generate_images(model=model, prompt=prompt, config=config)`.
3. Map `GenerateImagesResponse.generated_images` to `ImageResult`:
   - `image.image_bytes` -> `b64_json` (base64-encoded, no data URI prefix, to
     match the OpenAI convention).
   - `image.gcs_uri` -> `url`.
   - `generated_image.enhanced_prompt` -> `revised_prompt`.
4. Return `ImageGenResponse(images=[...], model=model, raw=response)`.

Retry with exponential backoff on exceptions, reusing `_sleep_backoff` from
`openai_responses.py`.

#### `_size_to_aspect_ratio(size)`

Map OpenAI-style sizes to Imagen aspect ratios:

| size        | aspect_ratio |
|-------------|--------------|
| 1024x1024   | 1:1          |
| 512x512     | 1:1          |
| 256x256     | 1:1          |
| 1792x1024   | 16:9         |
| 1024x1792   | 9:16         |
| 1536x1024   | 3:2          |
| 1024x1536   | 2:3          |
| unknown     | 1:1          |

### `ImageGenRouter` — modify `src/galet/imagegen_router.py`

- Add optional `gemini_api: Optional[GeminiImageGenApi] = None` constructor
  parameter; default to `GeminiImageGenApi()`.
- Route models with `gemini` or `imagen` prefix to the Gemini backend.
- Keep the existing OpenAI/DALL-E routing and the `ValueError` fallback.

### `GeminiApi` — modify `src/galet/gemini_api.py`

- Extend `_content_to_parts()` so an image part also accepts `source.url` or
  `source.uri`, producing the interactions `ImageContent` form
  `{"type": "image", "uri": url, "mime_type": mime_type}`.
- Keep the existing base64 `source.data` handling unchanged.

## SDK reference (google-genai 2.19.0)

- `client.models.generate_images(*, model, prompt, config=None) ->
  GenerateImagesResponse`.
- `GenerateImagesConfig` fields: `number_of_images`, `aspect_ratio`,
  `image_size`, `output_mime_type`, etc.
- `GenerateImagesResponse.generated_images: list[GeneratedImage]`.
- `GeneratedImage` fields: `image` (has `image_bytes`, `gcs_uri`, `mime_type`),
  `enhanced_prompt`.
- Interactions `ImageContent` fields: `type` ("image"), `data`, `uri`,
  `mime_type`.

## Test plan

- `tests/test_gemini_imagegen.py` (new): config mapping (`n`, size->aspect),
  base64 output from `image_bytes`, `url` from `gcs_uri`, `revised_prompt` from
  `enhanced_prompt`, retry behaviour, credential-building delegation.
- `tests/test_imagegen_router.py`: add Gemini/Imagen dispatch cases.
- `tests/test_gemini_api.py`: add `_content_to_parts` cases for base64 and URL
  image parts.

Run with:

```
galet/.venv/bin/python -m pytest tests/ -q
```

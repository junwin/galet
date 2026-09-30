# Image generation with explicit messages

`ImageGenApi.generate_image` keeps its existing `model`, `prompt`, `size`,
`quality`, and `n` arguments and adds optional `messages`.

```python
from galet.imagegen_dto import ImageInput, ImageMessage
from galet.imagegen_router import ImageGenRouter

response = ImageGenRouter().generate_image(
    model="gpt-image-1",
    prompt="Change the background; keep the garments unchanged.",
    messages=[ImageMessage(
        role="user",
        text="Use this reference photograph.",
        images=(ImageInput(data=photo_bytes, mime_type="image/jpeg"),),
    )],
)
```

Callers resolve uploaded IDs, storage, and URLs before constructing `ImageInput`.
Galet receives bytes and a MIME type, never a host path or unresolved UUID.
Only user and assistant turns are supported. No history is retrieved implicitly.
An earlier generated image can be supplied as an assistant turn's image bytes.

Gemini preserves turns as `Content` records (`assistant` becomes `model`) and
sends references as inline image parts. The final prompt is appended as a user
turn. OpenAI's Images API does not accept conversation messages: explicit text
turns are serialized into a labelled transcript, and images are multipart
reference inputs to `images.edit`. Text-only requests use `images.generate`.
OpenAI references require a GPT Image model; unsupported models fail before
calling the provider. This is transcript-based refinement, not a native OpenAI
chat session. Model-specific size, quality, and reference limits still apply.

`ImageResult.mime_type` defaults to PNG for backwards compatibility; Gemini
preserves the returned inline image MIME type. The router strips `openai/` and
`gemini/` provider prefixes before invoking a backend.

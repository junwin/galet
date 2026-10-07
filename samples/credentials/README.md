# Sample credential files

Templates for the provider credential files that galet reads from the directory
named by `GALET_CREDENTIAL_PATH` (or the `--credential-path` flag, or
`Settings(credential_path=...)`).

Copy the templates you need into a private directory outside the repository,
drop the `.example` suffix, and replace the placeholder value with the real
key. Then point galet at that directory:

```bash
export GALET_CREDENTIAL_PATH=/home/myname/credential
```

| Template | Final file name | JSON key |
|---|---|---|
| `oaicred.json.example` | `oaicred.json` | `openai_api_key` |
| `deepseek_cred.json.example` | `deepseek_cred.json` | `deepseek_api_key` |
| `gemini_cred.json.example` | `gemini_cred.json` | `gemini_api_key` (or `api_key`) |
| `mistral_cred.json.example` | `mistral_cred.json` | `mistral_api_key` |

Only the templates (with the `.example` suffix) are tracked by git. The real
files (`oaicred.json`, `*_cred.json`) are ignored, so a copy kept in the
repository will not be committed. Keep the credential directory, and each file
in it, readable only by the account that runs galet.

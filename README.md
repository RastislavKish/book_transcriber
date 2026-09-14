# book_transcriber

Transcribe a directory of scanned book pages (`.png` / `.jpg`) into Markdown
plain text using a vision-capable LLM over an OpenAI-compatible API.

## Usage

```
book_transcriber <INPUT> <OUTPUT> [OPTIONS]

  <INPUT>   directory of page images (.png / .jpg / .jpeg)
  <OUTPUT>  directory for the resulting <stem>.md files

  -b, --batch-size <N>   images per request (default: 1)
  -s, --start <N>        1-indexed page (position in sorted list) to start at
  -n, --count <N>        number of pages to transcribe (default: all remaining)
  -m, --model <NAME>     model to use (a key in [models]); overrides default_model
      --config <PATH>    config file (default: ~/.config/book_transcriber/config.toml)
      --overwrite        re-transcribe pages even if their .md already exists
      --max-retries <N>  retries per request on transient errors (default: 5)
```

- Each image is written to `<OUTPUT>/<stem>.md` (e.g. `12.png` → `12.md`).
- Files whose name begins with a number are sorted naturally (`2.png` before `10.png`).
- **Resume is on by default:** pages whose output already exists are skipped, so
  a re-run continues after an interruption. Use `--overwrite` to force.
- If `<OUTPUT>/prompt` exists, its contents are used as the transcription prompt
  instead of the built-in default.
- With `--batch-size > 1`, the model is asked to separate pages with a marker;
  if it doesn't comply, the raw response is saved to `<a>-<b>.raw.md` (nothing is
  lost) and those pages should be re-run with a smaller batch.
- Transient failures — rate limits (429), overload (429/529), 5xx, and network
  errors — are retried with exponential backoff (full jitter), honoring a
  `Retry-After` header when present. Terminal errors (400/401/404, …) fail
  immediately. Tune with `--max-retries` (0 disables).
- Token usage is reported at the end, with an estimated cost when per-model
  pricing is set in the config.

## Configuration

`~/.config/book_transcriber/config.toml`:

```toml
default_model = "qwen-3.8-27b"

[providers.Cerebras]
base_url = "https://api.cerebras.ai/v1"   # OpenAI-compatible base; no trailing /chat/completions
api_key = "..."

[models."qwen-3.8-27b"]
provider = "Cerebras"
model_id = "qwen-3.8-27b"
# reasoning_effort = "low"        # omit for none
# input_price_per_mtok = 0.10     # optional, USD per 1M tokens, for cost reporting
# output_price_per_mtok = 0.30
```

Add more `[providers.<Name>]` and `[models."<name>"]` tables as needed; pick one
per run with `--model`.

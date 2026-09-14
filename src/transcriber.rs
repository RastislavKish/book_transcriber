use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Value, json};

use crate::config::ResolvedModel;

/// Token usage returned by the provider for one request.
#[derive(Debug, Default, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

pub struct Transcriber {
    client: reqwest::blocking::Client,
}

impl Transcriber {
    pub fn new() -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .context("building HTTP client")?;
        Ok(Self { client })
    }

    /// Send `prompt` plus every image to the model and return the raw text
    /// response together with the reported token usage.
    pub fn transcribe(
        &self,
        model: &ResolvedModel<'_>,
        prompt: &str,
        images: &[&Path],
    ) -> Result<(String, Usage)> {
        let mut content: Vec<Value> = Vec::with_capacity(images.len() + 1);
        content.push(json!({ "type": "text", "text": prompt }));
        for path in images {
            let data_url = encode_image(path)?;
            content.push(json!({
                "type": "image_url",
                "image_url": { "url": data_url },
            }));
        }

        let mut body = json!({
            "model": model.model.model_id,
            "messages": [ { "role": "user", "content": content } ],
        });
        if let Some(effort) = &model.model.reasoning_effort {
            body["reasoning_effort"] = json!(effort);
        }

        let url = format!(
            "{}/chat/completions",
            model.provider.base_url.trim_end_matches('/')
        );

        let response = self
            .client
            .post(&url)
            .bearer_auth(&model.provider.api_key)
            .json(&body)
            .send()
            .with_context(|| format!("sending request to {url}"))?;

        let status = response.status();
        let text = response.text().context("reading response body")?;
        if !status.is_success() {
            bail!("provider returned {status}: {text}");
        }

        let value: Value =
            serde_json::from_str(&text).context("parsing response body as JSON")?;

        let message = value
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow!("response did not contain choices[0].message.content: {text}"))?
            .to_string();

        let usage = Usage {
            prompt_tokens: value
                .pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            completion_tokens: value
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        };

        Ok((clean_output(&message), usage))
    }
}

/// Read an image file and return a `data:` URL with base64 payload.
fn encode_image(path: &Path) -> Result<String> {
    let mime = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        other => bail!("unsupported image extension: {other:?}"),
    };
    let bytes = std::fs::read(path).with_context(|| format!("reading image {}", path.display()))?;
    Ok(format!("data:{mime};base64,{}", BASE64.encode(bytes)))
}

/// Strip reasoning-model artifacts: `<think>` blocks and a single wrapping
/// markdown code fence, so what we write is the transcription itself.
fn clean_output(raw: &str) -> String {
    let mut text = raw.to_string();

    // Remove <think>...</think> blocks (some reasoning models emit them).
    while let Some(start) = text.find("<think>") {
        if let Some(end) = text[start..].find("</think>") {
            let end = start + end + "</think>".len();
            text.replace_range(start..end, "");
        } else {
            break;
        }
    }

    let trimmed = text.trim();

    // Unwrap a single fenced block that spans the whole response, e.g.
    // ```markdown\n...\n``` .
    if trimmed.starts_with("```") {
        if let Some(first_newline) = trimmed.find('\n') {
            let after_fence = &trimmed[first_newline + 1..];
            if let Some(close) = after_fence.rfind("```") {
                return after_fence[..close].trim_end().to_string();
            }
        }
    }

    trimmed.to_string()
}

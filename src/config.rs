use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

/// Top-level configuration, loaded from
/// `~/.config/book_transcriber/config.toml`.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Name of the model (a key in `models`) used when `--model` is not given.
    pub default_model: String,
    /// Providers keyed by name, e.g. `[providers.Cerebras]`.
    #[serde(default)]
    pub providers: HashMap<String, Provider>,
    /// Models keyed by name, e.g. `[models."qwen-3.8-27b"]`.
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
}

#[derive(Debug, Deserialize)]
pub struct Provider {
    /// OpenAI-compatible base URL, e.g. `https://api.cerebras.ai/v1`.
    pub base_url: String,
    pub api_key: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Name of the provider (a key in `providers`).
    pub provider: String,
    /// The provider-side model identifier sent in the request body.
    pub model_id: String,
    /// Optional reasoning effort ("low"/"medium"/"high"). Omit for none.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Upper bound on tokens the model may generate per request.
    #[serde(default = "default_max_completion_tokens")]
    pub max_completion_tokens: u32,
    /// Optional pricing, USD per 1M tokens, used only for cost reporting.
    #[serde(default)]
    pub input_price_per_mtok: Option<f64>,
    #[serde(default)]
    pub output_price_per_mtok: Option<f64>,
}

fn default_max_completion_tokens() -> u32 {
    25_000
}

/// A model together with the provider it resolves to.
pub struct ResolvedModel<'a> {
    pub model: &'a ModelConfig,
    pub provider: &'a Provider,
}

impl Config {
    /// `~/.config/book_transcriber/config.toml`
    pub fn default_path() -> Result<PathBuf> {
        let dir = dirs::config_dir()
            .context("could not determine the user config directory (~/.config)")?;
        Ok(dir.join("book_transcriber").join("config.toml"))
    }

    pub fn load(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        let config: Config = toml::from_str(&text)
            .with_context(|| format!("parsing config file {}", path.display()))?;
        Ok(config)
    }

    /// Resolve a model by name (or the default) together with its provider.
    pub fn resolve<'a>(&'a self, name: Option<&str>) -> Result<ResolvedModel<'a>> {
        let name = name.unwrap_or(&self.default_model);
        let model = self
            .models
            .get(name)
            .ok_or_else(|| anyhow!("model '{name}' is not defined in [models]"))?;
        let provider = self.providers.get(&model.provider).ok_or_else(|| {
            anyhow!(
                "model '{name}' references provider '{}', which is not defined in [providers]",
                model.provider
            )
        })?;
        Ok(ResolvedModel { model, provider })
    }
}

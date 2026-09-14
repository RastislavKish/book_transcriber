mod config;
mod transcriber;

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;

use config::Config;
use transcriber::{Transcriber, Usage};

/// Marker the model is asked to place between pages in a multi-image batch.
const PAGE_BREAK: &str = "<<<--- PAGE BREAK --->>>";

const DEFAULT_PROMPT: &str = "\
You are transcribing scanned pages of a book into clean Markdown plain text. \
Reproduce the text faithfully, preserving reading order, paragraphs, headings, \
lists, and emphasis using Markdown. Do not add commentary, do not summarize, \
and do not wrap your answer in a code fence. Output only the transcription.";

/// Transcribe a directory of scanned book pages into Markdown using a
/// vision-capable LLM.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Directory containing the input images (.png / .jpg / .jpeg).
    input: PathBuf,

    /// Directory where the resulting Markdown files are written.
    /// A file named `prompt` here, if present, is used as the user prompt.
    output: PathBuf,

    /// How many images to send to the model in a single request.
    #[arg(short, long, default_value_t = 1)]
    batch_size: usize,

    /// 1-indexed page (position in the sorted list) to start from.
    #[arg(short, long, default_value_t = 1)]
    start: usize,

    /// Number of pages to transcribe (default: all remaining from --start).
    #[arg(short = 'n', long)]
    count: Option<usize>,

    /// Model name to use (a key in [models]); overrides default_model.
    #[arg(short, long)]
    model: Option<String>,

    /// Path to the config file (default: ~/.config/book_transcriber/config.toml).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Re-transcribe pages even if their output file already exists
    /// (resume/skip is on by default).
    #[arg(long)]
    overwrite: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    if args.batch_size == 0 {
        bail!("--batch-size must be at least 1");
    }
    if args.start == 0 {
        bail!("--start is 1-indexed and must be at least 1");
    }

    let config_path = match &args.config {
        Some(p) => p.clone(),
        None => Config::default_path()?,
    };
    let config = Config::load(&config_path)?;
    let model = config.resolve(args.model.as_deref())?;

    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("creating output directory {}", args.output.display()))?;

    let prompt = load_prompt(&args.output)?;

    // Collect and naturally sort the input images.
    let mut images = list_images(&args.input)?;
    images.sort_by(|a, b| natural_cmp(&file_name(a), &file_name(b)));
    if images.is_empty() {
        bail!("no .png/.jpg/.jpeg images found in {}", args.input.display());
    }

    // Apply --start / --count over the sorted list.
    let start_idx = args.start - 1;
    if start_idx >= images.len() {
        bail!(
            "--start {} is past the last page ({} images available)",
            args.start,
            images.len()
        );
    }
    let end_idx = match args.count {
        Some(n) => (start_idx + n).min(images.len()),
        None => images.len(),
    };
    let selected = &images[start_idx..end_idx];

    // Resolve output paths and, unless --overwrite, drop already-done pages.
    let mut pending: Vec<Page> = Vec::new();
    let mut skipped = 0usize;
    for img in selected {
        let out = output_path_for(&args.output, img);
        if !args.overwrite && out.exists() {
            skipped += 1;
            continue;
        }
        pending.push(Page {
            image: img.clone(),
            output: out,
        });
    }

    let model_name = args.model.as_deref().unwrap_or(&config.default_model);
    println!(
        "Model: {model_name} ({} via {})",
        model.model.model_id, model.model.provider
    );
    println!(
        "Pages {}..={} of {} selected; {} to transcribe, {} already done.",
        args.start,
        end_idx,
        images.len(),
        pending.len(),
        skipped
    );
    if pending.is_empty() {
        println!("Nothing to do.");
        return Ok(());
    }

    let transcriber = Transcriber::new()?;
    let mut total = Usage::default();

    for batch in pending.chunks(args.batch_size) {
        let paths: Vec<&Path> = batch.iter().map(|p| p.image.as_path()).collect();
        let batch_prompt = build_prompt(&prompt, batch.len());

        let label = batch_label(batch);
        println!("Transcribing {label} ...");

        let (text, usage) = transcriber
            .transcribe(&model, &batch_prompt, &paths)
            .with_context(|| format!("transcribing {label}"))?;
        total.prompt_tokens += usage.prompt_tokens;
        total.completion_tokens += usage.completion_tokens;

        write_batch(batch, &text, &args.output)?;
    }

    report_usage(&total, &model);
    Ok(())
}

/// One page: its source image and the Markdown file it maps to.
struct Page {
    image: PathBuf,
    output: PathBuf,
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string()
}

fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("reading input directory {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        let is_image = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
            .unwrap_or(false);
        if path.is_file() && is_image {
            out.push(path);
        }
    }
    Ok(out)
}

fn output_path_for(output_dir: &Path, image: &Path) -> PathBuf {
    let stem = image.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
    output_dir.join(format!("{stem}.md"))
}

fn load_prompt(output_dir: &Path) -> Result<String> {
    let prompt_file = output_dir.join("prompt");
    if prompt_file.exists() {
        let text = std::fs::read_to_string(&prompt_file)
            .with_context(|| format!("reading prompt file {}", prompt_file.display()))?;
        Ok(text.trim().to_string())
    } else {
        Ok(DEFAULT_PROMPT.to_string())
    }
}

/// Augment the base prompt with page-delimiter instructions for batches > 1.
fn build_prompt(base: &str, image_count: usize) -> String {
    if image_count <= 1 {
        base.to_string()
    } else {
        format!(
            "{base}\n\nYou are given {image_count} page images in order. \
Transcribe each one, and separate consecutive pages with a line containing \
exactly:\n{PAGE_BREAK}\nDo not add this marker before the first page or after \
the last one."
        )
    }
}

fn batch_label(batch: &[Page]) -> String {
    match (batch.first(), batch.last()) {
        (Some(first), _) if batch.len() == 1 => file_name(&first.image),
        (Some(first), Some(last)) => {
            format!("{}..{}", file_name(&first.image), file_name(&last.image))
        }
        _ => String::from("(empty)"),
    }
}

/// Split a batch response on the page-break marker and write each page.
fn write_batch(batch: &[Page], text: &str, output_dir: &Path) -> Result<()> {
    if batch.len() == 1 {
        write_page(&batch[0], text)?;
        return Ok(());
    }

    let parts: Vec<&str> = text.split(PAGE_BREAK).map(str::trim).collect();
    if parts.len() == batch.len() {
        for (page, part) in batch.iter().zip(parts) {
            write_page(page, part)?;
        }
    } else {
        // The model didn't delimit as asked. Don't guess a split and risk
        // misaligning pages: dump the raw response so nothing is lost.
        let name = format!(
            "{}-{}.raw.md",
            stem(&batch[0].image),
            stem(&batch[batch.len() - 1].image)
        );
        let raw = output_dir.join(&name);
        std::fs::write(&raw, text)
            .with_context(|| format!("writing {}", raw.display()))?;
        eprintln!(
            "warning: expected {} pages but got {} sections; wrote raw response to {} \
(re-run these pages with a smaller --batch-size)",
            batch.len(),
            parts.len(),
            raw.display()
        );
    }
    Ok(())
}

fn write_page(page: &Page, text: &str) -> Result<()> {
    std::fs::write(&page.output, text)
        .with_context(|| format!("writing {}", page.output.display()))?;
    println!("  wrote {}", page.output.display());
    Ok(())
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("page")
        .to_string()
}

fn report_usage(total: &Usage, model: &config::ResolvedModel<'_>) {
    println!("\nToken usage:");
    println!("  prompt (input):     {}", total.prompt_tokens);
    println!("  completion (output):{}", total.completion_tokens);
    println!(
        "  total:              {}",
        total.prompt_tokens + total.completion_tokens
    );

    let in_price = model.model.input_price_per_mtok;
    let out_price = model.model.output_price_per_mtok;
    if in_price.is_some() || out_price.is_some() {
        let cost = total.prompt_tokens as f64 / 1e6 * in_price.unwrap_or(0.0)
            + total.completion_tokens as f64 / 1e6 * out_price.unwrap_or(0.0);
        println!("  estimated cost:     ${cost:.4}");
    }
}

/// Compare two strings so that embedded numbers order numerically:
/// `2.png` sorts before `10.png`.
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (mut i, mut j) = (0usize, 0usize);

    while i < a.len() && j < b.len() {
        if a[i].is_ascii_digit() && b[j].is_ascii_digit() {
            let si = i;
            while i < a.len() && a[i].is_ascii_digit() {
                i += 1;
            }
            let sj = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            // Compare the digit runs, ignoring leading zeros.
            let na: String = a[si..i].iter().collect();
            let nb: String = b[sj..j].iter().collect();
            let ta = na.trim_start_matches('0');
            let tb = nb.trim_start_matches('0');
            let ord = ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb));
            if ord != Ordering::Equal {
                return ord;
            }
            // Equal value: longer run (more leading zeros) sorts first.
            let ord = na.len().cmp(&nb.len());
            if ord != Ordering::Equal {
                return ord;
            }
        } else {
            let ord = a[i].cmp(&b[j]);
            if ord != Ordering::Equal {
                return ord;
            }
            i += 1;
            j += 1;
        }
    }
    a.len().cmp(&b.len())
}

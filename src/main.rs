mod config;
mod pdf;
mod transcriber;

use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::thread;

use anyhow::{Context, Result, bail};
use clap::Parser;

use std::time::Duration;

use config::Config;
use pdf::Pdf;
use transcriber::{RetryConfig, Transcriber, Usage};

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
    /// Input source: a directory of images (.png / .jpg / .jpeg) or a .pdf file.
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

    /// Retries per request on transient errors (rate limits, overload, 5xx,
    /// network failures) before giving up. Uses exponential backoff.
    #[arg(long, default_value_t = 5)]
    max_retries: u32,

    /// Number of requests to run in parallel.
    #[arg(short, long, default_value_t = 4)]
    jobs: usize,

    /// Resolution to render PDF pages at (PDF input only). Lower values hurt
    /// OCR quality; ~200-300 is a good range.
    #[arg(long, default_value_t = 200.0)]
    dpi: f32,
}

/// A temporary directory removed when this guard is dropped.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!("book_transcriber-{}", std::process::id()));
        std::fs::create_dir_all(&path)
            .with_context(|| format!("creating temp directory {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
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

    let prompt = load_prompt(&args.output, config.default_prompt.as_deref())?;

    let model_name = args.model.as_deref().unwrap_or(&config.default_model);
    println!(
        "Model: {model_name} ({} via {})",
        model.model.model_id, model.model.provider
    );

    // Build the list of pages to transcribe from either a PDF or an image dir.
    // For a PDF, rendered page images live in `_tmp`, kept alive until the run
    // finishes.
    let mut _tmp: Option<TempDir> = None;
    let pending = if is_pdf(&args.input) {
        let (pages, tmp) = pdf_pages(&args)?;
        _tmp = Some(tmp);
        pages
    } else if args.input.is_dir() {
        image_dir_pages(&args)?
    } else {
        bail!(
            "input {} is neither a .pdf file nor a directory",
            args.input.display()
        );
    };

    if pending.is_empty() {
        println!("Nothing to do.");
        return Ok(());
    }

    let transcriber = Transcriber::new()?;
    let retry = RetryConfig {
        max_retries: args.max_retries,
        base_delay: Duration::from_secs(2),
        max_delay: Duration::from_secs(60),
    };

    let batches: Vec<&[Page]> = pending.chunks(args.batch_size).collect();
    let total_batches = batches.len();
    let workers = args.jobs.max(1).min(total_batches);
    if workers > 1 {
        println!("Running {workers} requests in parallel.");
    }

    // Shared state for the worker pool.
    let next = AtomicUsize::new(0); // index of the next batch to claim
    let done = AtomicUsize::new(0); // completed batches, for progress display
    let prompt_tokens = AtomicU64::new(0);
    let completion_tokens = AtomicU64::new(0);
    let output = Mutex::new(()); // serializes multi-line console output
    let failures: Mutex<Vec<String>> = Mutex::new(Vec::new());

    thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, AtomicOrdering::Relaxed);
                    if idx >= total_batches {
                        break;
                    }
                    let batch = batches[idx];
                    let paths: Vec<&Path> = batch.iter().map(|p| p.image.as_path()).collect();
                    let batch_prompt = build_prompt(&prompt, batch.len());
                    let label = batch_label(batch);

                    match transcriber.transcribe(&model, &batch_prompt, &paths, retry) {
                        Ok((text, usage)) => {
                            prompt_tokens.fetch_add(usage.prompt_tokens, AtomicOrdering::Relaxed);
                            completion_tokens
                                .fetch_add(usage.completion_tokens, AtomicOrdering::Relaxed);
                            let summary = write_batch(batch, &text, &args.output);
                            let n = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                            let _lock = output.lock().unwrap();
                            match summary {
                                Ok(s) => println!("[{n}/{total_batches}] {label}: {s}"),
                                Err(e) => {
                                    eprintln!("[{n}/{total_batches}] {label}: write failed: {e:#}");
                                    failures.lock().unwrap().push(format!("{label}: {e:#}"));
                                }
                            }
                        }
                        Err(e) => {
                            let _lock = output.lock().unwrap();
                            eprintln!("{label}: failed: {e:#}");
                            failures.lock().unwrap().push(format!("{label}: {e:#}"));
                        }
                    }
                }
            });
        }
    });

    let total = Usage {
        prompt_tokens: prompt_tokens.into_inner(),
        completion_tokens: completion_tokens.into_inner(),
    };
    report_usage(&total, &model);

    let failures = failures.into_inner().unwrap();
    if !failures.is_empty() {
        eprintln!("\n{} batch(es) failed:", failures.len());
        for f in &failures {
            eprintln!("  {f}");
        }
        eprintln!("Re-run to retry the failed pages (already-done pages are skipped).");
        bail!("{} batch(es) failed", failures.len());
    }
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

fn is_pdf(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("pdf"))
            .unwrap_or(false)
}

/// Selected, not-yet-done pages from a directory of images (natural order).
fn image_dir_pages(args: &Args) -> Result<Vec<Page>> {
    let mut images = list_images(&args.input)?;
    images.sort_by(|a, b| natural_cmp(&file_name(a), &file_name(b)));
    if images.is_empty() {
        bail!("no .png/.jpg/.jpeg images found in {}", args.input.display());
    }

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

    let mut pending = Vec::new();
    let mut skipped = 0usize;
    for img in &images[start_idx..end_idx] {
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

    println!(
        "Pages {}..={} of {} selected; {} to transcribe, {} already done.",
        args.start,
        end_idx,
        images.len(),
        pending.len(),
        skipped
    );
    Ok(pending)
}

/// Selected, not-yet-done pages from a PDF. Each pending page is rendered to a
/// PNG in a temp directory (returned so it outlives transcription); output
/// files are named by page number (e.g. `3.md`).
fn pdf_pages(args: &Args) -> Result<(Vec<Page>, TempDir)> {
    let doc = Pdf::open(&args.input)?;
    let total = doc.page_count()? as usize;
    if total == 0 {
        bail!("PDF {} has no pages", args.input.display());
    }

    let start_idx = args.start - 1;
    if start_idx >= total {
        bail!(
            "--start {} is past the last page ({total} pages in the PDF)",
            args.start
        );
    }
    let end = match args.count {
        Some(n) => (start_idx + n).min(total),
        None => total,
    };
    println!(
        "PDF: {total} pages; rendering pages {}..={end} at {:.0} DPI.",
        args.start, args.dpi
    );

    let tmp = TempDir::new()?;
    let mut pending = Vec::new();
    let mut skipped = 0usize;
    for page in (args.start)..=end {
        // `page` is the 1-based page number the user sees.
        let name = page.to_string();
        let output = args.output.join(format!("{name}.md"));
        if !args.overwrite && output.exists() {
            skipped += 1;
            continue;
        }
        let png = doc.render_page_png((page - 1) as i32, args.dpi)?;
        let image = tmp.path.join(format!("{name}.png"));
        std::fs::write(&image, png).with_context(|| format!("writing {}", image.display()))?;
        pending.push(Page { image, output });
    }

    println!(
        "Pages {}..={end} selected; {} to transcribe, {skipped} already done.",
        args.start,
        pending.len()
    );
    Ok((pending, tmp))
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

/// Resolve the prompt: an output-directory `prompt` file overrides everything,
/// then the config's `default_prompt`, then the built-in default.
fn load_prompt(output_dir: &Path, config_default: Option<&str>) -> Result<String> {
    let prompt_file = output_dir.join("prompt");
    if prompt_file.exists() {
        let text = std::fs::read_to_string(&prompt_file)
            .with_context(|| format!("reading prompt file {}", prompt_file.display()))?;
        Ok(text.trim().to_string())
    } else {
        Ok(config_default.unwrap_or(DEFAULT_PROMPT).to_string())
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

/// Split a batch response on the page-break marker, write each page, and
/// return a short human-readable summary of what was written.
fn write_batch(batch: &[Page], text: &str, output_dir: &Path) -> Result<String> {
    if batch.len() == 1 {
        write_page(&batch[0], text)?;
        return Ok(format!("wrote {}", file_name(&batch[0].output)));
    }

    let parts: Vec<&str> = text.split(PAGE_BREAK).map(str::trim).collect();
    if parts.len() == batch.len() {
        for (page, part) in batch.iter().zip(&parts) {
            write_page(page, part)?;
        }
        let names: Vec<String> = batch.iter().map(|p| file_name(&p.output)).collect();
        Ok(format!("wrote {}", names.join(", ")))
    } else {
        // The model didn't delimit as asked. Don't guess a split and risk
        // misaligning pages: dump the raw response so nothing is lost.
        let name = format!(
            "{}-{}.raw.md",
            stem(&batch[0].image),
            stem(&batch[batch.len() - 1].image)
        );
        let raw = output_dir.join(&name);
        std::fs::write(&raw, text).with_context(|| format!("writing {}", raw.display()))?;
        Ok(format!(
            "expected {} pages but got {} sections; wrote raw response to {} \
(re-run these pages with a smaller --batch-size)",
            batch.len(),
            parts.len(),
            name
        ))
    }
}

fn write_page(page: &Page, text: &str) -> Result<()> {
    std::fs::write(&page.output, text)
        .with_context(|| format!("writing {}", page.output.display()))?;
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

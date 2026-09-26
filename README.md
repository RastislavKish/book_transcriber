# book_transcriber

This is a simple program which lets me transcribe a PDF or a directory of PNG / JPG images using an arbitrary large language model. LLMs, including relatively small models often runnable on consumer hardware, have gotten pretty good at creating highly accurate transcriptions of even complex documents, where they can wastly outperform traditional solutions. The user can request the output to be processed in a specific way, meaning the LLM can handle page structure - headings, tables, transcribe math notation, describe diagrams, images and plots, and even understand spatially aligned structures, like the Pascal triangle.

With this program, I'm trying to find out the best workflow for processing books with a screenreader, as well as determine the accuracy, strenghts and limitations of LLMs used for this purpose.

## Disclaimer

This project is 100% coded by Claude. I'm just lightly skimming through the output, but I'm not actively writing code nor steering the architectural decisions, because for the size of the project it's not worth it. The program is doing what I need it to do, and it's doing it really well, that's the important part for me. Anyone else is free to decide their priorities for themselves. As always, the project, by its license, does not come with any warranty, see the license text for more details.

## Usage

### An example config file

Since this is a project using large language models, you first need to configure the providers and models to be used. Create a config.toml in ~/.config/book_transcriber and give it the following content, replacing the services, models and instructions according to your needs:

```
default_model="qwen-3.8-27b"
default_prompt="Hello! Please transcribe these pages to Markdown. Use LaTeX for math expressions and replace any diagrams or images with placeholder alt descriptions, containing the relevant information for the particular image or diagram."

[providers]

[providers.Cerebras]

base_url="https://api.cerebras.ai/v1"
api_key="..."

[models]

[models."qwen-3.8-27b"]

provider="Cerebras"
model_id="qwen-3.8-27b"
```

### Transcription

I like to alias book_transcriber as btr:

```sh
btr book.pdf
```

Transcribes all pages in book.pdf, and puts them into a book.md file.

```sh
btr book.pdf, output_directory
```

Takes book.pdf and saves individual transcription pages into directory output_directory. Any already transcribed pages are skipped.

```sh
btr book_pages, output_directory
```

Reads images from directory book_pages and saves transcriptions into output_directory. If book_pages contains a plain-text file called prompt, this prompt is used for the transcription.

```sh
btr -s 120 -n 10 book.pdf output_directory
```

Transcribes 10 pages starting with page 120 and saves the result into output_directory.

The program also offers other configurable parameters, for example, how many images are given to the model at once, or how many API requests are performed simultaneously. See ```btr --help``` for more information.

## Build

The project should be cross-platform, it uses portable dependencies including statically linked mupdf for rendering PDF files. On Linux, you need Rust and Clang to perform the compilation:

```sh
cargo build --release -q
```

The result will be placed in the target/release directory.

## License

Copyright (C) 2026 Rastislav Kish

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.


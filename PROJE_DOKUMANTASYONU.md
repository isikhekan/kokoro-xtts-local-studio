# Kokoro — Project Documentation

This repository contains the official **Python library** for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), an open-weight text-to-speech (TTS) model, plus examples, tests, and a local **TTS Studio** web UI (`tts_ui.py`) built with Gradio. It also includes a **demo** app for Hugging Face Spaces and a **kokoro.js** subtree for the JavaScript side (independent of the Python UI).

For a feature-focused TTS Studio cheat sheet (voices, XTTS languages, tables), see **[TTS_STUDIO.md](TTS_STUDIO.md)**. This document is the broader project reference.

---

## 1. What the project does

### 1.1 Kokoro (Python package — `kokoro/`)

- **Kokoro-82M** is a compact (~82M parameter), fast, high-quality TTS model aimed primarily at **English**, with preset voices for **American** and **British** accents.
- Weights are **Apache 2.0** licensed. `KModel` downloads `config.json` and checkpoint weights from Hugging Face (default repo `hexgrad/Kokoro-82M`) on first use.
- Two main types you use in code:
  - **`KPipeline`**: Text processing, grapheme-to-phoneme (G2P) via **[misaki](https://github.com/hexgrad/misaki)**, and voice/style packs. `lang_code` selects the processing line (in TTS Studio, American voices use pipeline `'a'` and British voices use `'b'`).
  - **`KModel`**: Phoneme sequence → waveform. One shared `KModel` instance can serve multiple pipelines to save memory.
- **`misaki[en]`** is required for English G2P. The main [README.md](README.md) documents extra `misaki` extras for other languages (Japanese, Chinese, etc.).

### 1.2 TTS Studio (`tts_ui.py`)

A **Gradio** app you run locally and open in a browser.

- **Two tabs**
  1. **Kokoro** — Fast English TTS: speed, pitch, optional “creativity” (light noise on the style vector), primary/secondary voice **blend**.
  2. **XTTS-v2** — Coqui’s multilingual model: **built-in speakers** or **voice cloning** from uploaded or recorded reference audio (17 UI languages including Turkish `tr`).
- **UI language**: Toggle **English** / **Türkçe** at the top; labels and errors follow the selection.
- **Output**: WAV **download** plus **in-browser preview** (autoplay after generation).
- **Device**: Uses **CUDA** when PyTorch reports a GPU; otherwise **CPU**. Shown in the settings row.
- **Environment**: At startup, `tts_ui.py` sets `TORCHAUDIO_USE_TORCHCODEC=0` to avoid **torchcodec / FFmpeg** issues on some installs (see [TTS_STUDIO.md](TTS_STUDIO.md) known issues).

**Where files are written:** synthesized WAV paths are built with `tempfile.gettempdir()` (OS temp directory), not the project folder. Use **Download** in the UI to keep a copy elsewhere.

### 1.3 Other parts of the repo

| Path | Role |
|------|------|
| `demo/` | Gradio demo for Hugging Face Spaces (`app.py`); may use `spaces` GPU decorators. **Not** the same as TTS Studio. See `demo/requirements.txt` for that stack. |
| `examples/` | Example scripts and snippets. |
| `tests/` | Unit tests (e.g. `test_custom_stft.py`). |
| `kokoro.js/` | JavaScript-related code; follow that folder’s own README / tooling (`node_modules/` is gitignored there). |

---

## 2. Repository layout

```
kokoro/
├── kokoro/                 # Python package (KModel, KPipeline, …)
├── demo/                   # HF Spaces-oriented demo
├── examples/
├── tests/
├── kokoro.js/
├── tts_ui.py               # TTS Studio (Gradio)
├── pyproject.toml          # Package metadata + core kokoro dependencies
├── requirements.txt        # Editable install + TTS Studio + XTTS (coqui-tts)
├── uv.lock                 # Optional: reproducible installs with uv
├── README.md               # Upstream usage, Colab, espeak-ng, MPS notes
├── PROJE_DOKUMANTASYONU.md # This file
└── TTS_STUDIO.md           # TTS Studio feature summary
```

---

## 3. Requirements

| Topic | Detail |
|--------|--------|
| **Python** | `pyproject.toml`: **≥ 3.10 and &lt; 3.14**. **3.11–3.12** is the smoothest choice today. |
| **PyTorch** | ≥ 2.0. Default `pip` install is often CPU wheels; use CUDA wheels from [pytorch.org](https://pytorch.org) for NVIDIA GPUs. |
| **Disk** | Kokoro weights land in the Hugging Face cache. **XTTS-v2** adds roughly **~2 GB** on first synthesis if `coqui-tts` is installed. |
| **RAM / VRAM** | Kokoro is comfortable on CPU. XTTS benefits strongly from a GPU (e.g. **4+ GB VRAM**). |
| **espeak-ng** (optional) | Improves G2P for out-of-vocabulary words on some setups. Windows: MSI from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases); see [README.md](README.md). |

---

## 4. Installation

### 4.1 Virtual environment

**Windows (PowerShell)**

```powershell
cd path\to\kokoro
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux**

```bash
cd path/to/kokoro
python -m venv venv
source venv/bin/activate
```

### 4.2 Dependencies (`requirements.txt`)

```bash
pip install -U pip
pip install -r requirements.txt
```

This:

- Installs the **local** `kokoro` package in **editable** mode (`-e .`) and pulls **core** dependencies from `pyproject.toml`: `torch`, `numpy`, `transformers`, `huggingface_hub`, `loguru`, `misaki[en]>=0.9.4`.
- Installs **TTS Studio** extras: `gradio`, `scipy`, `soundfile`.
- Installs **`coqui-tts`** for the **XTTS-v2** tab. If you only need Kokoro, **comment out** the `coqui-tts` line in `requirements.txt` before installing to avoid the large dependency tree and model download.

**Alternative — uv:** If you use [uv](https://github.com/astral-sh/uv), you can sync from `uv.lock` / `pyproject.toml` per your team’s workflow; `requirements.txt` remains the pip-oriented path documented here.

### 4.3 CUDA (NVIDIA)

Install a CUDA-enabled `torch` **first** (pick the index URL that matches your CUDA version from the official site), then run `pip install -r requirements.txt` so the rest of the stack lines up. Example (verify current wheel URL):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4.4 Apple Silicon (optional GPU path)

For M-series Macs, upstream README suggests `PYTORCH_ENABLE_MPS_FALLBACK=1` when you want MPS-related behavior. TTS Studio does not set this for you; export it when running your script if needed.

---

## 5. Running TTS Studio

```bash
python tts_ui.py
```

Open **http://127.0.0.1:7860** (or **http://localhost:7860**).

**Network exposure:** The app launches with `server_name="0.0.0.0"`, so other machines on the **same network** can reach it if your firewall allows it. For a machine-only server, change `server_name` to `"127.0.0.1"` in `tts_ui.py` before running.

### 5.1 Kokoro tab

- **Speed** 0.5×–2.0×, **pitch** ±6 semitones (duration preserved via resampling), **creativity** slider on the style vector.
- **Primary / secondary voice** and **blend** ratio.
- Output: WAV in the system temp directory + Gradio file download.

### 5.2 XTTS-v2 tab

- Requires **`coqui-tts`**. Without it, Convert shows a localized error with install instructions.
- **Built-in** mode: pick a preset speaker.
- **Clone** mode: 6–30 s clean single-speaker reference (upload or microphone).

---

## 6. Running tests

From the repo root with your environment activated:

```bash
pip install pytest   # if not already present
pytest tests/
```

---

## 7. Developer notes

- Version is defined in `kokoro/__init__.py` and should match `pyproject.toml`.
- Editable install (`pip install -e .` or `pip install -r requirements.txt` with `-e .`) is required for `from kokoro import …` when working from a clone.
- **`demo/app.py`** targets Hugging Face Spaces; do not assume it matches `tts_ui.py` feature-for-feature.

---

## 8. Troubleshooting

| Issue | What to try |
|--------|-------------|
| **Python 3.14+** | Upper bound is &lt; 3.14 in `pyproject.toml`. Use **3.11–3.12** until upstream widens support. |
| **`misaki` / wheels on bleeding-edge Python** | Older `misaki[en]` pins or build toolchains may be needed; see [README.md](README.md) and [TTS_STUDIO.md](TTS_STUDIO.md). |
| **XTTS errors** | Ensure `coqui-tts` is installed; allow first-run download; check disk space. |
| **Odd English G2P on Windows** | Install **espeak-ng** (see README). |
| **torchcodec / FFmpeg** | Already mitigated in `tts_ui.py` via `TORCHAUDIO_USE_TORCHCODEC=0`. |

**Model cache:** Hugging Face assets usually live under `~/.cache/huggingface/` (or `%USERPROFILE%\.cache\huggingface\` on Windows) unless you override `HF_HOME`.

---

## 9. License

**Apache 2.0** — see [LICENSE](LICENSE).

---

## 10. Related docs

- **[TTS_STUDIO.md](TTS_STUDIO.md)** — UI features, voice IDs, XTTS language list, known issues.
- **[README.md](README.md)** — Library usage, Colab cells, espeak-ng, conda hints.

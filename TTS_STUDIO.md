# 🎙️ TTS Studio

A local text-to-speech web UI powered by **Kokoro** and **XTTS-v2**, with full voice control, multilingual support, and WAV download — all running on your own machine.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/your-username/kokoro
cd kokoro

# 2. Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# 3. Launch the UI
python tts_ui.py
```

Open your browser at **http://localhost:7860**

---

## Features

### Interface

- **Language selector** at the top — switch between **English** and **Türkçe** at any time
- Two tabs: **🐸 Kokoro** and **🦎 XTTS-v2**
- WAV file download + in-browser audio preview for every generation

---

### 🐸 Kokoro Tab

Lightweight, fast, high-quality English TTS (82M parameters, runs on CPU).

#### Fine-Tuning Controls

| Control | Description |
|---|---|
| **Speed** | 0.5× (slow) → 2.0× (fast) |
| **Pitch** | −6 semitones (deeper) → +6 semitones (higher) — duration preserved |
| **Creativity** | Adds slight random variation to the style vector; each generation sounds a bit different |

#### Voice Selection

| Control | Description |
|---|---|
| **Primary Voice** | 24 preset voices (American 🇺🇸 & British 🇬🇧, male & female) |
| **Secondary Voice** | Optional second voice to blend with |
| **Blend Ratio** | 0 = 100% Primary, 1 = 100% Secondary, 0.5 = equal mix |

#### Available Voices

**🇺🇸 American English — Female**
`af_heart` · `af_bella` · `af_nicole` · `af_aoede` · `af_kore` · `af_sarah` · `af_nova` · `af_sky` · `af_jessica`

**🇺🇸 American English — Male**
`am_michael` · `am_fenrir` · `am_puck` · `am_echo` · `am_eric` · `am_liam`

**🇬🇧 British English — Female**
`bf_emma` · `bf_isabella` · `bf_alice` · `bf_lily`

**🇬🇧 British English — Male**
`bm_george` · `bm_fable` · `bm_lewis` · `bm_daniel`

---

### 🦎 XTTS-v2 Tab

Multilingual voice cloning model by Coqui (~2 GB, downloads on first use).

#### Supported Languages (17)

| Code | Language | Code | Language |
|---|---|---|---|
| `en` | English | `tr` | Turkish |
| `de` | German | `fr` | French |
| `es` | Spanish | `it` | Italian |
| `pt` | Portuguese | `ru` | Russian |
| `zh-cn` | Chinese | `ja` | Japanese |
| `ko` | Korean | `ar` | Arabic |
| `hi` | Hindi | `nl` | Dutch |
| `pl` | Polish | `cs` | Czech |
| `hu` | Hungarian | | |

#### Voice Modes

**Built-in Voice** — Choose from 58 pre-trained speaker identities. No reference audio needed.

**Voice Cloning** — Upload or record 6–30 seconds of clean, single-speaker audio. XTTS will replicate that voice's characteristics in any supported language.

> Tips for best cloning quality:
> - Use a quiet environment with no background noise
> - Speak naturally at a consistent volume
> - 10–20 seconds gives the best results

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 – 3.13 (3.14 has partial support — see Known Issues) |
| PyTorch | ≥ 2.0 |
| Gradio | ≥ 4.0 |

**GPU (optional):** CUDA-compatible GPU with 4+ GB VRAM speeds up XTTS-v2 significantly. Kokoro runs well on CPU.

For CUDA support, replace the torch install:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Optional: XTTS-v2 Installation

XTTS-v2 is not installed by default. To enable it:

```bash
pip install coqui-tts
```

The model weights (~2 GB) are downloaded automatically on first use.

---

## Known Issues

### Python 3.14 Compatibility

Several dependencies do not yet ship pre-built wheels for Python 3.14:

| Package | Issue | Workaround |
|---|---|---|
| `misaki[en]` | Requires `<3.14` | Install `misaki[en]==0.7.4` |
| `tokenizers` | Requires Rust to compile | Use Python 3.10–3.13 |
| `transformers` | `isin_mps_friendly` removed in newer versions | Auto-patched by the app at startup |
| `torchcodec` | FFmpeg DLLs not found | Disabled via `TORCHAUDIO_USE_TORCHCODEC=0` |

For the smoothest experience, use **Python 3.11 or 3.12** with a fresh virtual environment.

### Windows: `espeak-ng` (optional)

Kokoro uses `espeak-ng` as a fallback G2P engine for out-of-vocabulary words. Without it, unknown words are skipped silently. Install from:
https://github.com/espeak-ng/espeak-ng/releases

---

## Project Structure

```
kokoro/
├── kokoro/          # Kokoro TTS Python package
├── demo/            # Original Gradio demo (streaming, GPU)
├── examples/        # Usage examples
├── tests/           # Unit tests
├── tts_ui.py        # ← TTS Studio (this UI)
├── requirements.txt # TTS Studio dependencies
└── TTS_STUDIO.md    # This file
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

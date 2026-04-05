# Kokoro & local TTS Studio

This repository is based on **[hexgrad/kokoro](https://github.com/hexgrad/kokoro)** — the inference library for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — and includes **`tts_ui.py`**, a local **Gradio** web UI for Kokoro plus optional **XTTS-v2** (Coqui). You can still [`pip install kokoro`](https://pypi.org/project/kokoro/) from PyPI; from a clone, use **`requirements.txt`** for the full UI stack.

**Quick UI reference** (voices, XTTS languages, tables): **[TTS_STUDIO.md](TTS_STUDIO.md)**.

> **Kokoro** is an open-weight TTS model with ~82M parameters: strong quality for its size, fast and efficient. Weights are Apache-licensed.

---

## 1. What this project does

### 1.1 Kokoro (Python package — `kokoro/`)

- **Kokoro-82M** targets **English** with **American** and **British** preset voices.
- **Apache 2.0** weights. `KModel` downloads `config.json` and checkpoints from Hugging Face (`hexgrad/Kokoro-82M`) on first use.
- **`KPipeline`**: text, G2P via **[misaki](https://github.com/hexgrad/misaki)**, voice packs. `lang_code` picks the line (TTS Studio: American → pipeline `'a'`, British → `'b'`).
- **`KModel`**: phonemes → waveform; reuse one model across pipelines to save memory.
- **`misaki[en]`** for English G2P. Extra `misaki` extras for other languages are covered under **[Using the Kokoro library](#using-the-kokoro-library-pip--colab)** below.

### 1.2 TTS Studio (`tts_ui.py`)

Run locally, open in a browser.

- **Tabs:** (1) **Kokoro** — speed, pitch, creativity, voice blend. (2) **XTTS-v2** — built-in speakers or voice cloning; 17 UI languages (including Turkish `tr`).
- **UI language:** **English** / **Türkçe**.
- **Output:** WAV download + in-browser preview.
- **Device:** CUDA if available, else CPU (shown in the UI).
- **`tts_ui.py`** sets `TORCHAUDIO_USE_TORCHCODEC=0` to reduce **torchcodec / FFmpeg** friction (see [TTS_STUDIO.md](TTS_STUDIO.md)).

**Output paths** use `tempfile.gettempdir()` — use **Download** in the UI to keep files.

### 1.3 Other paths in the repo

| Path | Role |
|------|------|
| `demo/` | Hugging Face Spaces Gradio demo — not the same as TTS Studio. See `demo/requirements.txt`. |
| `examples/` | Examples. |
| `tests/` | Unit tests. |
| `kokoro.js/` | JS subtree; see that folder’s README. |

---

## 2. Repository layout

```
kokoro/
├── kokoro/           # Python package (KModel, KPipeline, …)
├── demo/
├── examples/
├── tests/
├── kokoro.js/
├── tts_ui.py         # TTS Studio (Gradio)
├── pyproject.toml
├── requirements.txt  # Editable kokoro + UI + optional coqui-tts
├── uv.lock           # Optional: uv
├── README.md         # This file (main documentation)
├── TTS_STUDIO.md     # TTS Studio feature cheat sheet
└── PROJE_DOKUMANTASYONU.md  # Redirect → README.md
```

---

## 3. Requirements

| Topic | Detail |
|--------|--------|
| **Python** | `pyproject.toml`: **≥ 3.10 and &lt; 3.14**. **3.11–3.12** recommended. |
| **PyTorch** | ≥ 2.0. CUDA wheels: [pytorch.org](https://pytorch.org). |
| **Disk** | HF cache for Kokoro; XTTS ~**2 GB** on first use if `coqui-tts` is installed. |
| **RAM / VRAM** | Kokoro on CPU is fine; XTTS benefits from GPU (e.g. **4+ GB VRAM**). |
| **espeak-ng** (optional) | Better OOV G2P. **Windows:** MSI — [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases); details [below](#windows-installation-espeak-ng). |

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

- Editable **`kokoro`** (`-e .`) + core deps from `pyproject.toml`: `torch`, `numpy`, `transformers`, `huggingface_hub`, `loguru`, `misaki[en]>=0.9.4`.
- TTS Studio: `gradio`, `scipy`, `soundfile`.
- **`coqui-tts`** for the XTTS tab — comment that line in `requirements.txt` if you only want Kokoro.

**uv:** you may sync from `uv.lock` / `pyproject.toml` instead.

### 4.3 CUDA (NVIDIA)

Install CUDA-enabled `torch` first, then:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

(Verify the wheel URL for your CUDA version on the PyTorch site.)

### 4.4 Apple Silicon (optional)

Upstream suggests `PYTORCH_ENABLE_MPS_FALLBACK=1` for M-series Macs when using GPU-related paths. TTS Studio does not set this; export it if you run scripts that need it.

---

## 5. Running TTS Studio

```bash
python tts_ui.py
```

Open **http://127.0.0.1:7860**. The app uses `server_name="0.0.0.0"`, so LAN clients may reach it; for localhost-only, set `server_name="127.0.0.1"` in `tts_ui.py`.

### Kokoro tab

Speed 0.5×–2.0×, pitch ±6 semitones, creativity, primary/secondary **blend**. WAV → temp dir + download.

### XTTS-v2 tab

Requires **`coqui-tts`**. Built-in speakers or clone mode (6–30 s clean reference).

---

## 6. Running tests

```bash
pip install pytest   # if needed
pytest tests/
```

---

## 7. Developer notes

- Version: `kokoro/__init__.py` ↔ `pyproject.toml`.
- From a clone: editable install required for `from kokoro import …`.
- **`demo/app.py`** is for Spaces; not feature-identical to `tts_ui.py`.

---

## 8. Troubleshooting

| Issue | What to try |
|--------|-------------|
| **Python 3.14+** | Use **3.11–3.12** until `pyproject.toml` allows newer. |
| **`misaki` / wheels** | See [TTS_STUDIO.md](TTS_STUDIO.md) and pins in this repo. |
| **XTTS** | `coqui-tts` installed; first-run download; disk space. |
| **Windows G2P** | Install **espeak-ng** ([below](#windows-installation-espeak-ng)). |
| **torchcodec** | Mitigated in `tts_ui.py` via `TORCHAUDIO_USE_TORCHCODEC=0`. |

**HF cache:** `~/.cache/huggingface/` or `%USERPROFILE%\.cache\huggingface\` unless `HF_HOME` is set.

---

## 9. License

**Apache 2.0** — see [LICENSE](LICENSE).

---

## 10. Related docs

- **[TTS_STUDIO.md](TTS_STUDIO.md)** — UI details, voices, XTTS languages, known issues.
- **Upstream:** [github.com/hexgrad/kokoro](https://github.com/hexgrad/kokoro)

---

## Using the Kokoro library (pip & Colab)

Below is the classic **pip / Colab** flow from upstream (works without `tts_ui.py`).

### Basic usage (Colab)

[Google Colab](https://colab.research.google.com/). [Samples](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/SAMPLES.md).

```py
!pip install -q kokoro>=0.9.4 soundfile
!apt-get -qq -y install espeak-ng > /dev/null 2>&1
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
pipeline = KPipeline(lang_code='a')
text = '''
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''
generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000)
```

Under the hood: [`misaki`](https://pypi.org/project/misaki/) — https://github.com/hexgrad/misaki

### Advanced usage (Colab)

```py
!pip install -q kokoro>=0.9.4 soundfile
!apt-get -qq -y install espeak-ng > /dev/null 2>&1

from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch

# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='a')

text = '''
The sky above the port was the color of television, tuned to a dead channel.
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters.
'''

generator = pipeline(
    text, voice='af_heart',
    speed=1, split_pattern=r'\n+'
)

for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000)
```

---

### Windows installation (espeak-ng)

1. [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases) → **Latest release**
2. Download the `*.msi` for your arch (e.g. **espeak-ng-…-x64.msi**)
3. Run the installer

Advanced: [espeak-ng Windows guide](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)

---

### macOS Apple Silicon GPU

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python run-your-kokoro-script.py
```

---

### Conda environment

```yaml
name: kokoro
channels:
  - defaults
dependencies:
  - python==3.9
  - libstdcxx~=12.4.0
  - pip:
      - kokoro>=0.3.1
      - soundfile
      - misaki[en]
```

---

### Acknowledgements

- [@yl4579](https://huggingface.co/yl4579) — StyleTTS 2 architecture.
- [@Pendrokar](https://huggingface.co/Pendrokar) — Kokoro in the TTS Spaces Arena.
- Synthetic training data contributors, compute sponsors, and the community.
- Discord: https://discord.gg/QuGxSWBfQy
- **Kokoro** (心) — heart / spirit; also a [Terminator franchise character](https://terminator.fandom.com/wiki/Kokoro) alongside [Misaki](https://github.com/hexgrad/misaki?tab=readme-ov-file#acknowledgements).

<img src="https://static0.gamerantimages.com/wordpress/wp-content/uploads/2024/08/terminator-zero-41-1.jpg" width="400" alt="kokoro" />

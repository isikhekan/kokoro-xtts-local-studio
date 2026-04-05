import os
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"  # disable torchcodec FFmpeg dependency

from kokoro import KModel, KPipeline
import gradio as gr
import numpy as np
import tempfile
import torch

# ===========================================================================
# Shared: device detection
# ===========================================================================
CUDA_AVAILABLE = torch.cuda.is_available()
device = 'cuda' if CUDA_AVAILABLE else 'cpu'

# ===========================================================================
# I18N — all UI strings in English and Turkish
# ===========================================================================
STRINGS = {
    "en": {
        # App
        "title": "# 🎙️ TTS Studio\nConvert text to speech and download as WAV.",
        "settings_hdr": "### ⚙️ Settings",
        "ui_lang_label": "Interface Language",
        "device_label": "Device",
        # Kokoro tab
        "k_tuning_hdr": "### 🎛️ Fine-Tuning",
        "k_speed": "🐢 Speed 🐇",
        "k_pitch": "🔉 Pitch  (− lower / + higher)",
        "k_creativity": "🎲 Creativity  (slight variation each generation)",
        "k_voice_hdr": "### 🎤 Voice Selection",
        "k_voice1": "Primary Voice",
        "k_voice2": "Secondary Voice (blend)",
        "k_voice2_none": "— None —",
        "k_blend": "Blend Ratio  (0 = Primary  /  1 = Secondary)",
        "k_filename": "File name",
        "k_filename_ph": "output",
        "k_text": "Text",
        "k_text_ph": "Enter the text to convert to speech...",
        "k_btn": "Convert",
        "k_file": "Download",
        "k_preview": "Preview",
        # XTTS tab
        "x_info": "> ℹ️ Model downloads on first Convert (~2 GB). Subsequent uses are fast.",
        "x_lang": "Language",
        "x_speed": "🐢 Speed 🐇",
        "x_mode": "Voice Mode",
        "x_mode_choices": [("Built-in Voice", "builtin"), ("Voice Cloning", "clone")],
        "x_builtin": "Select Built-in Voice",
        "x_ref_src": "Reference Audio Source",
        "x_ref_src_choices": [("Upload File", "upload"), ("Record with Microphone", "mic")],
        "x_ref_upload": "Audio File  (6–30 sec, single speaker, clean)",
        "x_ref_mic": "Record with Microphone  (Record → Stop → Convert)",
        "x_filename": "File name",
        "x_filename_ph": "output_xtts",
        "x_text": "Text",
        "x_text_ph": "Enter the text to convert to speech...",
        "x_btn": "Convert",
        "x_file": "Download",
        "x_preview": "Preview",
        # XTTS language names
        "xtts_langs": {
            "tr": "Turkish (tr)", "en": "English (en)", "de": "German (de)",
            "fr": "French (fr)", "es": "Spanish (es)", "it": "Italian (it)",
            "pt": "Portuguese (pt)", "ru": "Russian (ru)", "zh-cn": "Chinese (zh-cn)",
            "ja": "Japanese (ja)", "ko": "Korean (ko)", "ar": "Arabic (ar)",
            "hi": "Hindi (hi)", "nl": "Dutch (nl)", "pl": "Polish (pl)",
            "cs": "Czech (cs)", "hu": "Hungarian (hu)",
        },
        # Error messages
        "err_no_text": "Please enter the text to convert.",
        "err_no_audio": "No audio generated. Check your text.",
        "err_no_speaker": "Please select a built-in voice.",
        "err_no_ref": "Please upload a reference audio file (6–30 seconds, single speaker).",
        "err_no_coqui": "coqui-tts is not installed. Run: pip install coqui-tts",
    },
    "tr": {
        # App
        "title": "# 🎙️ TTS Studio\nMetni sese çevir ve WAV olarak indir.",
        "settings_hdr": "### ⚙️ Ayarlar",
        "ui_lang_label": "Arayüz Dili",
        "device_label": "Donanım",
        # Kokoro tab
        "k_tuning_hdr": "### 🎛️ İnce Ayarlar",
        "k_speed": "🐢 Hız 🐇",
        "k_pitch": "🔉 Perde  (− kalın / + ince)",
        "k_creativity": "🎲 Yaratıcılık  (her üretimde hafif farklı tonlama)",
        "k_voice_hdr": "### 🎤 Ses Seçimi",
        "k_voice1": "Ana Ses",
        "k_voice2": "İkinci Ses (karıştır)",
        "k_voice2_none": "— Yok —",
        "k_blend": "Ses Karışım Oranı  (0 = Ana Ses  /  1 = İkinci Ses)",
        "k_filename": "Dosya adı",
        "k_filename_ph": "output",
        "k_text": "Metin",
        "k_text_ph": "Sese çevrilecek metni buraya girin...",
        "k_btn": "Çevir",
        "k_file": "İndir",
        "k_preview": "Önizleme",
        # XTTS tab
        "x_info": "> ℹ️ Model ilk Çevir butonunda indirilir (~2 GB). Sonraki kullanımlar hızlıdır.",
        "x_lang": "Dil",
        "x_speed": "🐢 Hız 🐇",
        "x_mode": "Ses Modu",
        "x_mode_choices": [("Hazır Ses", "builtin"), ("Ses Klonlama", "clone")],
        "x_builtin": "Hazır Ses Seç",
        "x_ref_src": "Referans Ses Kaynağı",
        "x_ref_src_choices": [("Dosya Yükle", "upload"), ("Mikrofon ile Kaydet", "mic")],
        "x_ref_upload": "Ses Dosyası  (6-30 sn, tek konuşmacı, gürültüsüz)",
        "x_ref_mic": "Mikrofon ile Kaydet  (Kaydet → Dur → Çevir)",
        "x_filename": "Dosya adı",
        "x_filename_ph": "output_xtts",
        "x_text": "Metin",
        "x_text_ph": "Sese çevrilecek metni buraya girin...",
        "x_btn": "Çevir",
        "x_file": "İndir",
        "x_preview": "Önizleme",
        # XTTS language names
        "xtts_langs": {
            "tr": "Türkçe (tr)", "en": "İngilizce (en)", "de": "Almanca (de)",
            "fr": "Fransızca (fr)", "es": "İspanyolca (es)", "it": "İtalyanca (it)",
            "pt": "Portekizce (pt)", "ru": "Rusça (ru)", "zh-cn": "Çince (zh-cn)",
            "ja": "Japonca (ja)", "ko": "Korece (ko)", "ar": "Arapça (ar)",
            "hi": "Hintçe (hi)", "nl": "Felemenkçe (nl)", "pl": "Lehçe (pl)",
            "cs": "Çekçe (cs)", "hu": "Macarca (hu)",
        },
        # Error messages
        "err_no_text": "Lütfen sese çevrilecek metni girin.",
        "err_no_audio": "Ses üretilemedi. Metni kontrol edin.",
        "err_no_speaker": "Lütfen bir hazır ses seçin.",
        "err_no_ref": "Lütfen referans ses dosyası yükleyin (6-30 saniye, tek konuşmacı).",
        "err_no_coqui": "coqui-tts kurulu değil. Terminalde çalıştırın: pip install coqui-tts",
    },
}

# lang_code → display name (used by synthesize_xtts to look up internal code)
XTTS_LANG_CODES = {
    "tr": "tr", "en": "en", "de": "de", "fr": "fr", "es": "es",
    "it": "it", "pt": "pt", "ru": "ru", "zh-cn": "zh-cn", "ja": "ja",
    "ko": "ko", "ar": "ar", "hi": "hi", "nl": "nl", "pl": "pl",
    "cs": "cs", "hu": "hu",
}

UI_LANG_OPTIONS = ["English", "Türkçe"]
DEFAULT_LANG = "en"

def _s(ui_lang_label: str) -> dict:
    return STRINGS["tr" if ui_lang_label == "Türkçe" else "en"]

def _xtts_lang_choices(ui_lang_label: str):
    s = _s(ui_lang_label)
    return list(s["xtts_langs"].values())

def _xtts_lang_code(display_name: str, ui_lang_label: str) -> str:
    """Convert display name back to lang code."""
    s = _s(ui_lang_label)
    for code, name in s["xtts_langs"].items():
        if name == display_name:
            return code
    return "en"

# ===========================================================================
# KOKORO setup
# ===========================================================================
kokoro_model = KModel().to(device).eval()
kokoro_pipelines = {
    'a': KPipeline(lang_code='a', model=False),
    'b': KPipeline(lang_code='b', model=False),
}

VOICES = {
    '🇺🇸 ♀ Heart ❤️':   'af_heart',
    '🇺🇸 ♀ Bella 🔥':   'af_bella',
    '🇺🇸 ♀ Nicole 🎧':  'af_nicole',
    '🇺🇸 ♀ Aoede':       'af_aoede',
    '🇺🇸 ♀ Kore':        'af_kore',
    '🇺🇸 ♀ Sarah':       'af_sarah',
    '🇺🇸 ♀ Nova':        'af_nova',
    '🇺🇸 ♀ Sky':         'af_sky',
    '🇺🇸 ♀ Jessica':     'af_jessica',
    '🇺🇸 ♂ Michael':     'am_michael',
    '🇺🇸 ♂ Fenrir':      'am_fenrir',
    '🇺🇸 ♂ Puck':        'am_puck',
    '🇺🇸 ♂ Echo':        'am_echo',
    '🇺🇸 ♂ Eric':        'am_eric',
    '🇺🇸 ♂ Liam':        'am_liam',
    '🇬🇧 ♀ Emma':        'bf_emma',
    '🇬🇧 ♀ Isabella':    'bf_isabella',
    '🇬🇧 ♀ Alice':       'bf_alice',
    '🇬🇧 ♀ Lily':        'bf_lily',
    '🇬🇧 ♂ George':      'bm_george',
    '🇬🇧 ♂ Fable':       'bm_fable',
    '🇬🇧 ♂ Lewis':       'bm_lewis',
    '🇬🇧 ♂ Daniel':      'bm_daniel',
}

for v in VOICES.values():
    kokoro_pipelines[v[0]].load_voice(v)

# Kokoro helpers
def _blend_voices(pipeline, voice1, voice2, blend):
    pack1 = pipeline.load_voice(voice1)
    if not voice2 or blend == 0.0:
        return pack1
    return (1.0 - blend) * pack1 + blend * pipeline.load_voice(voice2)

def _pitch_shift(audio, semitones):
    if semitones == 0:
        return audio
    from scipy.signal import resample
    factor = 2 ** (semitones / 12)
    shifted = resample(audio, int(round(len(audio) / factor)))
    return resample(shifted, len(audio))

def _add_creativity(pack, amount):
    if amount == 0:
        return pack
    return pack + torch.randn_like(pack) * amount

def _save_wav(path, audio, sr=24000):
    try:
        import soundfile as sf
        sf.write(path, audio, sr)
    except ImportError:
        from scipy.io.wavfile import write as wav_write
        wav_write(path, sr, (audio * 32767).astype(np.int16))

def synthesize_kokoro(filename, text, voice1, voice2, blend, speed, pitch, creativity, ui_lang):
    s = _s(ui_lang)
    if not text or not text.strip():
        raise gr.Error(s["err_no_text"])

    pipeline = kokoro_pipelines[voice1[0]]
    pack = _blend_voices(pipeline, voice1, voice2, blend)
    pack = _add_creativity(pack, creativity)

    chunks = []
    for _, ps, _ in pipeline(text, voice1, speed):
        ref_s = pack[len(ps) - 1].to(device)
        chunks.append(kokoro_model(ps, ref_s, speed).numpy())

    if not chunks:
        raise gr.Error(s["err_no_audio"])

    combined = _pitch_shift(np.concatenate(chunks), pitch)
    safe_name = (filename or "output").strip()
    if not safe_name.lower().endswith(".wav"):
        safe_name += ".wav"
    out_path = os.path.join(tempfile.gettempdir(), safe_name)
    _save_wav(out_path, combined)
    return out_path, (24000, combined)

# ===========================================================================
# XTTS-v2 setup (lazy load)
# ===========================================================================
xtts_model = None

XTTS_SPEAKERS = [
    "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema",
    "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara",
    "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
    "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie",
    "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min",
    "Viktor Eka", "Abrahan Mack", "Adde Michal", "Baldur Sanjin",
    "Craig Gutsy", "Damien Black", "Gilberto Mathias", "Ilkin Urbano",
    "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid",
    "Viktor Menelaos", "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho",
    "Uta Obando", "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger",
    "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick",
    "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María",
    "Rosemarie Olivia", "Ige Behringer", "Filip Traverse", "Damjan Chapman",
    "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı",
    "Ferran Simen", "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski",
]

def _load_xtts(s: dict):
    global xtts_model
    if xtts_model is None:
        try:
            from TTS.api import TTS as CoquiTTS
        except ImportError:
            raise gr.Error(s["err_no_coqui"])
        xtts_model = CoquiTTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=CUDA_AVAILABLE,
        )
    return xtts_model

def synthesize_xtts(filename, text, lang_display, mode, builtin_speaker,
                    ref_upload, ref_mic, speed, ui_lang):
    s = _s(ui_lang)
    ref_audio = ref_upload or ref_mic

    if not text or not text.strip():
        raise gr.Error(s["err_no_text"])

    tts = _load_xtts(s)
    lang_code = _xtts_lang_code(lang_display, ui_lang)

    safe_name = (filename or "output_xtts").strip()
    if not safe_name.lower().endswith(".wav"):
        safe_name += ".wav"
    out_path = os.path.join(tempfile.gettempdir(), safe_name)

    if mode == "builtin":
        if not builtin_speaker:
            raise gr.Error(s["err_no_speaker"])
        tts.tts_to_file(text=text, speaker=builtin_speaker,
                        language=lang_code, speed=speed, file_path=out_path)
    else:
        if not ref_audio:
            raise gr.Error(s["err_no_ref"])
        tts.tts_to_file(text=text, speaker_wav=ref_audio,
                        language=lang_code, speed=speed, file_path=out_path)

    import soundfile as sf
    audio_data, sr = sf.read(out_path)
    return out_path, (sr, audio_data)

# ===========================================================================
# Gradio UI
# ===========================================================================
_s0 = STRINGS[DEFAULT_LANG]
_xtts_lang_list = list(_s0["xtts_langs"].values())

with gr.Blocks(title="TTS Studio", theme=gr.themes.Soft()) as app:

    lang_state = gr.State(value=DEFAULT_LANG)

    # ── Title ────────────────────────────────────────────────────────────────
    title_md = gr.Markdown(_s0["title"])

    # ── Settings (top, outside tabs) ────────────────────────────────────────
    with gr.Group():
        settings_md = gr.Markdown(_s0["settings_hdr"])
        with gr.Row():
            lang_selector = gr.Dropdown(
                choices=UI_LANG_OPTIONS,
                value="English",
                label=_s0["ui_lang_label"],
                scale=1,
            )
            device_info = gr.Textbox(
                value=f"{'CUDA (GPU)' if CUDA_AVAILABLE else 'CPU'}",
                label=_s0["device_label"],
                interactive=False,
                scale=1,
            )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── 🐸 Kokoro ────────────────────────────────────────────────────────
        with gr.Tab("🐸 Kokoro"):
            with gr.Row():
                with gr.Column(scale=3):

                    # Fine-tuning (top)
                    k_tuning_md = gr.Markdown(_s0["k_tuning_hdr"])
                    with gr.Row():
                        k_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                            label=_s0["k_speed"])
                        k_pitch = gr.Slider(-6.0, 6.0, value=0.0, step=0.5,
                                            label=_s0["k_pitch"])
                    k_creativity = gr.Slider(0.0, 0.3, value=0.0, step=0.01,
                                             label=_s0["k_creativity"])

                    # Voice selection
                    k_voice_md = gr.Markdown(_s0["k_voice_hdr"])
                    with gr.Row():
                        k_voice1 = gr.Dropdown(
                            choices=list(VOICES.items()), value="af_heart",
                            label=_s0["k_voice1"])
                        k_voice2 = gr.Dropdown(
                            choices=[(_s0["k_voice2_none"], "")] + list(VOICES.items()),
                            value="", label=_s0["k_voice2"])
                    k_blend = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                        label=_s0["k_blend"])

                    # Text + filename
                    k_filename = gr.Textbox(label=_s0["k_filename"],
                                            placeholder=_s0["k_filename_ph"],
                                            max_lines=1)
                    k_text = gr.Textbox(label=_s0["k_text"],
                                        placeholder=_s0["k_text_ph"], lines=8)

                    k_btn = gr.Button(_s0["k_btn"], variant="primary", size="lg")

                with gr.Column(scale=2):
                    k_file  = gr.File(label=_s0["k_file"])
                    k_audio = gr.Audio(label=_s0["k_preview"],
                                       interactive=False, autoplay=True)

            k_btn.click(
                fn=synthesize_kokoro,
                inputs=[k_filename, k_text, k_voice1, k_voice2, k_blend,
                        k_speed, k_pitch, k_creativity, lang_state],
                outputs=[k_file, k_audio],
            )

        # ── 🦎 XTTS-v2 ───────────────────────────────────────────────────────
        with gr.Tab("🦎 XTTS-v2"):
            x_info_md = gr.Markdown(_s0["x_info"])
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        x_lang = gr.Dropdown(
                            choices=_xtts_lang_list,
                            value=_s0["xtts_langs"]["en"],
                            label=_s0["x_lang"])
                        x_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                            label=_s0["x_speed"])

                    x_mode = gr.Radio(
                        choices=_s0["x_mode_choices"],
                        value="builtin",
                        label=_s0["x_mode"])

                    x_builtin = gr.Dropdown(
                        choices=XTTS_SPEAKERS, value=XTTS_SPEAKERS[0],
                        label=_s0["x_builtin"], visible=True)

                    with gr.Group(visible=False) as x_clone_group:
                        x_ref_src = gr.Radio(
                            choices=_s0["x_ref_src_choices"],
                            value="upload",
                            label=_s0["x_ref_src"])
                        x_ref_upload = gr.Audio(
                            label=_s0["x_ref_upload"],
                            type="filepath", sources=["upload"], visible=True)
                        x_ref_mic = gr.Audio(
                            label=_s0["x_ref_mic"],
                            type="filepath", sources=["microphone"], visible=False)

                    x_filename = gr.Textbox(label=_s0["x_filename"],
                                            placeholder=_s0["x_filename_ph"],
                                            max_lines=1)
                    x_text = gr.Textbox(label=_s0["x_text"],
                                        placeholder=_s0["x_text_ph"], lines=8)

                    x_btn = gr.Button(_s0["x_btn"], variant="primary", size="lg")

                with gr.Column(scale=2):
                    x_file  = gr.File(label=_s0["x_file"])
                    x_audio = gr.Audio(label=_s0["x_preview"],
                                       interactive=False, autoplay=True)

            # Mode visibility toggles
            def _toggle_mode(mode):
                return (gr.update(visible=(mode == "builtin")),
                        gr.update(visible=(mode == "clone")))
            x_mode.change(fn=_toggle_mode, inputs=x_mode,
                          outputs=[x_builtin, x_clone_group])

            def _toggle_ref_src(src):
                return (gr.update(visible=(src == "upload")),
                        gr.update(visible=(src == "mic")))
            x_ref_src.change(fn=_toggle_ref_src, inputs=x_ref_src,
                             outputs=[x_ref_upload, x_ref_mic])

            x_btn.click(
                fn=synthesize_xtts,
                inputs=[x_filename, x_text, x_lang, x_mode, x_builtin,
                        x_ref_upload, x_ref_mic, x_speed, lang_state],
                outputs=[x_file, x_audio],
            )

    # ── Language change handler ───────────────────────────────────────────────
    def change_language(ui_lang_label):
        s = _s(ui_lang_label)
        lang = "tr" if ui_lang_label == "Türkçe" else "en"
        new_xtts_langs = list(s["xtts_langs"].values())
        return (
            # global
            gr.update(value=s["title"]),
            gr.update(value=s["settings_hdr"]),
            gr.update(label=s["ui_lang_label"]),
            gr.update(label=s["device_label"]),
            # kokoro
            gr.update(value=s["k_tuning_hdr"]),
            gr.update(label=s["k_speed"]),
            gr.update(label=s["k_pitch"]),
            gr.update(label=s["k_creativity"]),
            gr.update(value=s["k_voice_hdr"]),
            gr.update(label=s["k_voice1"]),
            gr.update(label=s["k_voice2"],
                      choices=[( s["k_voice2_none"], "")] + list(VOICES.items())),
            gr.update(label=s["k_blend"]),
            gr.update(label=s["k_filename"], placeholder=s["k_filename_ph"]),
            gr.update(label=s["k_text"], placeholder=s["k_text_ph"]),
            gr.update(value=s["k_btn"]),
            gr.update(label=s["k_file"]),
            gr.update(label=s["k_preview"]),
            # xtts
            gr.update(value=s["x_info"]),
            gr.update(label=s["x_lang"], choices=new_xtts_langs,
                      value=s["xtts_langs"]["en"]),
            gr.update(label=s["x_speed"]),
            gr.update(label=s["x_mode"], choices=s["x_mode_choices"]),
            gr.update(label=s["x_builtin"]),
            gr.update(label=s["x_ref_src"], choices=s["x_ref_src_choices"]),
            gr.update(label=s["x_ref_upload"]),
            gr.update(label=s["x_ref_mic"]),
            gr.update(label=s["x_filename"], placeholder=s["x_filename_ph"]),
            gr.update(label=s["x_text"], placeholder=s["x_text_ph"]),
            gr.update(value=s["x_btn"]),
            gr.update(label=s["x_file"]),
            gr.update(label=s["x_preview"]),
            # state
            lang,
        )

    lang_selector.change(
        fn=change_language,
        inputs=[lang_selector],
        outputs=[
            title_md, settings_md, lang_selector, device_info,
            k_tuning_md, k_speed, k_pitch, k_creativity,
            k_voice_md, k_voice1, k_voice2, k_blend,
            k_filename, k_text, k_btn, k_file, k_audio,
            x_info_md, x_lang, x_speed, x_mode,
            x_builtin, x_ref_src, x_ref_upload, x_ref_mic,
            x_filename, x_text, x_btn, x_file, x_audio,
            lang_state,
        ],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)

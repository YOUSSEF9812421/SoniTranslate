# Chatterbox reference voices

Put the voice you want to clone here (wav / mp3 / ogg / m4a / flac).
Every file appears in the TTS selector as `>name (Chatterbox) CBX-TTS`.

- 10-30 seconds of clean speech: one speaker, no music, no echo, no noise.
- Name the file `name.ar.wav` (language code before the extension) to say which
  language it was recorded in. If it differs from the target language,
  `cfg_weight` is set to 0 automatically so the accent is not carried over.
- Only clone your own voice or a voice you have permission to use.

Needs the separate environment (same as NAMAA):

    uv venv /kaggle/working/cb_env --python 3.11
    uv pip install --python /kaggle/working/cb_env/bin/python chatterbox-tts "setuptools<82"

Check it (downloads the model, prints load time and GPU memory):

    /kaggle/working/cb_env/bin/python soni_translate/chatterbox_worker.py --warmup

Optional environment variables: SONITR_CHATTERBOX_PYTHON, SONITR_CHATTERBOX_CFG,
SONITR_CHATTERBOX_EXAGGERATION, SONITR_CHATTERBOX_ALLOW_CPU.

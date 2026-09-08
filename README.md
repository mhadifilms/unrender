# Unrender

Unrender turns a flattened video into editable shots, scenes, dialogue lines,
audio layers, and an OpenTimelineIO timeline. It keeps original media intact
and writes derived media and manifests to a separate run directory.

The core workflow includes:

- Shot detection, video and image-sequence cutting, and proxy creation.
- Narrative scene grouping using image, audio, and dialogue features.
- AudioShake dialogue/music/effects and anonymous speaker-stem separation.
- Transcription with word timing, dialogue-line clips, and per-shot stems.
- Face and voice clustering, reviewed and named by a human.
- Audio analysis, M&E processing, restoration, and non-destructive transforms.
- Editable OTIO timelines and JSON/CSV interchange artifacts.

Face presence and voice identity are separate evidence. Reviewers assign the
same speaker name to the corresponding face and voice groups. Core does not
infer who is speaking from mouth motion or an audio-video model. Anonymous
speaker channels are not assumed to represent the same person throughout a video.

## Install

Use Python 3.10+ with `ffmpeg` and `ffprobe` on `PATH`:

```bash
git clone https://github.com/mhadifilms/unrender.git
cd unrender
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -e '.[dev,detect,timeline]'
```

Install from source. The project has no tagged releases, release assets, or
package publishing workflow. Use a Git commit when you need to reproduce an
installation; `0.0.0` is fixed Python packaging metadata.

Install only the optional backends needed for your workflow:

| Extra | Capability |
| --- | --- |
| `face` | InsightFace face detection and embeddings |
| `voice` | Voice embeddings and clustering |
| `transcription` | WhisperX ASR and word alignment |
| `scenes` | Image features for scene grouping |
| `scripts` | XLSX and DOCX source transcripts |
| `mne` | Audio event classification |
| `audio-analysis` | Optional acoustic measurements |
| `audio-models` | Optional audio analysis providers |
| `flashsr` | Audio bandwidth restoration |
| `timeline` | OpenTimelineIO export |

The reconstruction example below needs the `face`, `voice`, and `scenes` extras
in the activated environment:

```bash
python -m pip install -e '.[face,voice,scenes]'
```

The `voice` and `transcription` extras use different pyannote versions. Install
transcription into a separate environment, then point Unrender at its interpreter.
Run these commands from the repository directory:

```bash
python3 -m venv "$HOME/.venvs/unrender-transcription"
"$HOME/.venvs/unrender-transcription/bin/python" -m pip install -e '.[transcription]'
export UNRENDER_TRANSCRIPTION_PYTHON="$HOME/.venvs/unrender-transcription/bin/python"
```

ASR requests cross that process boundary through JSON files. Models download or
load when their commands first need them. For gated Hugging Face models, accept
the model's access terms and provide an authorized token through `HF_TOKEN`.

AudioShake is the only source and speaker separation backend. Supply its API key
through `AUDIOSHAKE_API_KEY`; separation uploads the selected audio to AudioShake.
Third-party services and model weights have their own terms and are not bundled.
Dependencies retain their own licenses.
In particular, [InsightFace's pretrained models](https://github.com/deepinsight/insightface#license)
are restricted to non-commercial research, and the optional
[RE-USE weights](https://huggingface.co/nvidia/RE-USE) have NVIDIA's noncommercial
license. Review the selected model's terms before using it.

## Reconstruct a video

Create `project.json` using your own media and output paths:

```json
{
  "paths": {
    "master": "/path/to/video.mov",
    "run_dir": "/path/to/work/video"
  },
  "audio_assets": {
    "original": {"full_mix": "/path/to/video.mov"}
  },
  "speakers": {
    "Speaker A": {"aliases": []},
    "Speaker B": {"aliases": []}
  }
}
```

```bash
unrender shots detect -p project.json
unrender audio separate -p project.json --dry-run
unrender audio separate -p project.json
unrender audio transcribe-lines -p project.json
unrender scenes group -p project.json
unrender audio build-clips -p project.json
unrender face build -p project.json
unrender voice build -p project.json
unrender labels template -p project.json
```

Review the face grids and voice samples referenced by `labels.csv`, then fill
its `speaker` column using the configured speaker names. Alternatively,
`unrender labels interactive -p project.json` provides an interactive labeling flow.

```bash
unrender labels apply -p project.json
unrender shots match -p project.json
unrender audio resolve-clips -p project.json
unrender audio map-dialogue -p project.json
unrender timeline build -p project.json
```

`timeline.otio` contains shot clips, scene markers, per-speaker dialogue tracks,
and music/effects layers when available. It references media on disk; use
`--bundle` for a self-contained media bundle. Full-length supplied dialogue can
skip dialogue/music/effects separation with
`unrender audio separate -p project.json --dx-stem /path/to/dialogue.wav`; this
still runs AudioShake speaker separation on the supplied dialogue.

To import an existing source transcript instead of running ASR, use
`unrender audio transcribe-lines -p project.json --transcript /path/to/transcript.csv`.
CSV, TSV, JSON, XLSX, and DOCX inputs are supported. Use `--transcript-fps` for
frame-based timecodes; project configuration can provide `paths.transcript` and
`audio.transcript_fps`. Generic `audio transform` recipes provide audio processing
operations; inspect `--help` for their available controls.

## Develop

The package separates `editorial/` timeline structure, `analysis/identity/`
human-labeled clusters, and `audio/` media processing and transcription.
Dubbing, dialogue replacement, synthesis models, automatic speaker attribution,
experiments, and host integrations are maintained separately.

```bash
ruff check .
black --check .
pytest -q
mypy
```

Tests use synthetic media and mocked service/model boundaries. They do not
require paid API calls or download model weights. Run directories, media, model
weights, credentials, and local configuration stay out of Git.

## License

Unrender's code is available under the [MIT license](LICENSE). Dependencies,
external models, and services retain their own licenses and terms described above.

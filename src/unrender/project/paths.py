from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def from_path(cls, path: Path | str) -> RunPaths:
        return cls(Path(path).expanduser().resolve())

    @property
    def face_db(self) -> Path:
        return self.root / "face_db.json"

    @property
    def voice_db(self) -> Path:
        return self.root / "voice_db.json"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def unmapped_speakers_dir(self) -> Path:
        return self.audio_dir / "unmapped_speakers"

    @property
    def source_stems_dir(self) -> Path:
        return self.audio_dir / "source_stems"

    @property
    def normalized_audio_dir(self) -> Path:
        return self.audio_dir / "normalized_inputs"

    @property
    def source_separation_json(self) -> Path:
        return self.audio_dir / "source_separation.json"

    @property
    def audio_separation_json(self) -> Path:
        return self.audio_dir / "audio_separation.json"

    @property
    def speaker_stems_dir(self) -> Path:
        return self.unmapped_speakers_dir

    @property
    def diarization_turns_json(self) -> Path:
        return self.audio_dir / "diarization_turns.json"

    @property
    def unmapped_speaker_glob(self) -> str:
        return str(self.speaker_stems_dir / "*_speaker_*_stem.wav")

    @property
    def shot_mapped_dir(self) -> Path:
        return self.audio_dir / "mapped"

    @property
    def shot_dx_dir(self) -> Path:
        return self.audio_dir / "shot_dx"

    @property
    def shot_dx_plan_json(self) -> Path:
        return self.root / "shot_dx_plan.json"

    @property
    def merged_dx_stem(self) -> Path:
        return self.source_stems_dir / "merged_DX_stem.wav"

    @property
    def dialogue_lines_json(self) -> Path:
        return self.root / "dialogue_lines.json"

    @property
    def dialogue_lines_csv(self) -> Path:
        return self.root / "dialogue_lines.csv"

    @property
    def voice_clips_dir(self) -> Path:
        return self.audio_dir / "voice_clips"

    @property
    def voice_clips_json(self) -> Path:
        return self.audio_dir / "voice_clips.json"

    @property
    def voice_clips_csv(self) -> Path:
        return self.audio_dir / "voice_clips.csv"

    @property
    def voice_clips_glob(self) -> str:
        return str(self.voice_clips_dir / "**" / "*_stem.wav")

    @property
    def dialogue_mapped_dir(self) -> Path:
        return self.audio_dir / "dialogue_mapped"

    @property
    def dialogue_stem_plan_json(self) -> Path:
        return self.root / "dialogue_stem_plan.json"

    @property
    def dialogue_stem_plan_csv(self) -> Path:
        return self.root / "dialogue_stem_plan.csv"

    @property
    def speaker_db(self) -> Path:
        return self.root / "speaker_db.json"

    @property
    def labels_csv(self) -> Path:
        return self.root / "labels.csv"

    @property
    def face_review_dir(self) -> Path:
        return self.root / "review" / "face"

    @property
    def voice_review_dir(self) -> Path:
        return self.root / "review" / "voice"

    @property
    def audio_analysis_dir(self) -> Path:
        return self.root / "analysis" / "audio"

    @property
    def audio_analysis_json(self) -> Path:
        return self.audio_analysis_dir / "audio_analysis.json"

    @property
    def audio_visualization_dir(self) -> Path:
        return self.root / "review" / "audio_analysis"

    @property
    def derived_audio_dir(self) -> Path:
        return self.root / "exports" / "derived_audio"

    @property
    def shot_matches_json(self) -> Path:
        return self.root / "shot_speaker_matches.json"

    @property
    def shot_matches_csv(self) -> Path:
        return self.root / "shot_speaker_matches.csv"

    @property
    def voice_matches_json(self) -> Path:
        return self.root / "voice_speaker_matches.json"

    @property
    def shot_stem_plan_json(self) -> Path:
        return self.root / "shot_stem_plan.json"

    @property
    def shot_stem_plan_csv(self) -> Path:
        return self.root / "shot_stem_plan.csv"

    @property
    def clip_stem_plan_json(self) -> Path:
        return self.root / "clip_stem_plan.json"

    @property
    def clip_stem_plan_csv(self) -> Path:
        return self.root / "clip_stem_plan.csv"

    @property
    def shot_dialogue_map_json(self) -> Path:
        return self.root / "shot_dialogue_map.json"

    @property
    def shot_dialogue_map_csv(self) -> Path:
        return self.root / "shot_dialogue_map.csv"

    @property
    def mne_dir(self) -> Path:
        return self.audio_dir / "mne"

    @property
    def mne_fx_dir(self) -> Path:
        return self.mne_dir / "fx"

    @property
    def mne_music_dir(self) -> Path:
        return self.mne_dir / "music"

    @property
    def mne_room_tone_dir(self) -> Path:
        return self.mne_dir / "room_tone"

    @property
    def mne_fx_classification_json(self) -> Path:
        return self.mne_dir / "fx_classification.json"

    @property
    def mne_music_cues_json(self) -> Path:
        return self.mne_dir / "music_cues.json"

    @property
    def mne_room_tone_json(self) -> Path:
        return self.mne_dir / "room_tone.json"

    @property
    def media_dir(self) -> Path:
        return self.root / "media"

    @property
    def proxy_manifest_json(self) -> Path:
        return self.media_dir / "proxy_manifest.json"

    @property
    def shots_manifest_json(self) -> Path:
        return self.root / "shots.json"

    @property
    def scenes_manifest_json(self) -> Path:
        return self.root / "scenes.json"

    @property
    def speaker_attribution_json(self) -> Path:
        """Which character owns each anonymous source group, and how it was decided."""
        return self.root / "speaker_attribution.json"

    @property
    def reviewed_speaker_db(self) -> Path:
        return self.root / "speaker_db.reviewed.json"

    @property
    def shots_media_dir(self) -> Path:
        return self.root / "shots"

    @property
    def timeline_otio(self) -> Path:
        return self.root / "timeline.otio"

    @property
    def export_dir(self) -> Path:
        return self.root / "exports"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.face_review_dir.mkdir(parents=True, exist_ok=True)
        self.voice_review_dir.mkdir(parents=True, exist_ok=True)
        self.audio_analysis_dir.mkdir(parents=True, exist_ok=True)
        self.audio_visualization_dir.mkdir(parents=True, exist_ok=True)
        self.derived_audio_dir.mkdir(parents=True, exist_ok=True)
        self.source_stems_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_audio_dir.mkdir(parents=True, exist_ok=True)
        self.unmapped_speakers_dir.mkdir(parents=True, exist_ok=True)
        self.shot_mapped_dir.mkdir(parents=True, exist_ok=True)
        self.voice_clips_dir.mkdir(parents=True, exist_ok=True)
        self.dialogue_mapped_dir.mkdir(parents=True, exist_ok=True)
        self.mne_fx_dir.mkdir(parents=True, exist_ok=True)
        self.mne_music_dir.mkdir(parents=True, exist_ok=True)
        self.mne_room_tone_dir.mkdir(parents=True, exist_ok=True)

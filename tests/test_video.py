from pathlib import Path

import pytest

from unrender.lib import video


@pytest.mark.parametrize("output", ["7127,\n", "\r\n7127\r\n", "7127"])
def test_count_video_frames_accepts_ffprobe_csv_side_data_column(
    monkeypatch: pytest.MonkeyPatch,
    output: str,
) -> None:
    monkeypatch.setattr(video, "run_video_tool", lambda *args, **kwargs: output)

    assert video.count_video_frames(Path("example.mp4")) == 7127


def test_count_video_frames_rejects_unparseable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video, "run_video_tool", lambda *args, **kwargs: "N/A,\n")

    with pytest.raises(RuntimeError, match="could not parse frame count"):
        video.count_video_frames(Path("example.mp4"))

from __future__ import annotations

from unrender.manifests import read_json
from unrender.manifests.store import ManifestStore, with_manifest_version


def test_manifest_store_writes_versioned_dialogue_lines(run_paths) -> None:
    store = ManifestStore(run_paths)

    store.write_dialogue_lines(
        [
            {
                "line_id": "DL_000001",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "cut_start_sec": 0.0,
                "cut_end_sec": 1.0,
                "text": "hello",
            }
        ]
    )

    data = read_json(run_paths.dialogue_lines_json)
    assert data["version"] == "1.0"
    assert store.load_dialogue_lines()[0].line_id == "DL_000001"


def test_with_manifest_version_preserves_existing_version() -> None:
    assert with_manifest_version({"version": "2.0", "items": []}) == {
        "version": "2.0",
        "items": [],
    }
    assert with_manifest_version({"items": []}) == {"version": "1.0", "items": []}

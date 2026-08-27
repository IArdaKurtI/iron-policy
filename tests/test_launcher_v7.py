from __future__ import annotations

import base64
import json
import os
from types import SimpleNamespace

import pytest

import launcher_v7
from run_experiments_v7 import PHASES


def test_training_records_are_discovered_with_progress(monkeypatch, tmp_path) -> None:
    runs_root = tmp_path / "runs_v7"
    run_path = runs_root / "minimal" / "asymmetric" / "smoke" / "seed_10"
    models = run_path / "models"
    logs = run_path / "logs"
    models.mkdir(parents=True)
    logs.mkdir()
    (models / "config_v7.json").write_text(
        json.dumps({"total_timesteps": 16_384}), encoding="utf-8"
    )
    (logs / "generation_metrics.csv").write_text(
        "generation,agent_timesteps,sps\n1,8192,1000\n2,16384,1000\n",
        encoding="utf-8",
    )
    (models / "leo_final_v7.zip").write_bytes(b"leo")
    (models / "t90_final_v7.zip").write_bytes(b"t90")
    monkeypatch.setattr(launcher_v7, "RUNS_ROOT", runs_root)

    records = launcher_v7.discover_training_records()

    assert len(records) == 1
    assert records[0].phase == "smoke"
    assert records[0].seed == 10
    assert records[0].completed_steps == 16_384
    assert records[0].total_steps == 16_384
    assert records[0].status == "Tamamlandı"
    assert records[0].size_bytes > 0
    assert launcher_v7.count_training_records() == 1


def test_delete_rejects_paths_outside_training_folder(monkeypatch, tmp_path) -> None:
    runs_root = tmp_path / "runs_v7"
    runs_root.mkdir()
    monkeypatch.setattr(launcher_v7, "RUNS_ROOT", runs_root)

    with pytest.raises(ValueError, match="Geçersiz eğitim klasörü"):
        launcher_v7.recycle_training_record(tmp_path / "outside" / "seed_10")


@pytest.mark.skipif(os.name != "nt", reason="Windows Recycle Bin command")
def test_delete_safely_encodes_training_paths_with_spaces(monkeypatch, tmp_path) -> None:
    runs_root = tmp_path / "runs with spaces"
    target = runs_root / "seed_10"
    target.mkdir(parents=True)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(launcher_v7, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(launcher_v7.subprocess, "run", fake_run)

    launcher_v7.recycle_training_record(target)

    command = captured["command"]
    assert isinstance(command, list)
    encoded = command[command.index("-EncodedCommand") + 1]
    decoded = base64.b64decode(encoded).decode("utf-16-le")
    assert str(target.resolve()) in decoded


def test_trash_destination_does_not_overwrite_existing_record(tmp_path) -> None:
    (tmp_path / "seed_10").mkdir()
    (tmp_path / "seed_10_1").mkdir()

    assert launcher_v7._unique_trash_target(tmp_path, "seed_10") == tmp_path / "seed_10_2"


def test_preregistered_training_schedule_is_unchanged() -> None:
    assert PHASES == {
        "smoke": (16_384, (10,), 2, 256, 256),
        "behavior": (200_000, (10,), 4, 512, 512),
        "pilot": (1_000_000, (10, 20, 30), 8, 2_048, 1_024),
        "full": (5_000_000, (10, 20, 30, 40, 50), 8, 2_048, 1_024),
    }


def test_shared_source_has_no_personal_absolute_path() -> None:
    project_root = launcher_v7.ROOT
    shared_files = [
        *project_root.glob("*.py"),
        *project_root.glob("*.bat"),
        *project_root.glob("*.vbs"),
        *project_root.glob("*.ps1"),
        *project_root.glob("*.sh"),
        project_root / "README.md",
    ]
    personal_prefix = "C:" + chr(92) + "Users" + chr(92) + "bukre"
    for source in shared_files:
        assert personal_prefix not in source.read_text(encoding="utf-8")


def test_training_and_technical_status_are_kept_separate() -> None:
    launcher = launcher_v7.TankLauncher.__new__(launcher_v7.TankLauncher)
    training_job = object()
    test_job = object()
    launcher.training_job = training_job
    launcher.test_job = test_job

    launcher.state = "test_status"
    assert launcher.viewed_job() is test_job

    launcher.open_status()
    assert launcher.state == "status"
    assert launcher.viewed_job() is training_job


def test_loading_screen_is_shown_before_ai_modules_are_imported() -> None:
    source = (launcher_v7.ROOT / "launcher_v7.py").read_text(encoding="utf-8")
    assert source.index("\nshow_startup_screen()\n") < source.index(
        "from tank_v7.environment import"
    )

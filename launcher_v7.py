#!/usr/bin/env python3
"""Single-window, no-command launcher for Iron Polcy v7."""

from __future__ import annotations

import os
import math
import re
import csv
import json
import base64
import shutil
import signal
import subprocess
import sys
import threading
import ctypes
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import pygame


ROOT = Path(__file__).resolve().parent
APP_ICON = ROOT / "assets" / "tank_v7_icon.png"
RUNS_ROOT = ROOT / "runs_v7"
WINDOW_SIZE = (900, 700)
WORLD_OFFSET = (50, 18)

SUPPORTED_LANGUAGES = {"tr", "en"}


def default_settings_path() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "IronPolcyV7" / "settings.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "IronPolcyV7" / "settings.json"
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "iron-polcy-v7" / "settings.json"


SETTINGS_PATH = default_settings_path()


def load_language(path: Path | None = None) -> str | None:
    target = path or SETTINGS_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return None
    language = payload.get("language") if isinstance(payload, dict) else None
    return str(language) if language in SUPPORTED_LANGUAGES else None


def save_language(language: str, path: Path | None = None) -> bool:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("language must be tr or en")
    target = path or SETTINGS_PATH
    temporary = target.with_suffix(".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps({"language": language}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return True
    except OSError:
        temporary.unlink(missing_ok=True)
        return False


ENGLISH_TEXT = {
    "Yükleniyor": "Loading",
    "Uygulama yükleniyor…": "Loading application…",
    "Hızlı kontrol — 16 bin adım": "Quick check — 16K steps",
    "Davranış eğitimi — 200 bin adım": "Behavior training — 200K steps",
    "Pilot eğitim — 3 × 1 milyon adım": "Pilot training — 3 × 1M steps",
    "Tam eğitim — 5 × 5 milyon adım": "Full training — 5 × 5M steps",
    "Program hazır. Bir seçenek seçin.": "Ready. Choose an option.",
    "Dil seçin": "Choose language",
    "Bu seçim daha sonra sağ üstten değiştirilebilir.": "You can change this later from the top-right corner.",
    "Türkçe": "Türkçe",
    "Tek pencere kontrol merkezi": "Single-window control center",
    "Hazır modelleri izle": "Watch ready models",
    "Son eğittiğim modelleri izle": "Watch my latest trained models",
    "Eğitim seçenekleri": "Training options",
    "Eğitim kayıtları": "Training records",
    "Eğitim durumunu göster": "Show training status",
    "Programı teknik olarak kontrol et": "Run technical checks",
    "Çıkış": "Exit",
    "Henüz eğitim kaydı bulunmuyor.": "No training records yet.",
    "Çalışıyor": "Running",
    "Tamamlandı": "Completed",
    "Yarım kaldı": "Incomplete",
    "Başlatıldı": "Started",
    "Aç": "Open",
    "Sil": "Delete",
    "Önceki": "Previous",
    "Sonraki": "Next",
    "Ana menüye dön": "Return to main menu",
    "Eğitim kaydını sil": "Delete training record",
    "Bu kayıt Geri Dönüşüm Kutusu’na taşınacak.": "This record will be moved to the Recycle Bin.",
    "Evet — kaydı sil": "Yes — delete record",
    "Hayır — vazgeç": "No — cancel",
    "Görüntü açmadan eğitim başlat": "Start training without match rendering",
    "Eğitim sırasında bu pencere açık kalır; başka pencere açılmaz.": "This window stays open during training; no second window appears.",
    "Teknik kontrol durumu": "Technical check status",
    "Eğitim durumu": "Training status",
    "Henüz teknik kontrol başlatılmadı.": "No technical check has been started yet.",
    "Bu kontrol eğitimlerden ve eğitim kayıtlarından bağımsızdır.": "This check is independent of training and training records.",
    "Teknik kontrolü başlat": "Start technical check",
    "Şu anda çalışan bir eğitim yok.": "No training is currently running.",
    "Kayıtları görmek veya silmek için aşağıdaki düğmeyi kullanın.": "Use the button below to view or delete records.",
    "Eğitim kayıtlarını göster": "Show training records",
    "Kayıtlı bir eğitim de bulunamadı.": "No saved training was found either.",
    "Yeni bir eğitim başlatmak için eğitim seçeneklerini açın.": "Open training options to start a new training run.",
    "Eğitim seçeneklerini aç": "Open training options",
    "Çalışıyor…": "Running…",
    "Sonuç hazırlanıyor…": "Preparing results…",
    "Senin isteğinle durduruldu": "Stopped at your request",
    "Tamamlanamadı": "Failed",
    "hazırlanıyor": "preparing",
    "Programın temel parçaları kontrol ediliyor.": "Checking the program's core components.",
    "Bu işlem tankları eğitmez ve eğitim kayıtlarını değiştirmez.": "This does not train tanks or modify training records.",
    "Otomatik kontroller başarıyla geçti.": "Automated checks passed.",
    "Program düzgün çalışıyor.": "The program is working correctly.",
    "Programın teknik kontrolü tamamlanamadı.": "The technical check could not be completed.",
    "Bu mesaj, eğitim kaydı bulunmadığı anlamına gelmez.": "This does not mean that no training record exists.",
    "Uygulamayı yeniden açıp kontrolü tekrar deneyebilirsiniz.": "Reopen the application and try the check again.",
    "Eğitim devam ediyor; ilerleme yukarıda canlı gösteriliyor.": "Training is running; live progress is shown above.",
    "Oluşan dosyalar otomatik olarak eğitim kayıtlarına ekleniyor.": "Generated files are automatically added to training records.",
    "Eğitim senin isteğinle durduruldu.": "Training was stopped at your request.",
    "O ana kadar oluşan dosyalar eğitim kayıtlarında saklandı.": "Files created up to that point were kept in training records.",
    "Eğitim başarıyla tamamlandı.": "Training completed successfully.",
    "Yeni modeller izlenmeye hazır.": "The new models are ready to watch.",
    "Eğitim beklenmedik biçimde durdu.": "Training stopped unexpectedly.",
    "Varsa yarım kalan dosyalar eğitim kayıtlarından görülebilir.": "Any partial files can be viewed in training records.",
    "Son eğitilen modelleri izle": "Watch latest trained models",
    "ESC: ana menü": "ESC: main menu",
    "Çıkış onayı": "Confirm exit",
    "Eğitim şu anda devam ediyor.": "Training is currently running.",
    "Çıkarsanız eğitim durdurulacak ve arkada çalışmayacak.": "If you exit, training will stop and will not continue in the background.",
    "Çıkmak istediğinizden emin misiniz?": "Are you sure you want to exit?",
    "Evet — eğitimi durdur ve çık": "Yes — stop training and exit",
    "Hayır — eğitime devam et": "No — continue training",
    "Program testleri": "Program tests",
    "Modeller yükleniyor…": "Loading models…",
    "Son eğitim": "Latest training",
    "Hazır modeller": "Ready models",
    "Leo kazandı": "Leo won",
    "T-90 kazandı": "T-90 won",
    "Çifte nakavt": "Double knockout",
    "Berabere": "Draw",
    "Maç tamamlandı": "Match completed",
    "Maçtan ana menüye dönüldü.": "Returned to the main menu from the match.",
    "Dil değiştirildi ancak tercih kaydedilemedi.": "Language changed, but the preference could not be saved.",
    "Sayfa {current} / {total}": "Page {current} / {total}",
    "{completed:,} / {total:,} adım • {size}": "{completed:,} / {total:,} steps • {size}",
    "{count} kayıtlı eğitim bulundu; bunlar şu anda aktif değil.": "{count} training records were found; none are currently active.",
    "Adım: {completed:,} / {total:,}": "Steps: {completed:,} / {total:,}",
    "Tur: {completed:,} / {total:,}": "Rounds: {completed:,} / {total:,}",
    "Seed: {seed}   Hız: {speed:,} adım/sn": "Seed: {seed}   Speed: {speed:,} steps/s",
    "{passed} otomatik kontrol başarıyla geçti.": "{passed} automated checks passed.",
    "Maç {match} • adım {step}": "Match {match} • step {step}",
    "{title} arka planda çalışıyor.": "{title} is running in the background.",
    "{title} sonucu hazırlanıyor.": "Preparing the result for {title}.",
    "{title} senin isteğinle durduruldu.": "{title} was stopped at your request.",
    "{title} tamamlandı.": "{title} completed.",
    "{title} tamamlanamadı; açıklama için durum ekranını açın.": "{title} failed; open the status screen for details.",
    "Eğitim klasörü artık mevcut değil. Kayıt listesi yenilendi.": "The training folder no longer exists. The record list was refreshed.",
    "Eğitim klasörü açılamadı: {error}": "The training folder could not be opened: {error}",
    "Devam eden eğitim kaydı silinemez.": "An active training record cannot be deleted.",
    "Seed {seed} kaydı Geri Dönüşüm Kutusu’na taşındı.": "Seed {seed} was moved to the Recycle Bin.",
    "Eğitim kaydı silinemedi: {error}": "The training record could not be deleted: {error}",
    "Önce devam eden işlemin tamamlanmasını bekleyin.": "Wait for the current operation to finish first.",
    "{title} başlatıldı.": "{title} started.",
    "Eğitim başlatılamadı: {error}": "Training could not be started: {error}",
    "Program testleri başlatıldı.": "Program tests started.",
    "Test başlatılamadı: {error}": "The test could not be started: {error}",
    "İzlenecek model bulunamadı. Önce bir eğitim tamamlayın.": "No model is available to watch. Complete a training run first.",
    "Modeller açılamadı: {error}": "Models could not be opened: {error}",
}


def translated(language: str, value: str) -> str:
    return ENGLISH_TEXT.get(value, value) if language == "en" else value


def show_startup_screen() -> None:
    """Show immediate feedback while the heavier AI modules are imported."""
    if __name__ != "__main__":
        return
    if os.name == "nt":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "IronPolcyV7.SpartanLauncher"
            )
        except (AttributeError, OSError):
            pass
    pygame.init()
    if APP_ICON.is_file():
        pygame.display.set_icon(pygame.image.load(str(APP_ICON)))
    language = load_language()
    loading_label = translated(language or "en", "Yükleniyor")
    pygame.display.set_caption(f"Iron Polcy v7 — {loading_label}")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    screen.fill((0, 0, 0))
    title_font = pygame.font.SysFont("Segoe UI Semibold", 42)
    message_font = pygame.font.SysFont("Segoe UI", 23)
    title = title_font.render("IRON POLCY v7", True, (255, 255, 255))
    loading_message = (
        translated(language, "Uygulama yükleniyor…")
        if language is not None
        else "Yükleniyor… / Loading…"
    )
    message = message_font.render(loading_message, True, (255, 255, 255))
    screen.blit(title, title.get_rect(center=(WINDOW_SIZE[0] // 2, 290)))
    screen.blit(message, message.get_rect(center=(WINDOW_SIZE[0] // 2, 350)))
    pygame.display.flip()
    pygame.event.pump()


show_startup_screen()

from tank_v7.environment import PhysicsTankEnvV7
from tank_v7.evaluation import NamedAgent, load_agent


PHASE_LABELS = {
    "smoke": "Hızlı kontrol — 16 bin adım",
    "behavior": "Davranış eğitimi — 200 bin adım",
    "pilot": "Pilot eğitim — 3 × 1 milyon adım",
    "full": "Tam eğitim — 5 × 5 milyon adım",
}
PHASE_DETAILS = {
    "smoke": (16_384, 1, 2, 256, 8),
    "behavior": (200_000, 1, 4, 512, 8),
    "pilot": (1_000_000, 3, 8, 2_048, 8),
    "full": (5_000_000, 5, 8, 2_048, 8),
}
PHASE_SEEDS = {
    "smoke": (10,),
    "behavior": (10,),
    "pilot": (10, 20, 30),
    "full": (10, 20, 30, 40, 50),
}


def latest_model_pair() -> tuple[Path, Path] | None:
    candidates = list((ROOT / "runs_v7").rglob("leo_final_v7.zip"))
    candidates = [path for path in candidates if (path.parent / "t90_final_v7.zip").is_file()]
    if not candidates:
        return None
    leo = max(candidates, key=lambda path: path.stat().st_mtime)
    return leo, leo.parent / "t90_final_v7.zip"


def ready_model_pair() -> tuple[Path, Path] | None:
    leo = ROOT / "models_v7" / "leo_final_v7.zip"
    t90 = ROOT / "models_v7" / "t90_final_v7.zip"
    return (leo, t90) if leo.is_file() and t90.is_file() else None


def console_python() -> Path:
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        candidate = executable.with_name("python.exe")
        if candidate.is_file():
            return candidate
    return executable


def configure_windows_app_identity() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "TankV7.SpartanLauncher"
        )
    except (AttributeError, OSError):
        pass


class BackgroundJob:
    GEN_PATTERN = re.compile(
        r"Gen\s+(\d+)\s+\|\s+steps=\s*([\d,]+)\s+\|\s+SPS=\s*([\d,]+)"
    )
    SEED_PATTERN = re.compile(r"--seed\s+(\d+)")

    def __init__(
        self,
        title: str,
        command: list[str],
        training_details: tuple[int, int, int, int, int] | None = None,
        training_phase: str | None = None,
    ) -> None:
        self.title = title
        self.lines: deque[str] = deque(maxlen=14)
        self.return_code: int | None = None
        self._lock = threading.Lock()
        self.training_details = training_details
        self.training_phase = training_phase
        self.started_at = time.time()
        self._last_disk_poll = 0.0
        self._disk_progress: dict[str, int | float | None] | None = None
        self.completed_runs = 0
        self.current_steps = 0
        self.current_generation = 0
        self.current_seed: int | None = None
        self.sps = 0
        self.stopped_by_user = False
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for raw in self.process.stdout:
            line = raw.strip()
            if line:
                with self._lock:
                    self.lines.append(line)
                    self._parse_progress(line)
        self.return_code = self.process.wait()

    def _parse_progress(self, line: str) -> None:
        seed_match = self.SEED_PATTERN.search(line)
        if seed_match:
            self.current_seed = int(seed_match.group(1))
        generation_match = self.GEN_PATTERN.search(line)
        if generation_match:
            self.current_generation = int(generation_match.group(1))
            self.current_steps = int(generation_match.group(2).replace(",", ""))
            self.sps = int(generation_match.group(3).replace(",", ""))
        if "v7 final models saved" in line and self.training_details is not None:
            self.completed_runs = min(self.completed_runs + 1, self.training_details[1])
            self.current_steps = 0
            self.current_generation = 0

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.lines)

    def progress_snapshot(self) -> dict[str, int | float | None] | None:
        disk_progress = self._progress_from_disk()
        with self._lock:
            if self.training_details is None:
                return None
            steps_per_run, run_count, n_envs, n_steps, n_epochs = self.training_details
            generations_per_run = math.ceil(steps_per_run / (n_envs * n_steps))
            total_steps = steps_per_run * run_count
            completed_steps = min(
                self.completed_runs * steps_per_run + self.current_steps, total_steps
            )
            completed_generations = min(
                self.completed_runs * generations_per_run + self.current_generation,
                generations_per_run * run_count,
            )
            memory_progress: dict[str, int | float | None] = {
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "percent": 100.0 * completed_steps / max(total_steps, 1),
                "completed_generations": completed_generations,
                "total_generations": generations_per_run * run_count,
                "completed_epochs": completed_generations * n_epochs,
                "total_epochs": generations_per_run * run_count * n_epochs,
                "seed": self.current_seed,
                "sps": self.sps,
            }
        if (
            disk_progress is not None
            and int(disk_progress["completed_steps"] or 0)
            >= int(memory_progress["completed_steps"] or 0)
        ):
            return disk_progress
        return memory_progress

    def _progress_from_disk(self) -> dict[str, int | float | None] | None:
        if self.training_details is None or self.training_phase is None:
            return None
        now = time.monotonic()
        if now - self._last_disk_poll < 0.5:
            return self._disk_progress
        self._last_disk_poll = now
        steps_per_run, run_count, n_envs, n_steps, n_epochs = self.training_details
        generations_per_run = math.ceil(steps_per_run / (n_envs * n_steps))
        completed_steps = 0
        completed_generations = 0
        active_seed: int | None = None
        active_sps = 0
        newest_update = 0.0
        for seed in PHASE_SEEDS[self.training_phase]:
            metrics_path = (
                ROOT
                / "runs_v7"
                / "minimal"
                / "asymmetric"
                / self.training_phase
                / f"seed_{seed}"
                / "logs"
                / "generation_metrics.csv"
            )
            try:
                modified = metrics_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if modified < self.started_at - 2.0:
                continue
            try:
                with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                    last_row = None
                    for row in csv.DictReader(handle):
                        last_row = row
            except (OSError, csv.Error, ValueError):
                continue
            if not last_row:
                continue
            try:
                steps = min(int(last_row["agent_timesteps"]), steps_per_run)
                generation = min(int(last_row["generation"]), generations_per_run)
                sps = int(float(last_row["sps_per_agent"]))
            except (KeyError, TypeError, ValueError):
                continue
            completed_steps += steps
            completed_generations += generation
            if modified >= newest_update:
                newest_update = modified
                active_seed = seed
                active_sps = sps
        if newest_update == 0.0:
            return self._disk_progress
        total_steps = steps_per_run * run_count
        total_generations = generations_per_run * run_count
        self._disk_progress = {
            "completed_steps": min(completed_steps, total_steps),
            "total_steps": total_steps,
            "percent": 100.0 * min(completed_steps, total_steps) / max(total_steps, 1),
            "completed_generations": min(completed_generations, total_generations),
            "total_generations": total_generations,
            "completed_epochs": min(completed_generations, total_generations) * n_epochs,
            "total_epochs": total_generations * n_epochs,
            "seed": active_seed,
            "sps": active_sps,
        }
        return self._disk_progress

    def stop(self) -> None:
        """Stop the orchestrator and every training child process it created."""
        if not self.running:
            return
        self.stopped_by_user = True
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                self.process.kill()
            else:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    action: Callable[[], None]
    enabled: bool = True
    compact: bool = False
    selected: bool = False


@dataclass(frozen=True)
class TrainingRecord:
    path: Path
    phase: str
    seed: int
    completed_steps: int
    total_steps: int
    status: str
    size_bytes: int
    modified: float


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _last_training_row(path: Path) -> dict[str, str] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            last = None
            for row in csv.DictReader(handle):
                last = row
            return last
    except (OSError, csv.Error):
        return None


def discover_training_records() -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    if not RUNS_ROOT.is_dir():
        return records
    for config_path in RUNS_ROOT.glob("*/*/*/seed_*/models/config_v7.json"):
        run_path = config_path.parent.parent
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            total_steps = int(payload.get("total_timesteps", 0))
            phase = run_path.parent.name
            seed = int(run_path.name.removeprefix("seed_"))
            modified = max(
                (item.stat().st_mtime for item in run_path.rglob("*") if item.is_file()),
                default=run_path.stat().st_mtime,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        row = _last_training_row(run_path / "logs" / "generation_metrics.csv")
        try:
            completed_steps = int(row["agent_timesteps"]) if row else 0
        except (KeyError, TypeError, ValueError):
            completed_steps = 0
        final_pair = (
            (run_path / "models" / "leo_final_v7.zip").is_file()
            and (run_path / "models" / "t90_final_v7.zip").is_file()
        )
        status = "Tamamlandı" if final_pair else ("Yarım kaldı" if completed_steps else "Başlatıldı")
        records.append(
            TrainingRecord(
                path=run_path,
                phase=phase,
                seed=seed,
                completed_steps=min(completed_steps, total_steps) if total_steps else completed_steps,
                total_steps=total_steps,
                status=status,
                size_bytes=_directory_size(run_path),
                modified=modified,
            )
        )
    return sorted(records, key=lambda record: record.modified, reverse=True)


def count_training_records() -> int:
    if not RUNS_ROOT.is_dir():
        return 0
    return sum(1 for path in RUNS_ROOT.glob("*/*/*/seed_*") if path.is_dir())


def _unique_trash_target(directory: Path, name: str) -> Path:
    candidate = directory / name
    counter = 1
    while candidate.exists():
        candidate = directory / f"{name}_{counter}"
        counter += 1
    return candidate


def recycle_training_record(path: Path) -> None:
    target = path.resolve()
    runs_root = RUNS_ROOT.resolve()
    if runs_root not in target.parents or not target.name.startswith("seed_"):
        raise ValueError("Geçersiz eğitim klasörü")
    if not target.is_dir():
        raise FileNotFoundError(target)
    if os.name == "nt":
        escaped_target = str(target).replace("'", "''")
        script = (
            "Add-Type -AssemblyName Microsoft.VisualBasic; "
            "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory("
            f"'{escaped_target}',"
            "[Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,"
            "[Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)"
        )
        encoded_script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded_script,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout.strip() or "Eğitim kaydı silinemedi")
        return

    if sys.platform == "darwin":
        trash_files = Path.home() / ".Trash"
        trash_files.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(_unique_trash_target(trash_files, target.name)))
        return

    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    trash_root = data_home / "Trash"
    trash_files = trash_root / "files"
    trash_info = trash_root / "info"
    trash_files.mkdir(parents=True, exist_ok=True)
    trash_info.mkdir(parents=True, exist_ok=True)
    destination = _unique_trash_target(trash_files, target.name)
    info_path = trash_info / f"{destination.name}.trashinfo"
    info_path.write_text(
        "[Trash Info]\n"
        f"Path={quote(str(target), safe='/')}\n"
        f"DeletionDate={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\n",
        encoding="utf-8",
    )
    try:
        shutil.move(str(target), str(destination))
    except Exception:
        info_path.unlink(missing_ok=True)
        raise


class TankLauncher:
    BG = (0, 0, 0)
    PANEL = (8, 8, 8)
    BUTTON = (16, 16, 16)
    BUTTON_HOVER = (32, 32, 32)
    DISABLED = (10, 10, 10)
    TEXT = (255, 255, 255)
    MUTED = (255, 255, 255)
    BLUE = (255, 255, 255)
    GREEN = (73, 190, 120)
    RED = (225, 91, 82)

    def __init__(self) -> None:
        configure_windows_app_identity()
        pygame.init()
        if APP_ICON.is_file():
            pygame.display.set_icon(pygame.image.load(str(APP_ICON)))
        pygame.display.set_caption("Iron Polcy v7 — Spartan")
        existing_screen = pygame.display.get_surface()
        self.screen = (
            existing_screen
            if existing_screen is not None and existing_screen.get_size() == WINDOW_SIZE
            else pygame.display.set_mode(WINDOW_SIZE)
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 24)
        self.small = pygame.font.SysFont("Segoe UI", 18)
        self.tiny = pygame.font.SysFont("Consolas", 15)
        self.title_font = pygame.font.SysFont("Segoe UI Semibold", 38)
        stored_language = load_language()
        self.language = stored_language or "tr"
        self.state = "main" if stored_language is not None else "language"
        self.message = "Program hazır. Bir seçenek seçin."
        self.message_color = self.MUTED
        self.buttons: list[Button] = []
        self.job: BackgroundJob | None = None
        self.training_job: BackgroundJob | None = None
        self.test_job: BackgroundJob | None = None
        self.match_env: PhysicsTankEnvV7 | None = None
        self.leo_agent: NamedAgent | None = None
        self.t90_agent: NamedAgent | None = None
        self.leo_obs = None
        self.t90_obs = None
        self.match_seed = 100_000
        self.match_number = 0
        self.match_label = ""
        self.match_result = ""
        self.result_frames = 0
        self.world_surface = pygame.Surface((800, 600))
        self.running = True
        self.state_before_confirmation = "main"
        self.training_records: list[TrainingRecord] = []
        self.records_page = 0
        self.pending_delete: TrainingRecord | None = None

    def t(self, value: str) -> str:
        return translated(self.language, value)

    def tf(self, value: str, **values: object) -> str:
        return self.t(value).format(**values)

    def change_language(self, language: str) -> None:
        if language not in SUPPORTED_LANGUAGES:
            return
        self.language = language
        if save_language(language):
            self.message = "Program hazır. Bir seçenek seçin."
            self.message_color = self.MUTED
        else:
            self.message = "Dil değiştirildi ancak tercih kaydedilemedi."
            self.message_color = self.RED

    def finish_language_selection(self, language: str) -> None:
        self.change_language(language)
        self.state = "main"

    def text(self, value: str, pos: tuple[int, int], color=None, font=None) -> None:
        surface = (font or self.font).render(self.t(value), True, color or self.TEXT)
        self.screen.blit(surface, pos)

    def centered(self, value: str, y: int, color=None, font=None) -> None:
        surface = (font or self.font).render(self.t(value), True, color or self.TEXT)
        self.screen.blit(surface, ((WINDOW_SIZE[0] - surface.get_width()) // 2, y))

    def button(self, y: int, label: str, action: Callable[[], None], enabled=True) -> None:
        self.buttons.append(
            Button(pygame.Rect(225, y, 450, 54), self.t(label), action, enabled)
        )

    def compact_button(
        self,
        rect: tuple[int, int, int, int],
        label: str,
        action: Callable[[], None],
        enabled: bool = True,
        selected: bool = False,
    ) -> None:
        self.buttons.append(
            Button(
                pygame.Rect(*rect),
                self.t(label),
                action,
                enabled,
                compact=True,
                selected=selected,
            )
        )

    def draw_buttons(self) -> None:
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            if not button.enabled:
                color = self.DISABLED
            elif button.selected:
                color = self.TEXT
            elif button.rect.collidepoint(mouse):
                color = self.BUTTON_HOVER
            else:
                color = self.BUTTON
            pygame.draw.rect(self.screen, color, button.rect, border_radius=10)
            pygame.draw.rect(self.screen, self.BLUE if button.enabled else self.DISABLED, button.rect, 2, 10)
            font = self.small if button.compact else self.font
            text_color = self.BG if button.selected else (
                self.TEXT if button.enabled else self.MUTED
            )
            rendered = font.render(button.label, True, text_color)
            self.screen.blit(rendered, rendered.get_rect(center=button.rect.center))

    def draw_language_switcher(self) -> None:
        self.compact_button(
            (752, 20, 52, 34),
            "TR",
            lambda: self.change_language("tr"),
            selected=self.language == "tr",
        )
        self.compact_button(
            (812, 20, 52, 34),
            "EN",
            lambda: self.change_language("en"),
            selected=self.language == "en",
        )

    def draw_language_selection(self) -> None:
        self.screen.fill(self.BG)
        self.buttons.clear()
        self.centered("IRON POLCY v7", 95, self.TEXT, self.title_font)
        self.centered("Dil seçin", 205, self.TEXT, self.font)
        self.centered("Choose language", 247, self.TEXT, self.font)
        self.centered(
            "Bu seçim daha sonra sağ üstten değiştirilebilir.",
            292,
            self.MUTED,
            self.small,
        )
        self.centered(
            "You can change this later from the top-right corner.",
            322,
            self.MUTED,
            self.small,
        )
        self.button(365, "Türkçe", lambda: self.finish_language_selection("tr"))
        self.button(435, "English", lambda: self.finish_language_selection("en"))
        self.draw_buttons()

    def draw_header(self, subtitle: str) -> None:
        self.centered("IRON POLCY v7", 48, self.TEXT, self.title_font)
        self.centered(subtitle, 100, self.MUTED, self.small)
        self.draw_language_switcher()

    def draw_main(self) -> None:
        self.screen.fill(self.BG)
        self.buttons.clear()
        self.draw_header("Tek pencere kontrol merkezi")
        self.button(135, "Hazır modelleri izle", lambda: self.start_match(False))
        self.button(193, "Son eğittiğim modelleri izle", lambda: self.start_match(True))
        self.button(251, "Eğitim seçenekleri", self.open_training)
        self.button(309, "Eğitim kayıtları", self.open_records)
        self.button(367, "Eğitim durumunu göster", self.open_status)
        self.button(425, "Programı teknik olarak kontrol et", self.start_tests, not self.job_running())
        self.button(483, "Çıkış", self.request_exit)
        self.draw_buttons()
        pygame.draw.rect(self.screen, self.PANEL, (90, 575, 720, 72), border_radius=10)
        self.centered(self.current_status(), 596, self.message_color, self.small)

    @staticmethod
    def _format_size(size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024.0 or unit == "GB":
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} GB"

    def record_is_active(self, record: TrainingRecord) -> bool:
        if self.training_job is None or not self.training_job.running:
            return False
        if self.training_job.training_phase != record.phase:
            return False
        progress = self.training_job.progress_snapshot()
        return progress is None or progress["seed"] is None or int(progress["seed"]) == record.seed

    def draw_records(self) -> None:
        self.screen.fill(self.BG)
        self.buttons.clear()
        self.draw_header("Eğitim kayıtları")
        per_page = 4
        total_pages = max(1, math.ceil(len(self.training_records) / per_page))
        self.records_page = min(self.records_page, total_pages - 1)
        start = self.records_page * per_page
        visible = self.training_records[start : start + per_page]
        if not visible:
            self.centered("Henüz eğitim kaydı bulunmuyor.", 300, self.TEXT, self.font)
        for index, record in enumerate(visible):
            y = 135 + index * 100
            active = self.record_is_active(record)
            status = self.t("Çalışıyor") if active else self.t(record.status)
            phase_label = self.t(
                PHASE_LABELS.get(record.phase, record.phase)
            ).split("—")[0].strip()
            percentage = (
                100.0 * record.completed_steps / record.total_steps
                if record.total_steps
                else 0.0
            )
            date_text = time.strftime("%d.%m.%Y %H:%M", time.localtime(record.modified))
            pygame.draw.rect(self.screen, self.PANEL, (45, y, 810, 86), border_radius=10)
            pygame.draw.rect(self.screen, self.TEXT, (45, y, 810, 86), 1, 10)
            self.text(f"{phase_label} • seed {record.seed}", (65, y + 10), self.TEXT, self.small)
            self.text(
                f"{status}  |  {record.completed_steps:,} / {record.total_steps:,}  "
                f"(%{percentage:.1f})  |  {self._format_size(record.size_bytes)}",
                (65, y + 39),
                self.TEXT,
                self.tiny,
            )
            self.text(date_text, (515, y + 12), self.TEXT, self.tiny)
            self.compact_button(
                (665, y + 25, 80, 38),
                "Aç",
                lambda record=record: self.open_record_folder(record),
            )
            self.compact_button(
                (755, y + 25, 80, 38),
                "Sil",
                lambda record=record: self.request_delete_record(record),
                not active,
            )
        if total_pages > 1:
            self.compact_button(
                (225, 555, 130, 42),
                "Önceki",
                self.previous_records_page,
                self.records_page > 0,
            )
            self.centered(
                self.tf(
                    "Sayfa {current} / {total}",
                    current=self.records_page + 1,
                    total=total_pages,
                ),
                565,
                self.TEXT,
                self.small,
            )
            self.compact_button(
                (545, 555, 130, 42),
                "Sonraki",
                self.next_records_page,
                self.records_page + 1 < total_pages,
            )
        if self.message:
            visible_message = self.message if len(self.message) <= 100 else self.message[:97] + "…"
            self.centered(visible_message, 595, self.message_color, self.tiny)
        self.button(620, "Ana menüye dön", self.open_main)
        self.draw_buttons()

    def draw_delete_confirmation(self) -> None:
        self.screen.fill(self.BG)
        self.buttons.clear()
        self.draw_header("Eğitim kaydını sil")
        record = self.pending_delete
        if record is None:
            self.state = "records"
            return
        phase_label = self.t(PHASE_LABELS.get(record.phase, record.phase)).split("—")[0].strip()
        pygame.draw.rect(self.screen, self.PANEL, (100, 180, 700, 220), border_radius=12)
        pygame.draw.rect(self.screen, self.TEXT, (100, 180, 700, 220), 1, 12)
        self.centered(f"{phase_label} • seed {record.seed}", 220, self.TEXT, self.font)
        self.centered(
            self.tf(
                "{completed:,} / {total:,} adım • {size}",
                completed=record.completed_steps,
                total=record.total_steps,
                size=self._format_size(record.size_bytes),
            ),
            270,
            self.TEXT,
            self.small,
        )
        self.centered("Bu kayıt Geri Dönüşüm Kutusu’na taşınacak.", 315, self.TEXT, self.small)
        self.button(440, "Evet — kaydı sil", self.confirm_delete_record)
        self.button(510, "Hayır — vazgeç", self.cancel_delete_record)
        self.draw_buttons()

    def draw_training(self) -> None:
        self.screen.fill(self.BG)
        self.buttons.clear()
        self.draw_header("Görüntü açmadan eğitim başlat")
        available = not self.job_running()
        self.button(154, PHASE_LABELS["smoke"], lambda: self.start_training("smoke"), available)
        self.button(218, PHASE_LABELS["behavior"], lambda: self.start_training("behavior"), available)
        self.button(282, PHASE_LABELS["pilot"], lambda: self.start_training("pilot"), available)
        self.button(346, PHASE_LABELS["full"], lambda: self.start_training("full"), available)
        self.button(430, "Eğitim durumunu göster", self.open_status)
        self.button(494, "Ana menüye dön", self.open_main)
        self.draw_buttons()
        self.centered("Eğitim sırasında bu pencere açık kalır; başka pencere açılmaz.", 580, self.MUTED, self.small)

    def viewed_job(self) -> BackgroundJob | None:
        return self.test_job if self.state == "test_status" else self.training_job

    def draw_status(self) -> None:
        self.screen.fill(self.BG)
        self.buttons.clear()
        test_view = self.state == "test_status"
        job = self.viewed_job()
        self.draw_header("Teknik kontrol durumu" if test_view else "Eğitim durumu")
        pygame.draw.rect(self.screen, self.PANEL, (55, 145, 790, 390), border_radius=10)
        record_count = count_training_records()
        if job is None:
            if test_view:
                self.centered("Henüz teknik kontrol başlatılmadı.", 240, self.TEXT, self.font)
                self.centered(
                    "Bu kontrol eğitimlerden ve eğitim kayıtlarından bağımsızdır.",
                    295,
                    self.MUTED,
                    self.small,
                )
                self.button(405, "Teknik kontrolü başlat", self.start_tests, not self.job_running())
            elif record_count:
                self.centered("Şu anda çalışan bir eğitim yok.", 220, self.TEXT, self.font)
                self.centered(
                    self.tf(
                        "{count} kayıtlı eğitim bulundu; bunlar şu anda aktif değil.",
                        count=record_count,
                    ),
                    278,
                    self.MUTED,
                    self.small,
                )
                self.centered(
                    "Kayıtları görmek veya silmek için aşağıdaki düğmeyi kullanın.",
                    315,
                    self.MUTED,
                    self.small,
                )
                self.button(405, "Eğitim kayıtlarını göster", self.open_records)
            else:
                self.centered("Şu anda çalışan bir eğitim yok.", 220, self.TEXT, self.font)
                self.centered("Kayıtlı bir eğitim de bulunamadı.", 278, self.MUTED, self.small)
                self.centered(
                    "Yeni bir eğitim başlatmak için eğitim seçeneklerini açın.",
                    315,
                    self.MUTED,
                    self.small,
                )
                self.button(405, "Eğitim seçeneklerini aç", self.open_training)
        else:
            if job.running:
                color = self.BLUE
                status = "Çalışıyor…"
            elif job.return_code is None:
                color = self.BLUE
                status = "Sonuç hazırlanıyor…"
            elif job.stopped_by_user:
                color = self.MUTED
                status = "Senin isteğinle durduruldu"
            elif job.return_code == 0:
                color = self.GREEN
                status = "Tamamlandı"
            else:
                color = self.RED
                status = "Tamamlanamadı"
            self.text(job.title, (78, 167), self.TEXT, self.font)
            status_surface = self.small.render(self.t(status), True, color)
            self.screen.blit(status_surface, (820 - status_surface.get_width(), 170))
            progress = job.progress_snapshot()
            if progress is not None:
                bar = pygame.Rect(78, 212, 744, 24)
                pygame.draw.rect(self.screen, (12, 18, 25), bar, border_radius=7)
                fill_width = int(bar.width * float(progress["percent"]) / 100.0)
                if fill_width:
                    pygame.draw.rect(
                        self.screen,
                        self.GREEN,
                        (bar.x, bar.y, fill_width, bar.height),
                        border_radius=7,
                    )
                self.text(f"%{float(progress['percent']):.1f}", (760, 214), self.TEXT, self.tiny)
                self.text(
                    self.tf(
                        "Adım: {completed:,} / {total:,}",
                        completed=int(progress["completed_steps"]),
                        total=int(progress["total_steps"]),
                    ),
                    (78, 250),
                    self.TEXT,
                    self.small,
                )
                self.text(
                    self.tf(
                        "Tur: {completed:,} / {total:,}",
                        completed=int(progress["completed_generations"]),
                        total=int(progress["total_generations"]),
                    ),
                    (390, 250),
                    self.TEXT,
                    self.small,
                )
                self.text(
                    f"PPO epoch: {int(progress['completed_epochs']):,} / {int(progress['total_epochs']):,}",
                    (78, 278),
                    self.TEXT,
                    self.small,
                )
                seed_text = self.t("hazırlanıyor") if progress["seed"] is None else str(progress["seed"])
                self.text(
                    self.tf(
                        "Seed: {seed}   Hız: {speed:,} adım/sn",
                        seed=seed_text,
                        speed=int(progress["sps"]),
                    ),
                    (390, 278),
                    self.MUTED,
                    self.small,
                )
            if test_view:
                if job.running or job.return_code is None:
                    message_lines = [
                        "Programın temel parçaları kontrol ediliyor.",
                        "Bu işlem tankları eğitmez ve eğitim kayıtlarını değiştirmez.",
                    ]
                elif job.return_code == 0:
                    passed = next(
                        (
                            match.group(1)
                            for line in reversed(job.snapshot())
                            if (match := re.search(r"(\d+) passed", line))
                        ),
                        None,
                    )
                    result_text = (
                        self.tf(
                            "{passed} otomatik kontrol başarıyla geçti.",
                            passed=passed,
                        )
                        if passed
                        else self.t("Otomatik kontroller başarıyla geçti.")
                    )
                    message_lines = [result_text, "Program düzgün çalışıyor."]
                else:
                    message_lines = [
                        "Programın teknik kontrolü tamamlanamadı.",
                        "Bu mesaj, eğitim kaydı bulunmadığı anlamına gelmez.",
                        "Uygulamayı yeniden açıp kontrolü tekrar deneyebilirsiniz.",
                    ]
            elif job.running or job.return_code is None:
                message_lines = [
                    "Eğitim devam ediyor; ilerleme yukarıda canlı gösteriliyor.",
                    "Oluşan dosyalar otomatik olarak eğitim kayıtlarına ekleniyor.",
                ]
            elif job.stopped_by_user:
                message_lines = [
                    "Eğitim senin isteğinle durduruldu.",
                    "O ana kadar oluşan dosyalar eğitim kayıtlarında saklandı.",
                ]
            elif job.return_code == 0:
                message_lines = [
                    "Eğitim başarıyla tamamlandı.",
                    "Yeni modeller izlenmeye hazır.",
                ]
            else:
                message_lines = [
                    "Eğitim beklenmedik biçimde durdu.",
                    "Varsa yarım kalan dosyalar eğitim kayıtlarından görülebilir.",
                ]
            y = 340 if progress is not None else 245
            for line in message_lines:
                self.centered(line, y, self.MUTED, self.small)
                y += 38
            if record_count and not test_view:
                self.button(475, "Eğitim kayıtlarını göster", self.open_records)
        self.button(565, "Ana menüye dön", self.open_main)
        if (
            not test_view
            and job is not None
            and job.training_details is not None
            and not job.running
            and job.return_code == 0
            and latest_model_pair()
        ):
            self.button(629, "Son eğitilen modelleri izle", lambda: self.start_match(True))
        self.draw_buttons()

    def draw_match(self) -> None:
        assert self.match_env is not None
        self.screen.fill(self.BG)
        self.buttons.clear()
        self.match_env._draw_world(self.world_surface)
        self.screen.blit(self.world_surface, WORLD_OFFSET)
        pygame.draw.rect(self.screen, (10, 15, 21), (50, 618, 800, 64))
        self.text(self.match_label, (68, 630), self.TEXT, self.small)
        self.text("ESC: ana menü", (690, 630), self.MUTED, self.small)
        result = self.match_result or self.tf(
            "Maç {match} • adım {step}",
            match=self.match_number + 1,
            step=self.match_env.current_step,
        )
        self.text(result, (68, 655), self.GREEN if self.match_result else self.MUTED, self.small)
        self.draw_language_switcher()
        self.draw_buttons()

    def draw_exit_confirmation(self) -> None:
        self.screen.fill(self.BG)
        self.buttons.clear()
        self.draw_header("Çıkış onayı")
        pygame.draw.rect(self.screen, self.PANEL, (110, 180, 680, 205), border_radius=12)
        self.centered("Eğitim şu anda devam ediyor.", 220, self.TEXT, self.font)
        self.centered(
            "Çıkarsanız eğitim durdurulacak ve arkada çalışmayacak.",
            270,
            self.MUTED,
            self.small,
        )
        self.centered("Çıkmak istediğinizden emin misiniz?", 310, self.RED, self.small)
        self.button(430, "Evet — eğitimi durdur ve çık", self.confirm_exit)
        self.button(500, "Hayır — eğitime devam et", self.cancel_exit)
        self.draw_buttons()

    def current_status(self) -> str:
        if self.job is None:
            return self.t(self.message)
        title = self.t(self.job.title)
        if self.job.running:
            return self.tf("{title} arka planda çalışıyor.", title=title)
        if self.job.return_code is None:
            return self.tf("{title} sonucu hazırlanıyor.", title=title)
        if self.job.stopped_by_user:
            return self.tf("{title} senin isteğinle durduruldu.", title=title)
        if self.job.return_code == 0:
            return self.tf("{title} tamamlandı.", title=title)
        return self.tf(
            "{title} tamamlanamadı; açıklama için durum ekranını açın.",
            title=title,
        )

    def job_running(self) -> bool:
        return self.job is not None and self.job.running

    def open_main(self) -> None:
        self.state = "main"

    def open_training(self) -> None:
        self.state = "training"

    def open_records(self) -> None:
        self.training_records = discover_training_records()
        self.records_page = 0
        self.pending_delete = None
        self.state = "records"

    def previous_records_page(self) -> None:
        self.records_page = max(0, self.records_page - 1)

    def next_records_page(self) -> None:
        total_pages = max(1, math.ceil(len(self.training_records) / 4))
        self.records_page = min(total_pages - 1, self.records_page + 1)

    def open_record_folder(self, record: TrainingRecord) -> None:
        if not record.path.is_dir():
            self.message = "Eğitim klasörü artık mevcut değil. Kayıt listesi yenilendi."
            self.message_color = self.RED
            self.open_records()
            return
        try:
            if os.name == "nt":
                os.startfile(record.path)  # type: ignore[attr-defined]
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen(
                    [opener, str(record.path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except (AttributeError, OSError) as exc:
            self.message = self.tf("Eğitim klasörü açılamadı: {error}", error=exc)
            self.message_color = self.RED

    def request_delete_record(self, record: TrainingRecord) -> None:
        if self.record_is_active(record):
            self.message = "Devam eden eğitim kaydı silinemez."
            self.message_color = self.RED
            return
        self.pending_delete = record
        self.state = "confirm_delete"

    def cancel_delete_record(self) -> None:
        self.pending_delete = None
        self.state = "records"

    def confirm_delete_record(self) -> None:
        record = self.pending_delete
        if record is None:
            self.state = "records"
            return
        if self.record_is_active(record):
            self.message = "Devam eden eğitim kaydı silinemez."
            self.message_color = self.RED
            self.cancel_delete_record()
            return
        try:
            recycle_training_record(record.path)
            self.message = self.tf(
                "Seed {seed} kaydı Geri Dönüşüm Kutusu’na taşındı.",
                seed=record.seed,
            )
            self.message_color = self.GREEN
        except Exception as exc:
            self.message = self.tf("Eğitim kaydı silinemedi: {error}", error=exc)
            self.message_color = self.RED
        self.training_records = discover_training_records()
        total_pages = max(1, math.ceil(len(self.training_records) / 4))
        self.records_page = min(self.records_page, total_pages - 1)
        self.pending_delete = None
        self.state = "records"

    def open_status(self) -> None:
        self.state = "status"

    def start_training(self, phase: str) -> None:
        if self.job_running():
            self.message = "Önce devam eden işlemin tamamlanmasını bekleyin."
            self.message_color = self.RED
            return
        command = [
            str(console_python()),
            "-u",
            str(ROOT / "run_experiments_v7.py"),
            phase,
            "--execute",
        ]
        try:
            self.job = BackgroundJob(
                PHASE_LABELS[phase],
                command,
                training_details=PHASE_DETAILS[phase],
                training_phase=phase,
            )
            self.training_job = self.job
            self.message = self.tf(
                "{title} başlatıldı.", title=self.t(PHASE_LABELS[phase])
            )
            self.message_color = self.BLUE
            self.state = "status"
        except Exception as exc:
            self.message = self.tf("Eğitim başlatılamadı: {error}", error=exc)
            self.message_color = self.RED
            self.state = "main"

    def start_tests(self) -> None:
        if self.job_running():
            return
        command = [
            str(console_python()),
            "-B",
            "-u",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        try:
            self.job = BackgroundJob("Program testleri", command)
            self.test_job = self.job
            self.message = "Program testleri başlatıldı."
            self.message_color = self.BLUE
            self.state = "test_status"
        except Exception as exc:
            self.message = self.tf("Test başlatılamadı: {error}", error=exc)
            self.message_color = self.RED

    def start_match(self, latest: bool) -> None:
        pair = latest_model_pair() if latest else ready_model_pair()
        if pair is None:
            self.message = "İzlenecek model bulunamadı. Önce bir eğitim tamamlayın."
            self.message_color = self.RED
            self.state = "main"
            return
        self.screen.fill(self.BG)
        self.centered("Modeller yükleniyor…", 315, self.TEXT, self.font)
        pygame.display.flip()
        try:
            self.leo_agent = load_agent(pair[0], "Leo")
            self.t90_agent = load_agent(pair[1], "T-90")
            self.match_env = PhysicsTankEnvV7(seed=self.match_seed)
            self.match_number = 0
            self.match_result = ""
            self.result_frames = 0
            self.match_label = "Son eğitim" if latest else "Hazır modeller"
            self.reset_match()
            self.state = "match"
        except Exception as exc:
            self.close_match()
            self.message = self.tf("Modeller açılamadı: {error}", error=exc)
            self.message_color = self.RED
            self.state = "main"

    def reset_match(self) -> None:
        assert self.match_env is not None
        seed = self.match_seed + self.match_number // 2
        leo_left = self.match_number % 2 == 0
        self.leo_obs, self.t90_obs, _ = self.match_env.reset_joint(seed=seed, leo_on_left=leo_left)
        self.match_result = ""
        self.result_frames = 0

    def update_match(self) -> None:
        if self.match_env is None or self.leo_agent is None or self.t90_agent is None:
            return
        if self.result_frames > 0:
            self.result_frames -= 1
            if self.result_frames == 0:
                self.match_number += 1
                self.reset_match()
            return
        leo_action, _ = self.leo_agent.agent.predict(self.leo_obs, deterministic=True)
        t90_action, _ = self.t90_agent.agent.predict(self.t90_obs, deterministic=True)
        (
            self.leo_obs,
            self.t90_obs,
            _,
            _,
            terminated,
            truncated,
            info,
        ) = self.match_env.step_joint(leo_action, t90_action, compact_info=True)
        if terminated or truncated:
            names = {
                "leo_win": "Leo kazandı",
                "t90_win": "T-90 kazandı",
                "double_ko": "Çifte nakavt",
                "timeout_draw": "Berabere",
            }
            self.match_result = names.get(str(info.get("outcome")), "Maç tamamlandı")
            self.result_frames = 45

    def close_match(self) -> None:
        if self.match_env is not None:
            self.match_env.close()
        self.match_env = None
        self.leo_agent = None
        self.t90_agent = None
        self.leo_obs = None
        self.t90_obs = None

    def request_exit(self) -> None:
        if self.job_running():
            if self.state != "confirm_exit":
                self.state_before_confirmation = self.state
            self.state = "confirm_exit"
            return
        self.running = False

    def cancel_exit(self) -> None:
        self.state = self.state_before_confirmation

    def confirm_exit(self) -> None:
        if self.job_running() and self.job is not None:
            self.job.stop()
        self.running = False

    def handle_click(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.enabled and button.rect.collidepoint(pos):
                button.action()
                return

    def run(self) -> int:
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.request_exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        if self.state == "language":
                            continue
                        if self.state == "match":
                            self.close_match()
                            self.state = "main"
                            self.message = "Maçtan ana menüye dönüldü."
                            self.message_color = self.MUTED
                        elif self.state == "confirm_delete":
                            self.cancel_delete_record()
                        elif self.state == "records":
                            self.state = "main"
                        elif self.state != "main":
                            self.state = "main"
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.handle_click(event.pos)

                if self.state == "language":
                    self.draw_language_selection()
                elif self.state == "match":
                    self.update_match()
                    self.draw_match()
                elif self.state == "confirm_exit":
                    self.draw_exit_confirmation()
                elif self.state == "training":
                    self.draw_training()
                elif self.state in {"status", "test_status"}:
                    self.draw_status()
                elif self.state == "records":
                    self.draw_records()
                elif self.state == "confirm_delete":
                    self.draw_delete_confirmation()
                else:
                    self.draw_main()
                pygame.display.flip()
                self.clock.tick(30)
        finally:
            self.close_match()
            pygame.quit()
        return 0


def main() -> int:
    return TankLauncher().run()


if __name__ == "__main__":
    raise SystemExit(main())

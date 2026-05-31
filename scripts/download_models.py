#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import urllib.request
import importlib.util
import zipfile
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LOCAL_YOLO_CANDIDATES = [
    Path.home() / "Desktop/01_进行中项目/2026-04_robot_life_dev/models/yolo/yolov8n.pt",
    Path.home() / ".openclaw/workspace/tmp_om1/system_hw_test/yolov8n.pt",
]
TEN_VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.int8.onnx"
SENSEVOICE_ONNX_DIR = MODELS / "asr" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
STREAMING_ZIPFORMER_CTC_DIR = MODELS / "asr" / "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
STREAMING_ZIPFORMER_CTC_ARCHIVE = "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2"
STREAMING_ZIPFORMER_CTC_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    + STREAMING_ZIPFORMER_CTC_ARCHIVE
)
VOICEPRINT_MODEL = MODELS / "voiceprint" / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
VOICEPRINT_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
SENSEVOICE_ONNX_FILES = {
    "model.int8.onnx": [
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx",
        "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx",
    ],
    "tokens.txt": [
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt",
        "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt",
    ],
    "test_wavs/zh.wav": [
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/zh.wav",
        "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/zh.wav",
    ],
}
LR_ASD_WEIGHTS = {
    "pretrain_AVA.model": "https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/main/weight/pretrain_AVA.model",
    "finetuning_TalkSet.model": "https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/main/weight/finetuning_TalkSet.model",
}
INSIGHTFACE_BUFFALO_L_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
]
INSIGHTFACE_BUFFALO_L_FILES = [
    "det_10g.onnx",
    "w600k_r50.onnx",
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
]


def ensure_dirs() -> None:
    for name in ["yolo", "asr", "vad", "face", "asd", "voiceprint"]:
        (MODELS / name).mkdir(parents=True, exist_ok=True)


def copy_yolo() -> None:
    dest = MODELS / "yolo" / "yolov8n.pt"
    if dest.exists():
        print(f"OK yolo: {dest}")
        return
    for candidate in LOCAL_YOLO_CANDIDATES:
        if candidate.exists():
            shutil.copy2(candidate, dest)
            print(f"OK yolo copied: {candidate} -> {dest}")
            return
    print("WARN yolo: no local candidate found")


def download_sensevoice() -> None:
    dest = MODELS / "asr" / "SenseVoiceSmall"
    model_file = dest / "model.pt"
    if model_file.exists():
        print(f"OK SenseVoiceSmall: {model_file}")
        return
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("Installing modelscope...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "modelscope"])
        from modelscope import snapshot_download

    print("Downloading iic/SenseVoiceSmall from ModelScope...")
    snapshot_download("iic/SenseVoiceSmall", local_dir=str(dest), cache_dir=str(MODELS / "asr" / "modelscope"))
    print(f"OK SenseVoiceSmall: {model_file}")


def try_download(url: str, dest: Path, timeout_s: int = 20) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"OK existing: {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    try:
        print(f"Downloading {url} -> {dest}")
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            with temp.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        temp.replace(dest)
        print(f"OK downloaded: {dest}")
        return True
    except Exception as error:  # noqa: BLE001 - download script reports and continues.
        temp.unlink(missing_ok=True)
        print(f"WARN download failed: {url}: {error}")
        return False


def ensure_silero_vad() -> None:
    dest = MODELS / "vad" / "silero_vad_v6.onnx"
    if dest.exists():
        print(f"OK Silero VAD: {dest}")
        return
    print("WARN Silero VAD: no local candidate found")


def download_sensevoice_onnx() -> None:
    for relative, urls in SENSEVOICE_ONNX_FILES.items():
        dest = SENSEVOICE_ONNX_DIR / relative
        min_size = 100 * 1024 * 1024 if relative.endswith(".onnx") else 1024
        if dest.exists() and dest.stat().st_size >= min_size:
            print(f"OK SenseVoice ONNX file: {dest}")
            continue
        if dest.exists():
            print(f"WARN removing incomplete SenseVoice file: {dest}")
            dest.unlink()
        for url in urls:
            if try_download(url, dest, timeout_s=600):
                break


def download_streaming_zipformer_ctc() -> None:
    model = STREAMING_ZIPFORMER_CTC_DIR / "model.int8.onnx"
    tokens = STREAMING_ZIPFORMER_CTC_DIR / "tokens.txt"
    if model.exists() and model.stat().st_size > 10 * 1024 * 1024 and tokens.exists():
        print(f"OK streaming Zipformer CTC: {STREAMING_ZIPFORMER_CTC_DIR}")
        return
    archive_path = MODELS / "asr" / STREAMING_ZIPFORMER_CTC_ARCHIVE
    if not try_download(STREAMING_ZIPFORMER_CTC_URL, archive_path, timeout_s=900):
        print("WARN streaming Zipformer CTC: download unavailable")
        return
    if STREAMING_ZIPFORMER_CTC_DIR.exists():
        shutil.rmtree(STREAMING_ZIPFORMER_CTC_DIR)
    with tarfile.open(archive_path, "r:bz2") as archive:
        root = (MODELS / "asr").resolve()
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"unsafe archive member path: {member.name}")
        archive.extractall(root)
    if model.exists() and tokens.exists():
        print(f"OK streaming Zipformer CTC extracted: {STREAMING_ZIPFORMER_CTC_DIR}")
    else:
        print(f"WARN streaming Zipformer CTC incomplete: {STREAMING_ZIPFORMER_CTC_DIR}")


def download_optional_edge_models() -> None:
    try_download(TEN_VAD_URL, MODELS / "vad" / "ten-vad.int8.onnx")
    download_sensevoice_onnx()
    download_streaming_zipformer_ctc()
    try_download(VOICEPRINT_MODEL_URL, VOICEPRINT_MODEL, timeout_s=900)
    for filename, url in LR_ASD_WEIGHTS.items():
        try_download(url, MODELS / "asd" / "lr-asd" / filename)
    export_lr_asd_onnx()
    download_insightface_buffalo_l()

def export_lr_asd_onnx() -> None:
    checkpoint = MODELS / "asd" / "lr-asd" / "finetuning_TalkSet.model"
    output = MODELS / "asd" / "lr-asd" / "lr_asd_talkset.onnx"
    if not checkpoint.exists():
        print(f"WARN LR-ASD portable ONNX: missing checkpoint {checkpoint}")
        return
    if output.exists() and output.stat().st_mtime >= checkpoint.stat().st_mtime:
        print(f"OK LR-ASD portable ONNX: {output}")
        return
    try:
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts" / "export_lr_asd_onnx.py"),
             "--checkpoint", str(checkpoint), "--output", str(output)]
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"WARN LR-ASD portable ONNX export failed: {error}")


def insightface_model_dir() -> Path:
    return Path.home() / ".insightface" / "models" / "buffalo_l"


def insightface_buffalo_l_complete() -> bool:
    model_dir = insightface_model_dir()
    return all((model_dir / filename).exists() and (model_dir / filename).stat().st_size > 0 for filename in INSIGHTFACE_BUFFALO_L_FILES)


def safe_extract_zip(zip_path: Path, dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    root = dest_root.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"unsafe archive member path: {member.filename}")
        archive.extractall(root)


def install_insightface_buffalo_l(archive_path: Path) -> None:
    target_dir = insightface_model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_root = MODELS / "face" / ".buffalo_l_extract"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    try:
        safe_extract_zip(archive_path, temp_root)
        for filename in INSIGHTFACE_BUFFALO_L_FILES:
            candidates = [
                temp_root / filename,
                temp_root / "buffalo_l" / filename,
            ]
            source = next((candidate for candidate in candidates if candidate.exists()), None)
            if source is None:
                matches = list(temp_root.rglob(filename))
                source = matches[0] if matches else None
            if source is None:
                raise RuntimeError(f"missing {filename} in {archive_path}")
            shutil.copy2(source, target_dir / filename)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def download_insightface_buffalo_l() -> None:
    if insightface_buffalo_l_complete():
        print(f"OK InsightFace buffalo_l: {insightface_model_dir()}")
        return
    archive_path = MODELS / "face" / "buffalo_l.zip"
    for url in INSIGHTFACE_BUFFALO_L_URLS:
        if try_download(url, archive_path, timeout_s=900):
            break
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        print("WARN InsightFace buffalo_l: download unavailable")
        return
    try:
        install_insightface_buffalo_l(archive_path)
    except Exception as error:  # noqa: BLE001
        print(f"WARN InsightFace buffalo_l extract failed: {error}")
        return
    if insightface_buffalo_l_complete():
        print(f"OK InsightFace buffalo_l extracted: {insightface_model_dir()}")
    else:
        print(f"WARN InsightFace buffalo_l incomplete after extraction: {insightface_model_dir()}")


def write_manifest() -> None:
    manifest = MODELS / "MODEL_STATUS.txt"
    sensevoice_model = SENSEVOICE_ONNX_DIR / "model.int8.onnx"
    sensevoice_tokens = SENSEVOICE_ONNX_DIR / "tokens.txt"
    insightface_ok, insightface_error = insightface_status()
    lines = [
        "Video Speaker ID local model status",
        "",
        f"YOLO person detector: {(MODELS / 'yolo' / 'yolov8n.pt').exists()} {(MODELS / 'yolo' / 'yolov8n.pt')}",
        f"YOLO26 person detector ONNX: {(MODELS / 'yolo' / 'yolo26n.onnx').exists()} {(MODELS / 'yolo' / 'yolo26n.onnx')}",
        f"Silero VAD ONNX: {(MODELS / 'vad' / 'silero_vad_v6.onnx').exists()} {(MODELS / 'vad' / 'silero_vad_v6.onnx')}",
        f"SenseVoiceSmall ModelScope: {(MODELS / 'asr' / 'SenseVoiceSmall' / 'model.pt').exists()} {(MODELS / 'asr' / 'SenseVoiceSmall' / 'model.pt')}",
        f"Streaming Zipformer CTC: {(STREAMING_ZIPFORMER_CTC_DIR / 'model.int8.onnx').exists()} {STREAMING_ZIPFORMER_CTC_DIR}",
        f"Voiceprint speaker embedding: {VOICEPRINT_MODEL.exists()} {VOICEPRINT_MODEL}",
        f"TEN-VAD ONNX: {(MODELS / 'vad' / 'ten-vad.int8.onnx').exists()} {(MODELS / 'vad' / 'ten-vad.int8.onnx')}",
        f"Sherpa SenseVoice ONNX model: {sensevoice_model.exists()} {sensevoice_model}",
        f"Sherpa SenseVoice ONNX tokens: {sensevoice_tokens.exists()} {sensevoice_tokens}",
        f"InsightFace FaceAnalysis usable: {insightface_ok}{' (' + insightface_error + ')' if insightface_error else ''}",
        f"InsightFace buffalo_l complete: {insightface_buffalo_l_complete()} {insightface_model_dir()}",
        f"LR-ASD pretrain: {(MODELS / 'asd' / 'lr-asd' / 'pretrain_AVA.model').exists()} {(MODELS / 'asd' / 'lr-asd' / 'pretrain_AVA.model')}",
        f"LR-ASD TalkSet finetune: {(MODELS / 'asd' / 'lr-asd' / 'finetuning_TalkSet.model').exists()} {(MODELS / 'asd' / 'lr-asd' / 'finetuning_TalkSet.model')}",
        f"LR-ASD portable ONNX: {(MODELS / 'asd' / 'lr-asd' / 'lr_asd_talkset.onnx').exists()} {(MODELS / 'asd' / 'lr-asd' / 'lr_asd_talkset.onnx')}",
        "TensorRT deployment: pending; portable C++ runtime reports CPU ORT and rejects unlinked Orin profiles.",
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


def insightface_status() -> tuple[bool, str]:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    if importlib.util.find_spec("insightface") is None:
        return False, "package missing"
    try:
        from insightface.app import FaceAnalysis  # noqa: F401
        return True, ""
    except Exception as error:  # noqa: BLE001
        return False, f"{type(error).__name__}: {error}"


def main() -> int:
    ensure_dirs()
    (MODELS / "asd" / "lr-asd").mkdir(parents=True, exist_ok=True)
    copy_yolo()
    ensure_silero_vad()
    download_sensevoice()
    download_optional_edge_models()
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

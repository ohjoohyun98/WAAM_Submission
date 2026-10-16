"""Input provenance shared by validation and derived-artifact workers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

RESULT_SCHEMA_VERSION: Final = "4.0"
VALIDATION_INPUT_MANIFEST: Final = "validation_inputs.json"
INPUT_FILENAMES: Final = ("config.yaml", "trajectory.csv", "target.stl")


InputFingerprint = dict[str, int | str]
InputSignature = dict[str, InputFingerprint]


def file_fingerprint(path: Path) -> InputFingerprint:
    """Return a content-backed identity for one input file."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"Input changed while its fingerprint was being calculated: {path}")
    return {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def input_signature(job_dir: Path) -> InputSignature:
    """Capture content hashes and filesystem metadata for all required inputs."""
    result: InputSignature = {}
    for filename in INPUT_FILENAMES:
        result[filename] = file_fingerprint(job_dir / filename)
    return result


def write_validation_input_manifest(
    job_dir: Path,
    run_dir: Path,
    signature: InputSignature | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> None:
    """Write one canonical manifest for a completed validation run."""
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "inputs": signature if signature is not None else input_signature(job_dir),
    }
    if config_snapshot is not None:
        payload["config"] = config_snapshot
    target = run_dir / VALIDATION_INPUT_MANIFEST
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def verify_validation_inputs(
    job_dir: Path,
    run_dir: Path,
    *,
    current_signature: InputSignature | None = None,
) -> tuple[bool, str]:
    """Verify that a schema 4.0 run still refers to the current three inputs."""
    path = run_dir / VALIDATION_INPUT_MANIFEST
    if not path.is_file():
        return False, "현재 결과 형식에는 입력 지문이 없습니다. Validation을 다시 실행하세요."
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False, "입력 지문 파일의 최상위 값이 JSON object가 아닙니다."
        if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
            return False, "현재 결과 schema의 입력 지문이 아닙니다. Validation을 다시 실행하세요."
        recorded = payload.get("inputs")
        if not isinstance(recorded, dict) or any(
            not isinstance(recorded.get(filename), dict)
            or "sha256" not in recorded[filename]
            for filename in INPUT_FILENAMES
        ):
            return (
                False,
                "이 결과의 입력 지문에는 SHA-256이 없습니다. Validation을 다시 실행하세요.",
            )
        current = (
            current_signature if current_signature is not None else input_signature(job_dir)
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"입력 지문을 확인할 수 없습니다: {exc}"
    if recorded != current:
        return False, "검증 이후 입력 파일이 변경되었습니다. Validation을 다시 실행하세요."
    return True, ""

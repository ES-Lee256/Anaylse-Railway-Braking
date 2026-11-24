"""Download the selected CC BY 4.0 files from the Mendeley bearing dataset."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

DATASET_ID = "v43hmbwxpm"
API_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "mendeley_bearing_v2"
SELECTED_FILES = tuple(
    f"{prefix}-A-{trial}.mat"
    for prefix in ("H", "I", "O", "B", "C")
    for trial in range(1, 4)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_selected_files(
    destination: Path = DATA_DIRECTORY,
) -> tuple[Path, ...]:
    request = urllib.request.Request(
        API_URL, headers={"User-Agent": "RailwayVehicleCoursework/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)

    license_name = metadata.get("data_licence", {}).get("short_name")
    if license_name != "CC BY 4.0":
        raise ValueError(f"数据集许可发生变化，当前为 {license_name!r}。")
    files = {item["filename"]: item for item in metadata.get("files", [])}
    missing = set(SELECTED_FILES).difference(files)
    if missing:
        raise ValueError(f"数据集文件清单不完整：{sorted(missing)}")

    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename in SELECTED_FILES:
        entry = files[filename]
        expected_hash = entry["content_details"]["sha256_hash"]
        target = destination / filename
        if not target.is_file() or _sha256(target) != expected_hash:
            temporary = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(
                entry["content_details"]["download_url"],
                headers={"User-Agent": "RailwayVehicleCoursework/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    with temporary.open("wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                if _sha256(temporary) != expected_hash:
                    raise ValueError(f"{filename} 的 SHA-256 校验失败。")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        downloaded.append(target)
        print(f"Verified {filename} ({target.stat().st_size:,} bytes)")
    return tuple(downloaded)


if __name__ == "__main__":
    download_selected_files()

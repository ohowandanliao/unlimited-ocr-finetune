#!/usr/bin/env python3
"""Download MER-17M WebDataset shards directly into an explicit data root.

This downloader intentionally avoids the Hugging Face cache. Partial files,
the repository manifest, and verification state all live under --out.
Only primary ``.tar`` shards are selected; stale ``.tar.bak`` files are
excluded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path, PurePosixPath


REPO = "topdu/MER-17M"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
CHUNK_SIZE = 4 * 1024 * 1024
CURL = shutil.which("curl")


@dataclasses.dataclass(frozen=True)
class Shard:
    path: str
    size: int
    sha256: str

    @property
    def bucket(self) -> str:
        return self.path.split("/", 1)[0]


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def request_bytes(url: str, timeout: int) -> bytes:
    if CURL is None:
        raise RuntimeError("curl is required but was not found on PATH")
    result = subprocess.run(
        [
            CURL,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            str(timeout),
            url,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"curl failed ({result.returncode}): {detail}")
    return result.stdout


def fetch_manifest(endpoint: str, timeout: int) -> tuple[dict, list[Shard]]:
    url = f"{endpoint.rstrip('/')}/api/datasets/{REPO}?blobs=true"
    info = json.loads(request_bytes(url, timeout))
    shards: list[Shard] = []
    for item in info.get("siblings", []):
        path = str(item.get("rfilename", ""))
        if not path.endswith(".tar"):
            continue
        safe_path = PurePosixPath(path)
        if safe_path.is_absolute() or ".." in safe_path.parts:
            raise ValueError(f"unsafe repository path: {path!r}")
        lfs = item.get("lfs") or {}
        size = int(lfs.get("size") or item.get("size") or 0)
        sha256 = str(lfs.get("sha256") or "")
        if size <= 0 or len(sha256) != 64:
            raise ValueError(f"missing LFS metadata for {path!r}")
        shards.append(Shard(path=path, size=size, sha256=sha256))
    shards.sort(key=lambda shard: shard.path)
    if not shards:
        raise RuntimeError("manifest contains no primary .tar shards")
    return info, shards


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


class VerificationState:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.verified: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.verified[str(row.get("path", ""))] = str(row.get("sha256", ""))

    def contains(self, shard: Shard) -> bool:
        return self.verified.get(shard.path) == shard.sha256

    def add(self, shard: Shard) -> None:
        row = {
            "path": shard.path,
            "size": shard.size,
            "sha256": shard.sha256,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        encoded = json.dumps(row, ensure_ascii=True, sort_keys=True)
        with self.lock:
            if self.contains(shard):
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.verified[shard.path] = shard.sha256


def prepare_partial(destination: Path, shard: Shard) -> Path:
    partial = destination.with_name(destination.name + ".part")
    if destination.exists() and destination.stat().st_size != shard.size:
        if destination.stat().st_size < shard.size:
            if not partial.exists() or destination.stat().st_size > partial.stat().st_size:
                partial.unlink(missing_ok=True)
                destination.replace(partial)
            else:
                destination.unlink()
        else:
            destination.unlink()
    if partial.exists() and partial.stat().st_size > shard.size:
        partial.unlink()
    return partial


def download_url(endpoint: str, path: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"{endpoint.rstrip('/')}/datasets/{REPO}/resolve/main/{quoted}?download=true"


def transfer_once(url: str, partial: Path, expected_size: int, timeout: int) -> None:
    if CURL is None:
        raise RuntimeError("curl is required but was not found on PATH")
    result = subprocess.run(
        [
            CURL,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "30",
            "--max-time",
            str(timeout),
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise IOError(f"curl failed ({result.returncode}): {detail}")
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise IOError(f"incomplete transfer: got {actual_size}, expected {expected_size}")


def download_one(
    shard: Shard,
    data_root: Path,
    endpoint: str,
    timeout: int,
    retries: int,
    state: VerificationState,
) -> tuple[str, str]:
    destination = data_root / "raw" / PurePosixPath(shard.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == shard.size:
        if state.contains(shard) or sha256_file(destination) == shard.sha256:
            state.add(shard)
            return shard.path, "verified"
        destination.unlink()

    partial = prepare_partial(destination, shard)
    url = download_url(endpoint, shard.path)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            transfer_once(url, partial, shard.size, timeout)
            actual_sha256 = sha256_file(partial)
            if actual_sha256 != shard.sha256:
                partial.unlink(missing_ok=True)
                raise IOError(
                    f"SHA-256 mismatch for {shard.path}: {actual_sha256} != {shard.sha256}"
                )
            partial.replace(destination)
            state.add(shard)
            return shard.path, "downloaded"
        except OSError as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(30, attempt * 2))
    raise RuntimeError(f"failed after {retries} attempts: {shard.path}: {last_error}")


def print_summary(shards: list[Shard]) -> None:
    buckets: dict[str, list[Shard]] = {}
    for shard in shards:
        buckets.setdefault(shard.bucket, []).append(shard)
    for bucket in sorted(buckets):
        rows = buckets[bucket]
        print(f"{bucket:16s} files={len(rows):4d} size={human_bytes(sum(x.size for x in rows))}")
    print(f"TOTAL            files={len(shards):4d} size={human_bytes(sum(x.size for x in shards))}")


def require_external_root(data_root: Path) -> None:
    """Refuse dataset output on the same filesystem as the user's home directory."""
    probe = data_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if probe.stat().st_dev == Path.home().stat().st_dev:
        raise SystemExit(
            f"refusing --out on the system volume: {data_root}\n"
            "Choose a mounted external volume for MER-17M."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Explicit MER-17M data root (use external storage)")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--bucket", action="append", help="Download only this token bucket; repeatable")
    parser.add_argument("--path", action="append", help="Download only this exact shard path; repeatable")
    parser.add_argument("--max-files", type=int, default=0, help="Limit selected files for a pilot run")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.out).expanduser().resolve()
    require_external_root(data_root)
    info, shards = fetch_manifest(args.endpoint, args.timeout)
    if args.bucket:
        wanted = set(args.bucket)
        shards = [shard for shard in shards if shard.bucket in wanted]
    if args.path:
        wanted_paths = set(args.path)
        shards = [shard for shard in shards if shard.path in wanted_paths]
        missing_paths = wanted_paths - {shard.path for shard in shards}
        if missing_paths:
            raise SystemExit(f"unknown shard path(s): {', '.join(sorted(missing_paths))}")
    if args.max_files:
        shards = shards[: args.max_files]
    if not shards:
        raise SystemExit("no shards selected")
    print_summary(shards)
    if args.dry_run:
        return 0

    metadata_dir = data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "hf-repo-info.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")

    state = VerificationState(metadata_dir / "verified.jsonl")
    workers = max(1, min(args.workers, len(shards)))
    errors: list[str] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_one,
                shard,
                data_root,
                args.endpoint,
                args.timeout,
                args.retries,
                state,
            ): shard
            for shard in shards
        }
        for future in concurrent.futures.as_completed(futures):
            shard = futures[future]
            try:
                path, result = future.result()
                completed += 1
                print(f"[{completed}/{len(shards)}] {result:10s} {path}", flush=True)
            except Exception as error:  # keep other transfers alive
                message = f"{shard.path}: {error}"
                errors.append(message)
                print(f"[error] {message}", flush=True)

    if errors:
        print(f"download_mer17m_failed errors={len(errors)}")
        return 1
    print(f"download_mer17m_ok root={data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

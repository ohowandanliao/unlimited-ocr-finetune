#!/usr/bin/env python3
"""Download a complete public olmOCR 2 dataset to external storage.

The downloader reads the Hugging Face repository manifest directly, keeps
partials and verification state under --out, and does not use the Hugging
Face cache. LFS objects are verified with SHA-256; regular Git objects are
verified with their Git blob SHA-1.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path, PurePosixPath


REPOSITORIES = {
    "lightonocr": "lightonai/LightOnOCR-mix-0126",
    "marcodsn": "marcodsn/arxiv-markdown",
    "mix": "allenai/olmOCR-mix-1025",
    "pdfa": "pixparse/pdfa-eng-wds",
    "synthmix": "allenai/olmOCR-synthmix-1025",
}
DEFAULT_ENDPOINT = "https://hf-mirror.com"
CHUNK_SIZE = 4 * 1024 * 1024
CURL = shutil.which("curl")


@dataclasses.dataclass(frozen=True)
class RepoFile:
    path: str
    size: int
    hash_kind: str
    digest: str

    @property
    def identity(self) -> str:
        return f"{self.hash_kind}:{self.digest}"

    @property
    def group(self) -> str:
        parts = self.path.split("/", 1)
        return parts[0] if len(parts) > 1 else "(root)"


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def curl_base(timeout: int, proxy: str | None) -> list[str]:
    if CURL is None:
        raise RuntimeError("curl is required but was not found on PATH")
    command = [
        CURL,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--max-time",
        str(timeout),
    ]
    if proxy:
        command.extend(["--proxy", proxy])
    return command


def request_bytes(url: str, timeout: int, proxy: str | None) -> bytes:
    result = subprocess.run(
        [*curl_base(timeout, proxy), url],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"curl failed ({result.returncode}): {detail}")
    return result.stdout


def fetch_manifest(
    endpoint: str,
    repo: str,
    timeout: int,
    proxy: str | None,
) -> tuple[dict, list[RepoFile]]:
    url = f"{endpoint.rstrip('/')}/api/datasets/{repo}?blobs=true"
    info = json.loads(request_bytes(url, timeout, proxy))
    files: list[RepoFile] = []
    for item in info.get("siblings", []):
        path = str(item.get("rfilename", ""))
        safe_path = PurePosixPath(path)
        if not path or safe_path.is_absolute() or ".." in safe_path.parts:
            raise ValueError(f"unsafe repository path: {path!r}")

        lfs = item.get("lfs") or {}
        if lfs:
            size = int(lfs.get("size") or 0)
            hash_kind = "sha256"
            digest = str(lfs.get("sha256") or "")
            valid_digest = len(digest) == 64
        else:
            size = int(item.get("size") or 0)
            hash_kind = "git-sha1"
            digest = str(item.get("blobId") or "")
            valid_digest = len(digest) == 40
        if size < 0 or not valid_digest:
            raise ValueError(f"missing blob metadata for {path!r}")
        files.append(
            RepoFile(
                path=path,
                size=size,
                hash_kind=hash_kind,
                digest=digest,
            )
        )

    files.sort(key=lambda item: item.path)
    if not files:
        raise RuntimeError("repository manifest contains no files")
    return info, files


def digest_file(path: Path, item: RepoFile) -> str:
    if item.hash_kind == "sha256":
        digest = hashlib.sha256()
    elif item.hash_kind == "git-sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {item.size}\0".encode("ascii"))
    else:
        raise ValueError(f"unsupported hash kind: {item.hash_kind}")
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
                    identity = f"{row.get('hash_kind', '')}:{row.get('digest', '')}"
                    self.verified[str(row.get("path", ""))] = identity

    def contains(self, item: RepoFile) -> bool:
        return self.verified.get(item.path) == item.identity

    def add(self, item: RepoFile) -> None:
        row = {
            "path": item.path,
            "size": item.size,
            "hash_kind": item.hash_kind,
            "digest": item.digest,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        encoded = json.dumps(row, ensure_ascii=True, sort_keys=True)
        with self.lock:
            if self.contains(item):
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
            self.verified[item.path] = item.identity


def prepare_partial(destination: Path, item: RepoFile) -> Path:
    partial = destination.with_name(destination.name + ".part")
    if destination.exists() and destination.stat().st_size != item.size:
        if destination.stat().st_size < item.size:
            if not partial.exists() or destination.stat().st_size > partial.stat().st_size:
                partial.unlink(missing_ok=True)
                destination.replace(partial)
            else:
                destination.unlink()
        else:
            destination.unlink()
    if partial.exists() and partial.stat().st_size > item.size:
        partial.unlink()
    return partial


def download_url(endpoint: str, repo: str, revision: str, path: str) -> str:
    quoted_path = urllib.parse.quote(path, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    return (
        f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/"
        f"{quoted_revision}/{quoted_path}?download=true"
    )


def transfer_once(
    url: str,
    partial: Path,
    expected_size: int,
    timeout: int,
    proxy: str | None,
) -> None:
    result = subprocess.run(
        [
            *curl_base(timeout, proxy),
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
    item: RepoFile,
    data_root: Path,
    endpoint: str,
    repo: str,
    revision: str,
    timeout: int,
    retries: int,
    proxy: str | None,
    state: VerificationState,
) -> tuple[str, str]:
    destination = data_root / PurePosixPath(item.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == item.size:
        if state.contains(item) or digest_file(destination, item) == item.digest:
            state.add(item)
            return item.path, "verified"
        destination.unlink()

    partial = prepare_partial(destination, item)
    url = download_url(endpoint, repo, revision, item.path)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            transfer_once(url, partial, item.size, timeout, proxy)
            actual_digest = digest_file(partial, item)
            if actual_digest != item.digest:
                partial.unlink(missing_ok=True)
                raise IOError(
                    f"{item.hash_kind} mismatch for {item.path}: "
                    f"{actual_digest} != {item.digest}"
                )
            partial.replace(destination)
            state.add(item)
            return item.path, "downloaded"
        except OSError as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(30, attempt * 2))
    raise RuntimeError(f"failed after {retries} attempts: {item.path}: {last_error}")


def print_summary(files: list[RepoFile]) -> None:
    groups: dict[str, list[RepoFile]] = {}
    for item in files:
        groups.setdefault(item.group, []).append(item)
    for group in sorted(groups):
        rows = groups[group]
        print(f"{group:16s} files={len(rows):6d} size={human_bytes(sum(x.size for x in rows))}")
    print(f"TOTAL            files={len(files):6d} size={human_bytes(sum(x.size for x in files))}")


def require_external_root(data_root: Path) -> None:
    probe = data_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if probe.stat().st_dev == Path.home().stat().st_dev:
        raise SystemExit(
            f"refusing --out on the system volume: {data_root}\n"
            "Choose a mounted external volume for olmOCR data."
        )


def existing_parent(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def remaining_transfer_bytes(data_root: Path, files: list[RepoFile]) -> int:
    remaining = 0
    for item in files:
        destination = data_root / PurePosixPath(item.path)
        partial = destination.with_name(destination.name + ".part")
        reusable = 0
        for candidate in (destination, partial):
            if candidate.exists():
                size = candidate.stat().st_size
                if size <= item.size:
                    reusable = max(reusable, size)
        remaining += item.size - reusable
    return remaining


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(REPOSITORIES), default="mix")
    parser.add_argument("--out", required=True, help="Explicit external dataset directory")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--proxy", default=None, help="Optional curl proxy URL")
    parser.add_argument(
        "--include",
        action="append",
        help="Keep paths matching this shell-style pattern; repeatable",
    )
    parser.add_argument("--path", action="append", help="Download only this exact path; repeatable")
    parser.add_argument("--max-files", type=int, default=0, help="Limit files for a pilot run")
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=100.0,
        help="Refuse to start if selected transfers would leave less free space",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.out).expanduser().resolve()
    require_external_root(data_root)
    repo = REPOSITORIES[args.dataset]
    info, files = fetch_manifest(args.endpoint, repo, args.timeout, args.proxy)
    revision = str(info.get("sha") or "")
    if len(revision) != 40:
        raise SystemExit("repository manifest did not include a commit revision")

    if args.path:
        wanted_paths = set(args.path)
        available_paths = {item.path for item in files}
        missing_paths = wanted_paths - available_paths
        if missing_paths:
            raise SystemExit(f"unknown repository path(s): {', '.join(sorted(missing_paths))}")
        files = [item for item in files if item.path in wanted_paths]
    if args.include:
        files = [
            item
            for item in files
            if any(fnmatch.fnmatchcase(item.path, pattern) for pattern in args.include)
        ]
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit("no files selected")

    print(f"repo={repo} revision={revision}")
    print_summary(files)
    if args.dry_run:
        return 0

    required_bytes = remaining_transfer_bytes(data_root, files)
    free_bytes = shutil.disk_usage(existing_parent(data_root)).free
    reserve_bytes = int(args.min_free_gib * 1024**3)
    print(
        f"preflight required={human_bytes(required_bytes)} "
        f"free={human_bytes(free_bytes)} reserve={human_bytes(reserve_bytes)}"
    )
    if required_bytes + reserve_bytes > free_bytes:
        raise SystemExit(
            "refusing download: selected files would cross the minimum free-space reserve"
        )

    metadata_dir = data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "hf-repo-info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")

    state = VerificationState(metadata_dir / "verified.jsonl")
    workers = max(1, min(args.workers, len(files)))
    errors: list[str] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_one,
                item,
                data_root,
                args.endpoint,
                repo,
                revision,
                args.timeout,
                args.retries,
                args.proxy,
                state,
            ): item
            for item in files
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                path, result = future.result()
                completed += 1
                print(f"[{completed}/{len(files)}] {result:10s} {path}", flush=True)
            except Exception as error:
                message = f"{item.path}: {error}"
                errors.append(message)
                print(f"[error] {message}", flush=True)

    if errors:
        print(f"download_hf_repo_failed dataset={args.dataset} errors={len(errors)}")
        return 1
    print(f"download_hf_repo_ok dataset={args.dataset} root={data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

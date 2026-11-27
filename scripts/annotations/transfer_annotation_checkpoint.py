"""Move the annotation run tree through a private Hugging Face dataset.

    python scripts/annotations/transfer_annotation_checkpoint.py upload
    python scripts/annotations/transfer_annotation_checkpoint.py download --revision COMMIT_SHA

Pause annotation workers, supervisors, and other writers for the duration of
packaging, or pass --root pointing to a read-only filesystem snapshot. The
archive includes inputs, queues, runtime copies, partial work and results for
all runs. Model weights and services installed outside the run tree are separate.

Download on the new machine into an absent or empty directory. Use the original
absolute run path when resuming workers because queue files contain absolute
paths. Temporary files are placed alongside --root by default. For a read-only
snapshot, upload accepts --work-dir pointing to writable staging space. Requires
GNU tar, zstd, Python with tarfile.data_filter (3.10.12+), huggingface_hub and
python-dotenv. Authentication uses HF_TOKEN, the checkout's .env, or hf auth login.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

DEFAULT_ROOT = Path("/data/martino/corral-annotations")
DEFAULT_REPO = "MrtinoRG/corral-annotations"
ARCHIVE = "checkpoint.tar.zst"
MANIFEST = "checkpoint.json"
FORMAT = "corral-annotation-checkpoint-v1"
EXCLUDED = {
    ".aws",
    ".git",
    ".netrc",
    ".ssh",
    "__pycache__",
    "access.txt",
    "review-token",
    "stored_tokens",
    "token",
}
LOG = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dirs(root):
    runs = sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and not p.is_symlink()
        and (p / "manifest.json").is_file()
        and any((p / stage).is_dir() for stage in ("graphs", "checks", "combined"))
    )
    if not runs:
        raise ValueError(f"No annotation runs found in {root}")
    return runs


def internal_link(link, target, source_root):
    """Return an archive-relative symlink target, including for a mounted snapshot."""
    archive_path = Path(link)
    target_path = Path(target)
    if target_path.is_absolute():
        for base in (source_root, DEFAULT_ROOT):
            if target_path.is_relative_to(base):
                target_path = target_path.relative_to(base)
                break
        else:
            raise ValueError(f"Link points outside the annotation tree: {link}")
    else:
        target_path = archive_path.parent / target_path
    target_path = Path(os.path.normpath(target_path))
    if ".." in target_path.parts:
        raise ValueError(f"Link escapes the annotation tree: {link}")
    return os.path.relpath(target_path, archive_path.parent)


def inventory(root, listing=None):
    """Fingerprint membership and file metadata so a changing tree cannot be published."""
    digest = hashlib.sha256()

    def visit(path):
        name = path.name.lower()
        if (
            name in EXCLUDED
            or name == ".env"
            or name.startswith(".env.")
            or name.endswith(".env")
        ):
            return
        rel = path.relative_to(root)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            internal_link(rel, path.readlink(), root)
        elif not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"Unsupported file in annotation tree: {path}")
        record = os.fsencode(rel) + b"\0"
        if listing is not None:
            listing.write(record)
        digest.update(record)
        digest.update(
            f"{info.st_ino}/{info.st_mode}/{info.st_size}/{info.st_ctime_ns}\n".encode()
        )
        if stat.S_ISDIR(info.st_mode):
            for child in sorted(path.iterdir()):
                visit(child)

    for run in run_dirs(root):
        visit(run)
    return digest.digest()


def artifact(root, directory, key):
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError(f"Invalid artifact fingerprint in {directory}")
    path = directory / f"{key}.json"
    if path.is_symlink():
        # Resolve a mounted snapshot's absolute artifact links inside the snapshot.
        path = path.parent / internal_link(
            path.relative_to(root), path.readlink(), root
        )
    if not path.is_file():
        raise ValueError(f"Missing annotation artifact: {path}")
    return path


def verify_results(root):
    """Check finished graphs against receipts and follow per-trace artifact references."""
    totals = {}
    for run in run_dirs(root):
        counts = {}
        graphs = run / "graphs"
        if graphs.is_dir():
            counts["graphs"] = 0
            for receipt in (graphs / "completed").glob("*.json"):
                record = json.loads(receipt.read_bytes())
                if record["state"] in {"complete", "requires_review"}:
                    graph_path = Path(record["graph"])
                    if graph_path.is_absolute() or ".." in graph_path.parts:
                        raise ValueError(f"Invalid graph path in {receipt}")
                    graph = graphs / graph_path
                    if sha256(graph) != record["graph_sha256"]:
                        raise ValueError(f"Graph checksum mismatch: {graph}")
                    counts["graphs"] += 1
        for stage in ("checks", "combined"):
            batch = run / stage
            if not batch.is_dir():
                continue
            for kind in ("labels", "evidence"):
                count = 0
                for receipt in (batch / "completed" / kind).glob("*.json"):
                    record = json.loads(receipt.read_bytes())
                    if record["state"] not in {"complete", "requires_review"}:
                        continue
                    gpu = f"gpu-{record['gpu']}"
                    if Path(gpu).name != gpu:
                        raise ValueError(f"Invalid GPU in {receipt}")
                    directory = batch / "artifacts" / gpu
                    saved = json.loads(
                        artifact(root, directory / "runs", record["run"]).read_bytes()
                    )
                    for group in ("labels", "evidence"):
                        for key in saved[group].values():
                            artifact(root, directory / group, key)
                    count += 1
                counts[f"{stage}/{kind}"] = count
        totals[run.name] = counts
    return totals


def upload(api, repo, root, work):
    if any(work.is_relative_to(run) for run in run_dirs(root)):
        raise ValueError("Choose a staging directory outside the runs being packaged.")
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    head = api.repo_info(repo, repo_type="dataset")
    if head.private is not True:
        raise ValueError(f"Upload requires a private Hugging Face dataset: {repo}")
    listing = work / "files"
    with listing.open("wb") as entries:
        before = inventory(root, entries)
    results = verify_results(root)
    archive = work / ARCHIVE
    LOG.info("Packaging annotation runs from %s", root)
    subprocess.run(
        [
            "tar",
            "--create",
            "--zstd",
            "--file",
            archive,
            "--directory",
            root,
            "--no-recursion",
            "--null",
            "--files-from",
            listing,
        ],
        check=True,
    )
    if before != inventory(root):
        raise ValueError(
            "Annotation files changed during packaging. Retry from a quiescent tree."
        )
    manifest = {
        "format": FORMAT,
        "source_root": str(root),
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
        "results": results,
    }
    (work / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    if api.repo_info(repo, repo_type="dataset").private is not True:
        raise ValueError(f"Upload requires a private Hugging Face dataset: {repo}")
    commit = api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        parent_commit=head.sha,
        commit_message="Save annotation checkpoint",
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=work / name)
            for name in (ARCHIVE, MANIFEST)
        ],
    )
    LOG.info(
        "Uploaded %s at revision %s\n%s",
        repo,
        commit.oid,
        json.dumps(results, indent=2),
    )


def unpack(archive, destination, source_root):
    def safe_member(member, directory):
        if member.issym():
            member = member.replace(
                linkname=internal_link(member.name, member.linkname, source_root)
            )
        return tarfile.data_filter(member, directory)

    with subprocess.Popen(
        ["zstd", "-dc", "--", archive.resolve(strict=True)], stdout=subprocess.PIPE
    ) as decoder:
        with decoder.stdout as stream, tarfile.open(fileobj=stream, mode="r|") as files:
            files.errorlevel = 2
            files.extractall(destination, filter=safe_member)
            # Read through the frame trailer so zstd verifies its checksum and exits.
            while stream.read(1024 * 1024):
                pass
        if decoder.wait() != 0:
            raise ValueError(f"Failed to decompress {archive.name}")


def download(api, repo, root, work, revision):
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"Restore requires an absent or empty directory: {root}")
    info = api.repo_info(repo, repo_type="dataset", revision=revision)
    if info.private is not True or not info.sha:
        raise ValueError(f"Expected a private dataset with a commit: {repo}")

    def fetch(name):
        return Path(
            hf_hub_download(
                repo,
                name,
                repo_type="dataset",
                revision=info.sha,
                cache_dir=work / "cache",
                token=api.token,
            )
        )

    manifest = json.loads(fetch(MANIFEST).read_bytes())
    if manifest["format"] != FORMAT or not Path(manifest["source_root"]).is_absolute():
        raise ValueError("Unrecognized annotation checkpoint manifest")
    archive = fetch(ARCHIVE)
    if (
        archive.stat().st_size != manifest["bytes"]
        or sha256(archive) != manifest["sha256"]
    ):
        raise ValueError("Downloaded checkpoint failed SHA-256 or size verification")
    staged = work / "restored"
    staged.mkdir(mode=0o700)
    unpack(archive, staged, Path(manifest["source_root"]))
    results = verify_results(staged)
    if results != manifest["results"]:
        raise ValueError(
            "Restored annotation counts differ from the checkpoint manifest"
        )
    # POSIX rename is atomic and refuses to replace a nonempty directory.
    staged.rename(root)
    LOG.info(
        "Restored %s at revision %s to %s\n%s",
        repo,
        info.sha,
        root,
        json.dumps(results, indent=2),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("upload", "download"):
        sub = commands.add_parser(command)
        sub.add_argument(
            "--repo-id",
            default=DEFAULT_REPO,
            help="Private HF dataset (default: %(default)s)",
        )
        sub.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_ROOT,
            help="Annotation run directory (default: %(default)s)",
        )
        if command == "download":
            sub.add_argument(
                "--revision",
                default="main",
                help="Commit SHA or branch to restore (default: main)",
            )
        else:
            sub.add_argument(
                "--work-dir",
                type=Path,
                help="Writable directory for packaging the upload",
            )
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api = HfApi(token=os.environ.get("HF_TOKEN") or True)
    root = args.root.expanduser().absolute()
    if args.command == "download":
        if not hasattr(tarfile, "data_filter"):
            parser.error(
                "Download requires Python with tarfile.data_filter (3.10.12+)."
            )
        if root.is_symlink():
            parser.error("Restore destination cannot be a symbolic link.")
        root.parent.mkdir(parents=True, exist_ok=True)
    workspace = root.parent
    if args.command == "upload" and args.work_dir:
        workspace = args.work_dir.expanduser().resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix=".corral-checkpoint-", dir=workspace
    ) as temp:
        if args.command == "upload":
            upload(api, args.repo_id, root.resolve(strict=True), Path(temp))
        else:
            download(api, args.repo_id, root, Path(temp), args.revision)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

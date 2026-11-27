"""Export selected source files and prepare a reviewed, atomic Git candidate."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
LEDGER = "handoff/core-ledger.json"
CORE = tuple(
    "lm_trainer/" + name
    for name in (
        "LICENSE",
        "README.md",
        "TASK_A_CONTRACT.md",
        "TASK_B_CONTRACT.md",
        "TASK_C_CONTRACT.md",
        "THIRD_PARTY_LICENSE.txt",
        "__init__.py",
        "__main__.py",
        "chat/__init__.py",
        "chat/__main__.py",
        "chat/backend.py",
        "chat/context.py",
        "chat/memory.py",
        "chat/registry.py",
        "chat/runtime.py",
        "chat/types.py",
        "config/__init__.py",
        "config/common.py",
        "config/stages.py",
        "config/train_config.py",
        "eval/__init__.py",
        "eval/__main__.py",
        "eval/blind.py",
        "eval/cards.py",
        "eval/provider.py",
        "eval/report.py",
        "eval/runner.py",
        "eval/schema.py",
        "generation.py",
        "model/__init__.py",
        "model/loader.py",
        "model/quantization.py",
        "requirements.txt",
        "template_policy.py",
        "tests/test_contract.py",
        "tests/test_hf.py",
        "tests/test_task_a.py",
        "tests/test_task_b.py",
        "tests/test_task_c.py",
        "tests/test_task_d.py",
        "tests/test_task_d_ui.py",
        "train/__init__.py",
        "train/artifacts.py",
        "train/bootstrap.py",
        "train/checkpoints.py",
        "train/data.py",
        "train/dev.py",
        "train/encoding.py",
        "train/engine.py",
        "train/io.py",
        "train/metrics.py",
        "train/modeling.py",
        "train/performance.py",
        "train/reproducibility.py",
        "train/sampling.py",
        "train/schedule.py",
        "train/schema.py",
        "train/stages.py",
        "webui/__init__.py",
        "webui/__main__.py",
        "webui/app.py",
        "webui/markdown_safe.py",
        "webui/state.py",
    )
)
EXAMPLE = tuple(
    "examples/industrial_support/" + name
    for name in (
        "__init__.py",
        "__main__.py",
        "README.md",
        "manuals.json",
        "train.yaml",
        "model.yaml",
        "questions.jsonl",
    )
) + (
    "product_assets/example/config/train.yaml",
    "product_assets/example/config/model.yaml",
    "product_assets/example/data/train/fictional_lamp.jsonl",
    "product_assets/example/data/eval/fictional_lamp.jsonl",
    "product_assets/example/assets/.gitkeep",
    "product_assets/example/checkpoints/.gitkeep",
    "product_assets/example/runs/.gitkeep",
)
DENY = re.compile(
    r"diary[_-]?(?:core|homo)|\bboa(?:\d+|\b|_)|日记|小飞鱼|人格卡|/home/jutze|/Users/a114514|PRIVATE_FIXTURE_SENTINEL|-----BEGIN [^\n]*PRIVATE KEY-----|\bhf_[A-Za-z0-9]{24,}",
    re.I,
)
OLD_ASSET_PATH = re.compile(
    r"data/(?:dataset|val|large_language_model)(?:/|\b)|checkpoint/old_ckpt|"
    r"generate/(?:input_prompt|output|validation)|debug/(?:memory|prompt)|"
    r"product_assets/(?:private|user)(?:/|\b)|webui/diary_outputs"
)
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def encoded(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def digest(data):
    return sha256(data).hexdigest()


@contextmanager
def directory(path):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("parent traversal is not allowed")
    fd = os.open("/", DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            new = os.open(part, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = new
        yield fd
    finally:
        os.close(fd)


@dataclass(frozen=True)
class File:
    data: bytes
    mode: int

    def signature(self):
        return [digest(self.data), self.mode]


def regular(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or str(rel) != name or "\\" in name:
        raise ValueError("invalid relative path")
    with directory(Path(root) / rel.parent) as parent:
        fd = os.open(
            rel.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_mode & 0o7000
                or before.st_size > 2_000_000
            ):
                raise ValueError("expected a regular source file under 2 MB")
            data = stream.read(2_000_001)
            after = os.fstat(fd)
            if (before.st_size, before.st_ctime_ns) != (len(data), after.st_ctime_ns):
                raise ValueError("file changed while reading")
            return File(data, stat.S_IMODE(before.st_mode))


def check_content(name, file):
    text = file.data.decode("utf-8")
    if "\x00" in text or DENY.search(text) or OLD_ASSET_PATH.search(text):
        raise ValueError(f"content review required for {name}")


def write_snapshot(destination, files):
    for name, file in files.items():
        check_content(name, file)
    destination = Path(destination).absolute()
    with directory(destination.parent):
        destination.mkdir(mode=0o700, exist_ok=False)
    for name, file in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(file.data)
        path.chmod(file.mode & 0o777)


def export(source, destination, with_example=False):
    names = CORE + (EXAMPLE if with_example else ())
    files = {}
    for name in names:
        item = regular(source, name)
        files[name] = File(item.data, 0o755 if item.mode & 0o111 else 0o644)
    write_snapshot(destination, files)
    return {name: file.signature() for name, file in files.items()}


def unknown_files(root, names):
    # List names only inside the shared package. Never enter unlisted directories.
    expected = {}
    for name in names:
        path = PurePosixPath(name)
        for parent in path.parents:
            if parent != PurePosixPath("."):
                expected.setdefault(str(parent), set()).add(
                    path.relative_to(parent).parts[0]
                )
    extras = []
    for rel, children in expected.items():
        try:
            with directory(Path(root) / rel) as fd:
                for entry in set(os.listdir(fd)) - children:
                    if entry == "__pycache__" and stat.S_ISDIR(
                        os.stat(entry, dir_fd=fd, follow_symlinks=False).st_mode
                    ):
                        continue
                    extras.append(str(PurePosixPath(rel) / entry))
        except (FileNotFoundError, NotADirectoryError):
            # The corresponding allowlisted leaves are already reported as invalid.
            continue
        except OSError:
            extras.append(rel + " (cannot inspect directory)")
    return sorted(extras)


def classify(old, source, target, approved):
    if source is None or target is None:
        return "review: missing or invalid type"
    past_src, past_dst = old["source"], old["target"]
    if past_dst and target[1] != past_dst[1] or past_src and source[1] != past_src[1]:
        return "review: mode changed"
    if bool(source[1] & 0o111) != bool(target[1] & 0o111):
        return "review: incompatible executable mode"
    if source[0] == target[0]:
        return "identical"
    if source == past_src:
        return (
            "upstream unchanged"
            if target == past_dst
            else "review: target drift; preserve"
        )
    if (
        old["policy"] != "copy"
        or not past_src
        or not past_dst
        or past_src[0] != past_dst[0]
    ):
        return "review: initial baseline or adaptation"
    if target != past_dst:
        return "conflict: both changed"
    if approved != source[0]:
        return "review: source bytes unapproved"
    return "copy"


def preview(source, target, approval, *, names=CORE, establish=False):
    ledger_bytes = regular(target, LEDGER).data
    ledger = json.loads(ledger_bytes)
    if not isinstance(approval, dict):
        raise ValueError("approval must be an object mapping paths to SHA-256")
    if set(ledger["files"]) != set(names) or set(approval) - set(names):
        raise ValueError("ledger or approval outside the group allowlist")
    if any(
        not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
        for v in approval.values()
    ):
        raise ValueError("approval must pin SHA-256 bytes")
    rows, selected = {}, {}
    for name in names:
        files = []
        for root in (source, target):
            try:
                item = regular(root, name)
                check_content(name, item)
            except (OSError, ValueError):
                item = None
            files.append(item)
        src, dst = [f.signature() if f is not None else None for f in files]
        action = classify(ledger["files"][name], src, dst, approval.get(name))
        if (
            establish
            and action == "identical"
            and approval.get(name) == src[0]
            and (
                ledger["files"][name]["policy"] != "copy"
                or ledger["files"][name]["source"] != src
            )
        ):
            action = "enroll matched bytes"
        rows[name] = {"source": src, "target": dst, "action": action}
        if files[1] is not None:
            selected[name] = (
                File(files[0].data, files[1].mode) if action == "copy" else files[1]
            )
    unexpected = {
        side: unknown_files(path, names)
        for side, path in [("source", source), ("target", target)]
    }
    blocked = any(unexpected.values()) or any(
        r["action"].startswith(("review", "conflict")) for r in rows.values()
    )
    body = dict(
        source=str(Path(source).absolute()),
        target=str(Path(target).absolute()),
        files=rows,
        unexpected=unexpected,
        blocked=blocked,
        ledger_sha256=digest(ledger_bytes),
        approval=approval,
        establish=establish,
    )
    body["fingerprint"] = digest(encoded(body))
    return body, selected, ledger


def git(root, *args, data=None, env=None):
    return (
        subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            input=data,
            stdout=subprocess.PIPE,
            check=True,
            env=env,
        )
        .stdout.decode()
        .strip()
    )


def run_core_tests(snapshot):
    scratch = snapshot.parent / "runtime"
    scratch.mkdir()
    env = {
        k: os.environ[k] for k in ("PATH", "LANG", "LD_LIBRARY_PATH") if k in os.environ
    }
    env.update(
        HOME=str(scratch),
        TMPDIR=str(scratch),
        HF_HOME=str(scratch / "hf"),
        MPLCONFIGDIR=str(scratch / "mpl"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        ACCELERATE_USE_CPU="true",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONNOUSERSITE="1",
    )
    targets = (
        ["lm_trainer/tests"]
        if (snapshot / "examples" / "vision_support").is_dir()
        else ["lm_trainer/tests/test_contract.py", "lm_trainer/tests/test_hf.py"]
    )
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            *targets,
        ],
        cwd=snapshot,
        env=env,
        check=True,
    )


def candidate(
    source,
    target,
    get_approval,
    expected,
    *,
    names=CORE,
    test=run_core_tests,
    establish=False,
):
    target = Path(target).absolute()
    if git(target, "rev-parse", "--show-toplevel") != str(target):
        raise ValueError("target must be the root of a linked worktree")
    actual = git(target, "rev-parse", "--absolute-git-dir")
    common = git(target, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if actual == common:
        raise ValueError("use a linked review worktree")
    base = git(target, "rev-parse", "HEAD")

    def verify():
        result = preview(
            source, target, get_approval(), names=names, establish=establish
        )
        plan = result[0]
        if plan["blocked"] or plan["fingerprint"] != expected:
            raise ValueError("blocked group or stale preview")
        if git(target, "rev-parse", "HEAD") != base or git(
            target, "status", "--porcelain", "--", LEDGER, *names
        ):
            raise ValueError(
                "commit or review local group edits before preparing a candidate"
            )
        return result

    plan, files, ledger = verify()
    edits = {n: files[n] for n, row in plan["files"].items() if row["action"] == "copy"}
    enrolled = {
        n for n, row in plan["files"].items() if row["action"] == "enroll matched bytes"
    }
    if not edits and not enrolled:
        return "no changes"
    with TemporaryDirectory(prefix="core-review-") as tmp:
        scratch = Path(tmp)
        write_snapshot(scratch / "source", files)
        test(scratch / "source")
        if any(
            regular(scratch / "source", n) != f for n, f in files.items()
        ) or unknown_files(scratch / "source", names):
            raise ValueError("tested source snapshot changed")
        verify()
        source_hash = digest(
            encoded({n: row["source"] for n, row in plan["files"].items()})
        )
        for name, row in plan["files"].items():
            if (
                name in edits
                or name in enrolled
                or (
                    row["action"] == "identical"
                    and ledger["files"][name]["policy"] == "copy"
                    and plan["approval"].get(name) == row["source"][0]
                )
            ):
                ledger["files"][name].update(
                    policy="copy",
                    source=row["source"],
                    target=files[name].signature(),
                    accepted_snapshot=source_hash,
                )
        ledger["last_acceptance"] = {
            "source_snapshot": source_hash,
            "preview": expected,
            "validation": "clean CPU suite passed",
        }
        env = {**os.environ, "GIT_INDEX_FILE": str(scratch / "index")}
        git(target, "read-tree", base, env=env)
        edits[LEDGER] = File(encoded(ledger), regular(target, LEDGER).mode)
        for name, file in edits.items():
            blob = git(target, "hash-object", "-w", "--stdin", data=file.data)
            mode = "100755" if file.mode & 0o111 else "100644"
            git(
                target,
                "update-index",
                "--add",
                "--cacheinfo",
                mode,
                blob,
                name,
                env=env,
            )
        tree = git(target, "write-tree", env=env)
        commit = git(
            target,
            "-c",
            "user.name=Core snapshot review",
            "-c",
            "user.email=core-review@localhost",
            "commit-tree",
            tree,
            "-p",
            base,
            "-m",
            "Accept tested shared source snapshot",
        )
        verify()
        branch = "review/core-" + expected[:16]
        ref = "refs/heads/" + branch
        existing = git(target, "for-each-ref", "--format=%(objectname)", ref)
        if existing:
            if (
                git(target, "rev-parse", existing + "^{tree}") != tree
                or git(target, "rev-parse", existing + "^") != base
            ):
                raise ValueError(
                    "review reference already exists with different contents"
                )
        else:
            git(target, "update-ref", ref, commit, "0" * len(base))
        return branch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    out = sub.add_parser("export")
    out.add_argument("destination", type=Path)
    out.add_argument("--with-example", action="store_true")
    for command in ("preview", "candidate"):
        cmd = sub.add_parser(command)
        cmd.add_argument(
            "--establish",
            action="store_true",
            help="explicitly enroll approved byte-identical files for future copying",
        )
        cmd.add_argument("--source", required=True, type=Path)
        cmd.add_argument("--target", default=ROOT, type=Path)
        cmd.add_argument(
            "--approval",
            type=Path,
            help="JSON object mapping approved paths to exact SHA-256",
        )
        if command == "candidate":
            cmd.add_argument("--expect", required=True, help="fingerprint from preview")
    argv = sys.argv[1:]
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["preview", *argv]
    args = parser.parse_args(argv)
    if args.command == "export":
        print(digest(encoded(export(ROOT, args.destination, args.with_example))))
        return

    def approval():
        if args.approval is None:
            return {}
        return json.loads(regular(args.approval.parent, args.approval.name).data)

    if args.command == "preview":
        plan = preview(args.source, args.target, approval(), establish=args.establish)[
            0
        ]
        print(encoded(plan).decode(), end="")
        return 2 if plan["blocked"] else 0
    print(
        candidate(
            args.source, args.target, approval, args.expect, establish=args.establish
        )
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))

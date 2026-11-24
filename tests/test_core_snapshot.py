import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import core_snapshot as sync


NAMES = ("lm_trainer/value.py", "lm_trainer/reader.py")
FIRST, SECOND = NAMES


def put(root, name, data):
    path = root / name
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(data)
    return path


def current(root, name):
    return sync.regular(root, name).signature()


def check(snapshot):
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from lm_trainer.reader import read; assert read() > 0",
        ],
        cwd=snapshot,
        check=True,
    )


@pytest.fixture
def group(tmp_path):
    repo, wt, source = (tmp_path / part for part in ("repo", "wt", "source"))
    repo.mkdir()
    source.mkdir()
    sync.git(repo, "init", "-q")
    sync.git(repo, "config", "user.name", "Synthetic QA")
    sync.git(repo, "config", "user.email", "qa@example.invalid")
    for root in (repo, source):
        put(root, FIRST, b"VALUE = 1\n")
        put(
            root, SECOND, b"from .value import VALUE\n\ndef read():\n    return VALUE\n"
        )
    rows = {
        n: dict(policy="copy", source=current(source, n), target=current(repo, n))
        for n in NAMES
    }
    put(repo, sync.LEDGER, sync.encoded(dict(files=rows)))
    sync.git(repo, "add", ".")
    sync.git(repo, "commit", "-qm", "fixture")
    sync.git(repo, "worktree", "add", "--detach", str(wt), "HEAD")
    return source, wt


def plan(s, t, allow=None, establish=False):
    return sync.preview(s, t, allow or {}, names=NAMES, establish=establish)[0]


def accept(s, t, allow, p, test=check, establish=False):
    return sync.candidate(
        s,
        t,
        lambda: allow,
        p["fingerprint"],
        names=NAMES,
        test=test,
        establish=establish,
    )


def test_repeated_import_and_committed_target_drift(group):
    s, t = group
    assert {r["action"] for r in plan(s, t)["files"].values()} == {"identical"}
    assert accept(s, t, {}, plan(s, t)) == "no changes"
    put(s, FIRST, b"VALUE = 2\n")
    assert plan(s, t)["blocked"]
    approval = {FIRST: current(s, FIRST)[0]}
    p = plan(s, t, approval)
    assert not p["blocked"] and p["files"][FIRST]["action"] == "copy"
    previous, ledger = sync.git(t, "rev-parse", "HEAD"), (t / sync.LEDGER).read_bytes()
    branch = accept(s, t, approval, p)
    assert (
        sync.git(t, "rev-parse", "HEAD") == previous
        and (t / sync.LEDGER).read_bytes() == ledger
    )
    assert accept(s, t, approval, p) == branch
    assert set(sync.git(t, "diff", "--name-only", "HEAD", branch).splitlines()) == {
        FIRST,
        sync.LEDGER,
    }
    sync.git(t, "merge", "--ff-only", branch)
    assert current(s, FIRST) == current(t, FIRST)
    assert accept(s, t, approval, plan(s, t, approval)) == "no changes"
    put(t, FIRST, b"VALUE = 3\n")
    sync.git(t, "commit", "-qam", "local edit")
    assert (
        plan(s, t, approval)["files"][FIRST]["action"]
        == "review: target drift; preserve"
    )
    put(s, FIRST, b"VALUE = 4\n")
    approval[FIRST] = current(s, FIRST)[0]
    assert plan(s, t, approval)["files"][FIRST]["action"] == "conflict: both changed"


@pytest.mark.parametrize(
    "kind",
    [
        "deleted",
        "renamed",
        "file-link",
        "dir-link",
        "new",
        "mode",
        "fifo",
        "target-link",
        "target-missing",
    ],
)
def test_group_validation_never_partially_imports(group, kind):
    s, t = group
    put(s, FIRST, b"VALUE = 2\n")
    allow = {FIRST: current(s, FIRST)[0]}
    path = s / SECOND
    if kind in ("deleted", "renamed", "file-link", "fifo"):
        path.rename(s / ("outside" if kind != "renamed" else "lm_trainer/renamed.py"))
    if kind == "file-link":
        path.symlink_to(s / "outside")
    if kind == "fifo":
        os.mkfifo(path)
    if kind == "new":
        put(s, "lm_trainer/new.py", b"NEW = 1\n")
    if kind == "mode":
        path.chmod(0o755)
    if kind == "dir-link":
        (s / "lm_trainer").rename(s / "alternate")
        (s / "lm_trainer").symlink_to(s / "alternate", target_is_directory=True)
    if kind.startswith("target-"):
        (t / SECOND).rename(t / "outside")
        if kind == "target-link":
            (t / SECOND).symlink_to(t / "outside")
    p = plan(s, t, allow)
    original = (t / sync.LEDGER).read_bytes()
    assert p["blocked"]
    with pytest.raises(ValueError):
        accept(s, t, allow, p)
    assert (t / FIRST).read_text() == "VALUE = 1\n" and (
        t / sync.LEDGER
    ).read_bytes() == original


def test_adapted_unapproved_and_explicit_first_baseline(group):
    s, t = group
    file = t / sync.LEDGER
    ledger = json.loads(file.read_bytes())
    ledger["files"][FIRST].update(policy="adapted", source=None)
    file.write_bytes(sync.encoded(ledger))
    sync.git(t, "commit", "-qam", "mark adapted")
    put(s, FIRST, b"VALUE = 5\n")
    allow = {FIRST: current(s, FIRST)[0]}
    assert plan(s, t, allow)["files"][FIRST]["action"].startswith("review: initial")
    put(s, FIRST, (t / FIRST).read_bytes())
    allow[FIRST] = current(s, FIRST)[0]
    assert accept(s, t, allow, plan(s, t, allow)) == "no changes"
    p = plan(s, t, allow, establish=True)
    assert p["files"][FIRST]["action"] == "enroll matched bytes"
    branch = accept(s, t, allow, p, establish=True)
    sync.git(t, "merge", "--ff-only", branch)
    assert json.loads(file.read_bytes())["files"][FIRST]["policy"] == "copy"
    put(s, FIRST, b"VALUE = 8\n")
    assert plan(s, t, allow)["blocked"]


@pytest.mark.parametrize(
    "event",
    ["before", "during-source", "during-target", "during-approval", "test-failure"],
)
def test_frozen_bytes_and_failed_test_leave_approved_version(group, event):
    s, t = group
    put(s, FIRST, b"VALUE = 7\n")
    allow = {FIRST: current(s, FIRST)[0]}
    p = plan(s, t, allow)
    old = (t / sync.LEDGER).read_bytes()
    head = sync.git(t, "rev-parse", "HEAD")
    if event == "before":
        put(s, FIRST, b"VALUE = 8\n")

    def test(snapshot):
        check(snapshot)
        if event == "during-source":
            put(s, FIRST, b"VALUE = 9\n")
        if event == "during-target":
            put(t, SECOND, b"# concurrent local edit\n")
        if event == "during-approval":
            allow.clear()
        if event == "test-failure":
            raise subprocess.CalledProcessError(1, "synthetic test")

    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        accept(s, t, allow, p, test)
    assert (
        sync.git(t, "rev-parse", "HEAD") == head
        and (t / sync.LEDGER).read_bytes() == old
    )
    assert (
        sync.git(t, "for-each-ref", "--format=%(refname)", "refs/heads/review/") == ""
    )
    if event == "during-target":
        assert (t / SECOND).read_bytes() == b"# concurrent local edit\n"


def test_export_allowlist_content_and_no_history(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for name in sync.CORE + sync.EXAMPLE:
        put(root, name, sync.regular(sync.ROOT, name).data)
    for name in (
        "private/data.txt",
        ".git/config",
        "lm_trainer/notes.txt",
        "checkpoint/example.bin",
    ):
        put(root, name, b"PRIVATE_FIXTURE_SENTINEL")
    clean = tmp_path / "release"
    manifest = sync.export(root, clean)
    assert (
        set(manifest)
        == set(sync.CORE)
        == {p.relative_to(clean).as_posix() for p in clean.rglob("*") if p.is_file()}
    )
    assert all(
        b"PRIVATE_FIXTURE_SENTINEL" not in (clean / n).read_bytes() for n in manifest
    )
    name = sync.CORE[0]
    put(root, name, b"# PRIVATE_FIXTURE_SENTINEL\n")
    with pytest.raises(ValueError, match="content review"):
        sync.export(root, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    with pytest.raises(ValueError):
        sync.regular(root, "../private")
    root.joinpath(name).rename(root / "outside")
    root.joinpath(name).symlink_to(root / "outside")
    with pytest.raises(OSError):
        sync.export(root, tmp_path / "symlink")


def test_mixed_approval_and_reverting_accepted_group(group):
    s, t = group
    put(s, FIRST, b"VALUE = 3\n")
    put(s, SECOND, b"from .value import VALUE\n\ndef read():\n    return VALUE + 1\n")
    allow = {FIRST: current(s, FIRST)[0]}
    p = plan(s, t, allow)
    assert p["files"][FIRST]["action"] == "copy" and p["blocked"]
    with pytest.raises(ValueError):
        accept(s, t, allow, p)
    prior = sync.git(t, "rev-parse", "HEAD^{tree}")
    allow[SECOND] = current(s, SECOND)[0]
    branch = accept(s, t, allow, plan(s, t, allow))
    sync.git(t, "merge", "--ff-only", branch)
    sync.git(t, "revert", "--no-edit", "HEAD")
    assert sync.git(t, "rev-parse", "HEAD^{tree}") == prior


def test_complete_shared_group_using_production_gate(tmp_path):
    source, repo, target = [
        tmp_path / label for label in ("incoming", "repo", "review")
    ]
    sync.export(sync.ROOT, source)
    sync.export(sync.ROOT, repo)
    sync.git(repo, "init", "-q")
    sync.git(repo, "config", "user.name", "Synthetic QA")
    sync.git(repo, "config", "user.email", "qa@example.invalid")
    sync.git(repo, "add", ".")
    sync.git(repo, "commit", "-qm", "fixture")
    sync.git(repo, "worktree", "add", "--detach", str(target), "HEAD")
    rows = {
        n: dict(policy="copy", source=current(source, n), target=current(target, n))
        for n in sync.CORE
    }
    put(target, sync.LEDGER, sync.encoded(dict(files=rows)))
    sync.git(target, "add", sync.LEDGER)
    sync.git(target, "commit", "-qm", "accept fixture baseline")
    name = "lm_trainer/README.md"
    path = source / name
    path.write_bytes(path.read_bytes() + b"\nSynthetic acceptance update.\n")
    allow = {name: current(source, name)[0]}

    def prepare():
        plan = sync.preview(source, target, allow)[0]
        assert not plan["blocked"]
        return sync.candidate(source, target, lambda: allow, plan["fingerprint"])

    branch = prepare()
    sync.git(target, "merge", "--ff-only", branch)
    assert set(
        sync.git(target, "diff", "--name-only", "HEAD^", "HEAD").splitlines()
    ) == {name, sync.LEDGER}
    assert prepare() == "no changes"


def test_industrial_example_in_export(tmp_path):
    import os
    import subprocess
    import sys
    from tools import core_snapshot as sync

    dest = tmp_path / "independent"
    sync.export(sync.ROOT, dest, with_example=True)
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "PYTHONPATH" and not k.startswith("BOA_")
    }
    env["HOME"] = str(tmp_path)

    def run(*args):
        return subprocess.run(
            [sys.executable, "-B", "-m", *args],
            cwd=dest,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        ).stdout

    name = "examples.industrial_support"
    assert "XR17-A-1 §2" in run(name, "XR-17", "--revision", "A") and "青色" in run(
        name, "XR-17", "--revision", "A"
    )
    b = run(name, "XR-17", "--revision", "B")
    assert "琥珀色" in b and "XR17-B-2 §4" in b
    assert "请先确认" in run(name, "XR-17")
    for model, rev in [("XR-17", "C"), ("NO-MANUAL", "A")]:
        assert "证据不足" in run(name, model, "--revision", rev)
    cfg = json.loads(
        run(
            "lm_trainer",
            "--config",
            "examples/industrial_support/train.yaml",
            "--model-config",
            "examples/industrial_support/model.yaml",
            "--dry-run",
        )
    )
    assert (
        Path(cfg["data_file_path"])
        == dest / "examples/industrial_support/questions.jsonl"
        and cfg["backend"] == "hf"
    )
    assert not (dest / "diary_core").exists() and not (dest / "config").exists()

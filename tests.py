import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from shlex import quote

import pytest
from openreward.api.sandboxes.types import RunResult
from openreward.environments import JSONObject, ToolOutput

from swegym import MISSING_IMAGE_INSTANCE_IDS, BashParams, SWEGym

logger = logging.getLogger(__name__)

tasks = SWEGym.list_tasks("all")
EXAMPLE_SWE_GYM_TASK = tasks[0]


OPENREWARD_API_KEY = os.getenv("OPENREWARD_API_KEY", "")

@pytest.mark.skipif(not OPENREWARD_API_KEY, reason="OPENREWARD_API_KEY is not set")
@pytest.mark.asyncio
async def test_swe_gym_bash():
    env = SWEGym(task_spec=EXAMPLE_SWE_GYM_TASK, secrets={"OPENREWARD_API_KEY": OPENREWARD_API_KEY})
    try:
        await env.setup()

        output: ToolOutput = await env.bash(BashParams(command="whoami"))
        assert isinstance(output.metadata["output"], str)
        assert "root" in output.metadata["output"], f"Expected 'root' in output, got {output.metadata['output']}"
    finally:
        await env.teardown()


@pytest.mark.skipif(not OPENREWARD_API_KEY, reason="OPENREWARD_API_KEY is not set")
@pytest.mark.asyncio
@pytest.mark.parametrize("task", tasks)
async def test_swe_gym_golden_patch(task: JSONObject):
    env = SWEGym(task_spec=task, secrets={"OPENREWARD_API_KEY": OPENREWARD_API_KEY})
    try:
        await env.setup()
        assert env.computer is not None

        golden_patch = env.validated.patch
        await env.computer.check_run(f"echo {quote(golden_patch)} > /testbed/golden.patch")
        await env.computer.check_run(f"git apply /testbed/golden.patch")
        await env.computer.check_run(f"rm -f /testbed/golden.patch")

        res: ToolOutput = await env.answer()
        assert res.reward == 1, f"Expected reward of 1, got {res.reward}, full output: {res}"
        assert res.finished
    finally:
        await env.teardown()


@pytest.mark.skipif(not OPENREWARD_API_KEY, reason="OPENREWARD_API_KEY is not set")
@pytest.mark.asyncio
@pytest.mark.parametrize("task", tasks)
async def test_swe_gym_xfail_state(task: JSONObject):
    env = SWEGym(task_spec=task, secrets={"OPENREWARD_API_KEY": OPENREWARD_API_KEY})
    try:
        await env.setup()

        res: ToolOutput = await env.answer()
        assert res.reward == 0, f"Expected reward of 0, got {res.reward}, full output: {res}"
        assert res.finished
    finally:
        await env.teardown()


class _FakeComputer:
    """Answers the eval's sandbox calls without a sandbox."""

    async def check_run(self, cmd: str, **kwargs):
        return ""

    async def download(self, path: str):
        return b"diff --git a/f.py b/f.py\n" if path.endswith(".patch") else b"eval log"

    async def upload(self, local_path, container_path: str):
        pass

    async def run(self, cmd: str, **kwargs):
        return "", 0


# Needs no sandbox and no API key. Goes through _call_tool, the server path.
@pytest.mark.asyncio
@pytest.mark.parametrize("resolved", [True, False])
async def test_swe_gym_report_hides_test_names(resolved: bool, monkeypatch):
    env = SWEGym(task_spec=EXAMPLE_SWE_GYM_TASK, secrets={"api_key": "unused"})
    env.computer = _FakeComputer()
    f2p, p2p = list(env.test_spec.FAIL_TO_PASS), list(env.test_spec.PASS_TO_PASS)

    def fake_eval_report(test_spec, prediction, log_path, include_tests_status):
        return {test_spec.instance_id: {
            "patch_is_None": False, "patch_exists": True,
            "patch_successfully_applied": True, "resolved": resolved,
            "tests_status": {
                "FAIL_TO_PASS": {"success": f2p if resolved else [], "failure": [] if resolved else f2p},
                "PASS_TO_PASS": {"success": p2p, "failure": []},
            },
        }}

    monkeypatch.setattr("swegym.get_eval_report", fake_eval_report)
    out = (await env._call_tool("answer", {})).root.output
    assert out.reward == (1 if resolved else 0) and out.finished
    payload = json.dumps({
        "blocks": [b.model_dump() for b in out.blocks],
        "metadata": out.metadata, "reward": out.reward, "finished": out.finished,
    })
    for name in f2p + p2p:
        assert name not in payload
    counts = out.metadata["report"][env.test_spec.instance_id]["tests_status_counts"]
    assert counts["FAIL_TO_PASS"] == {"success": len(f2p) if resolved else 0,
                                      "failure": 0 if resolved else len(f2p)}


# Needs no sandbox and no API key.
def test_sandbox_blocks_network():
    # Every task comes from a public upstream pull request, so with network
    # access the agent could download the fix.
    env = SWEGym(task_spec=EXAMPLE_SWE_GYM_TASK, secrets={"api_key": "unused"})
    assert env.compute_settings.block_network is True
    # Proxy-aware tools fail at once instead of waiting for a connect timeout.
    sandbox_env = env.compute_settings.env or {}
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        assert sandbox_env.get(var, "").startswith("http://127.0.0.1:")


class _LogComputer(_FakeComputer):
    def __init__(self, log: bytes):
        self.log = log

    async def download(self, path: str):
        return b"diff --git a/f.py b/f.py\n" if path.endswith(".patch") else self.log


# Needs no sandbox and no API key. Runs the fork's real get_eval_report on a
# repo with capitals in its name, which its parser map keys in lowercase.
@pytest.mark.asyncio
async def test_swe_gym_grades_uppercase_repo():
    task = next(t for t in SWEGym.list_tasks("all") if t["repo"] == "Project-MONAI/MONAI")
    env = SWEGym(task_spec=task, secrets={"api_key": "unused"})
    tests = list(env.test_spec.FAIL_TO_PASS) + list(env.test_spec.PASS_TO_PASS)
    env.computer = _LogComputer("\n".join(f"PASSED {t}" for t in tests).encode())
    out = (await env._call_tool("answer", {})).root.output
    assert out.reward == 1 and out.finished



class LocalGitSandbox:
    """Runs commands with bash against a local directory standing in for the
    sandbox: `/testbed` is a real git repository. Like the SDK, `run` returns a
    RunResult, `check_run` raises on a non-zero exit and returns the output,
    and `upload`/`download` fail on missing files. `git config --system`
    writes to a scratch file, not the host's config. The eval script is
    recorded instead of run, since it needs the task's conda environment."""

    def __init__(self, root: Path) -> None:
        self.testbed = root / "testbed"
        self.env = {
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "GIT_CONFIG_SYSTEM": str(root / "gitconfig-system"),
            "GIT_CONFIG_GLOBAL": str(root / "gitconfig-global"),
        }
        self.env.pop("GIT_CONFIG_NOSYSTEM", None)
        self.eval_script: str | None = None

    def local(self, container_path: str) -> Path:
        return Path(container_path.replace("/testbed", str(self.testbed)))

    async def run(self, cmd: str, timeout: float | None = 300, max_bytes: int | None = 50_000, sanitise: bool = True) -> RunResult:
        if cmd.startswith("/bin/bash /testbed/eval_script.sh"):
            self.eval_script = self.local("/testbed/eval_script.sh").read_text()
            self.local("/testbed/eval_output.txt").write_text("eval log\n")
            return RunResult(output="", return_code=0)
        proc = subprocess.run(
            ["/bin/bash", "-c", cmd.replace("/testbed", str(self.testbed))], cwd=self.testbed, env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
        )
        return RunResult(output=proc.stdout, return_code=proc.returncode)

    async def check_run(self, cmd: str, timeout: float | None = 300, max_bytes: int | None = 50_000, sanitise: bool = True) -> str:
        result = await self.run(cmd, timeout=timeout, max_bytes=max_bytes, sanitise=sanitise)
        if result.return_code != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{result.output}")
        return result.output

    async def start(self) -> None:
        pass

    async def upload(self, local_path, container_path: str) -> None:
        self.local(container_path).write_bytes(Path(local_path).read_bytes())

    async def download(self, container_path: str) -> bytes:
        return self.local(container_path).read_bytes()

    def git(self, *args: str, cwd: Path | None = None) -> str:
        return subprocess.run(
            ["git", "-c", "protocol.file.allow=always", *args],
            cwd=cwd or self.testbed, env=self.env, check=True, capture_output=True, text=True,
        ).stdout

    def has_object(self, sha: str, cwd: Path | None = None) -> bool:
        return subprocess.run(
            ["git", "cat-file", "-e", sha], cwd=cwd or self.testbed, env=self.env, capture_output=True
        ).returncode == 0


BUGGY_CALC = "def add(a, b):\n    return a - b\n"
FIXED_CALC = "def add(a, b):\n    return a + b\n"


def _local_repo_env(root: Path) -> tuple[SWEGym, LocalGitSandbox, dict[str, str]]:
    """A repository set up like the task images: an upstream clone (with a
    submodule) reset to the base commit with its remote removed. The clone
    still holds a later fix commit and a tag on it, an unmerged branch, a
    stash, the reflog, and a later commit in the submodule's upstream."""
    sandbox = LocalGitSandbox(root)
    ident = ["-c", "user.name=u", "-c", "user.email=u@u"]
    shas: dict[str, str] = {}

    sub = root / "sub-upstream"
    sub.mkdir()
    (sub / "lib.py").write_text("X = 1\n")
    sandbox.git("init", "-q", cwd=sub)
    sandbox.git("add", "-A", cwd=sub)
    sandbox.git(*ident, "commit", "-q", "-m", "sub base", cwd=sub)
    shas["sub_base"] = sandbox.git("rev-parse", "HEAD", cwd=sub).strip()
    (sub / "lib.py").write_text("X = 2\n")
    sandbox.git(*ident, "commit", "-q", "-am", "sub later", cwd=sub)
    shas["sub_later"] = sandbox.git("rev-parse", "HEAD", cwd=sub).strip()

    upstream = root / "upstream"
    upstream.mkdir()
    (upstream / "calc.py").write_text("")
    (upstream / ".gitignore").write_text("*.egg-info/\n")
    sandbox.git("init", "-q", "-b", "main", cwd=upstream)
    sandbox.git("add", "-A", cwd=upstream)
    sandbox.git(*ident, "commit", "-q", "-m", "first", cwd=upstream)
    sandbox.git(*ident, "tag", "-a", "v0.9", "-m", "release 0.9", cwd=upstream)
    (upstream / "calc.py").write_text(BUGGY_CALC)
    sandbox.git(*ident, "submodule", "add", "-q", str(sub), "vendor", cwd=upstream)
    sandbox.git("-C", "vendor", "checkout", "-q", shas["sub_base"], cwd=upstream)
    sandbox.git("add", "-A", cwd=upstream)
    sandbox.git(*ident, "commit", "-q", "-m", "base", cwd=upstream)
    shas["base"] = sandbox.git("rev-parse", "HEAD", cwd=upstream).strip()
    sandbox.git("checkout", "-q", "-b", "feature", cwd=upstream)
    (upstream / "notes.txt").write_text("unmerged work\n")
    sandbox.git("add", "-A", cwd=upstream)
    sandbox.git(*ident, "commit", "-q", "-m", "feature", cwd=upstream)
    shas["feature"] = sandbox.git("rev-parse", "HEAD", cwd=upstream).strip()
    sandbox.git("checkout", "-q", "main", cwd=upstream)
    (upstream / "calc.py").write_text(FIXED_CALC)
    sandbox.git(*ident, "commit", "-q", "-am", "fix", cwd=upstream)
    shas["fix"] = sandbox.git("rev-parse", "HEAD", cwd=upstream).strip()
    sandbox.git(*ident, "tag", "-a", "v1.0", "-m", "release 1.0", cwd=upstream)
    shas["future_tag"] = sandbox.git("rev-parse", "v1.0", cwd=upstream).strip()
    shas["fixed_blob"] = sandbox.git("rev-parse", "HEAD:calc.py", cwd=upstream).strip()

    sandbox.git("clone", "-q", "--recurse-submodules", "-o", "origin", str(upstream), str(sandbox.testbed), cwd=root)
    sandbox.git("-C", "vendor", "fetch", "-q", "origin", shas["sub_later"])
    sandbox.git("reset", "-q", "--hard", shas["base"])
    sandbox.git("submodule", "update", "-q")
    sandbox.git("remote", "remove", "origin")
    (sandbox.testbed / "calc.py").write_text(FIXED_CALC)
    sandbox.git(*ident, "stash", "-q")
    (sandbox.testbed / "calc.egg-info").mkdir()
    (sandbox.testbed / "calc.egg-info" / "PKG-INFO").write_text("Name: calc\n")

    env = SWEGym(task_spec={**EXAMPLE_SWE_GYM_TASK, "base_commit": shas["base"]}, secrets={"api_key": "unused"})
    env.computer = sandbox
    return env, sandbox, shas


# Needs no sandbox and no API key.
@pytest.mark.asyncio
async def test_setup_leaves_no_git_route_to_future_commits(tmp_path: Path):
    env, sandbox, shas = _local_repo_env(tmp_path)
    base = shas["base"]
    describe = sandbox.git("describe", "--tags")
    assert all(sandbox.has_object(shas[k]) for k in ("fix", "feature", "future_tag", "fixed_blob"))

    await env.setup()

    assert sandbox.git("rev-list", "--all", "--reflog", "--not", base) == ""
    for key in ("fix", "feature", "future_tag", "fixed_blob"):
        assert not sandbox.has_object(shas[key]), key
    all_objects = sandbox.git("cat-file", "--batch-all-objects", "--batch-check").splitlines()
    assert len(all_objects) == len(sandbox.git("rev-list", "--objects", "--all").splitlines())

    assert sandbox.git("rev-parse", "HEAD").strip() == base
    assert sorted(sandbox.git("log", "--all", "--format=%s").split()) == ["base", "first"]
    assert sandbox.git("describe", "--tags") == describe
    assert sandbox.git("for-each-ref", "--format=%(refname)").split() == ["refs/heads/main", "refs/tags/v0.9"]
    assert sandbox.git("remote") == ""
    assert sandbox.git("status", "--porcelain") == ""
    assert (sandbox.testbed / "calc.py").read_text() == BUGGY_CALC

    vendor = sandbox.testbed / "vendor"
    assert sandbox.git("rev-parse", "HEAD", cwd=vendor).strip() == shas["sub_base"]
    assert sandbox.git("rev-list", "--all", "--reflog", "--not", "HEAD", cwd=vendor) == ""
    assert not sandbox.has_object(shas["sub_later"], cwd=vendor)
    assert sandbox.git("remote", cwd=vendor) == ""


# Needs no sandbox and no API key.
@pytest.mark.asyncio
async def test_answer_after_setup_diffs_against_base_commit(tmp_path: Path, monkeypatch):
    env, sandbox, shas = _local_repo_env(tmp_path)
    await env.setup()
    (sandbox.testbed / "calc.py").write_text(FIXED_CALC)
    seen = {}

    def fake_eval_report(test_spec, prediction, log_path, include_tests_status):
        seen["patch"] = prediction["model_patch"]
        return {test_spec.instance_id: {"resolved": True, "tests_status": {}}}

    monkeypatch.setattr("swegym.get_eval_report", fake_eval_report)
    out = (await env._call_tool("answer", {})).root.output
    assert out.reward == 1 and out.finished
    assert seen["patch"].count("diff --git") == 1
    assert "-    return a - b\n+    return a + b" in seen["patch"]
    assert f"git checkout {shas['base']} " in (sandbox.eval_script or "")
    sandbox.git("checkout", shas["base"], "--", "calc.py")
    assert (sandbox.testbed / "calc.py").read_text() == BUGGY_CALC


# Needs no sandbox and no API key.
@pytest.mark.parametrize("repo, size", [
    ("dask/dask", "1:4"),
    ("Project-MONAI/MONAI", "2:8"),
    ("pandas-dev/pandas", "1:2"),
    ("getmoto/moto", "1:2"),
])
def test_sandbox_machine_size_per_repo(repo: str, size: str):
    # dask's test suites run out of memory at 1:2, and MONAI needs more memory
    # to load its CUDA libraries.
    task = next(t for t in tasks if t["repo"] == repo)
    env = SWEGym(task_spec=task, secrets={"api_key": "unused"})
    assert env.compute_settings.machine_size == size


# Needs no sandbox and no API key.
@pytest.mark.parametrize("split", ["all", "lite"])
def test_unsolvable_instances_are_excluded(split: str):
    split_ids = {t["instance_id"] for t in SWEGym.list_tasks(split)}
    assert split_ids
    # modin's tests cannot start Ray in the sandbox.
    assert not [i for i in split_ids if i.startswith("modin-project__")]
    gold_failures = {
        line.strip() for line in (Path(__file__).parent / "gold_patch_failures.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    # Needs the network, fails with or without it, does not apply, and an image that does not exist.
    for iid in ("bokeh__bokeh-13041", "pandas-dev__pandas-50148", "python__mypy-11352", "conan-io__conan-13622"):
        assert iid in gold_failures | set(MISSING_IMAGE_INSTANCE_IDS)
        assert iid not in split_ids
    assert not split_ids & gold_failures


# Needs no sandbox and no API key.
@pytest.mark.parametrize("split", ["all", "lite"])
def test_statement_mismatch_instances_are_excluded(split: str):
    split_ids = {t["instance_id"] for t in SWEGym.list_tasks(split)}
    assert split_ids
    for iid in ("Project-MONAI__MONAI-3385", "iterative__dvc-4086"):
        assert iid not in split_ids


# Needs no sandbox and no API key.
def test_task_list_size():
    # Task ids are positions in this list, so a change to the excluded set renumbers them.
    assert len(SWEGym.list_tasks("all")) == 2004
    assert len(SWEGym.list_tasks("lite")) == 207
    # Solvable at 1:4, so it stays.
    assert "dask__dask-8945" in {t["instance_id"] for t in SWEGym.list_tasks("all")}

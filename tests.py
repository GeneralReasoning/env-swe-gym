import json
import logging
import os
from shlex import quote

import pytest
from openreward.environments import JSONObject, ToolOutput

from swegym import BashParams, SWEGym

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

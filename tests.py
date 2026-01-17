import logging
from shlex import quote
import os
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

import logging
from shlex import quote

import pytest

from matrix.server import JSONObject, ToolOutput
from environments.swegym.server import BashParams, SWEGym

logger = logging.getLogger(__name__)

tasks = SWEGym.get_tasks("all")
EXAMPLE_SWE_GYM_TASK = tasks[0]


@pytest.mark.asyncio
async def test_swe_gym_bash():
    env = SWEGym(task_spec=EXAMPLE_SWE_GYM_TASK)
    try:
        await env.setup()
        
        output: ToolOutput = await env.bash(BashParams(command="whoami"))
        assert isinstance(output.data["output"], str)
        assert "root" in output.data["output"], f"Expected 'root' in output, got {output.data['output']}"
    finally:
        await env.teardown()


@pytest.mark.asyncio
@pytest.mark.parametrize("task", tasks)
async def test_swe_gym_golden_patch(task: JSONObject):
    env = SWEGym(task_spec=task)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("task", tasks)
async def test_swe_gym_xfail_state(task: JSONObject):
    env = SWEGym(task_spec=task)
    try:
        await env.setup()

        res: ToolOutput = await env.answer()
        assert res.reward == 0, f"Expected reward of 0, got {res.reward}, full output: {res}"
        assert res.finished
    finally:
        await env.teardown()

import base64
import json
import tempfile
import asyncio
from pathlib import Path, PurePosixPath
from shlex import quote
from typing import Any, List

from datasets import load_dataset
from openreward import AsyncOpenReward, SandboxSettings
from openreward.environments import (Environment, JSONObject, TextBlock,
                                     ToolOutput, tool, Split)
from pydantic import BaseModel
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec import make_test_spec

from instructions import BASH_ONLY_INSTRUCTIONS
from utils import decode_patch_bytes

with open(Path(__file__).parent / "missing_images.txt", "r") as f:
    MISSING_IMAGE_INSTANCE_IDS = [line.strip() for line in f.readlines()]

DATASET = load_dataset("SWE-Gym/SWE-Gym", split="train")
EXAMPLES: list[dict[str, Any]] = DATASET.to_pandas().to_dict(orient="records")  # type: ignore
EXAMPLES = [i for i in EXAMPLES if i["instance_id"] not in MISSING_IMAGE_INSTANCE_IDS]

LITE_DATASET = load_dataset("SWE-Gym/SWE-Gym-Lite", split="train")
LITE_INSTANCE_IDS: list[str] = LITE_DATASET.to_pandas()["instance_id"].tolist()  # type: ignore
LITE_INSTANCE_IDS = [i for i in LITE_INSTANCE_IDS if i not in MISSING_IMAGE_INSTANCE_IDS]

# Reward for a submission made after the task has already been scored. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


class BashParams(BaseModel, extra="forbid"):
    command: str

class ValidatedSpec(BaseModel, extra="forbid"):
    instance_id: str
    hints_text: str
    patch: str
    test_patch: str
    created_at: str
    problem_statement: str
    repo: str
    base_commit: str
    version: str
    PASS_TO_PASS: list[str]
    FAIL_TO_PASS: list[str]
    max_response_length: int | None = None

# Text Editor tool params

class ViewParams(BaseModel, extra="forbid"):
    path: str
    start: int | None = None  # 1-indexed inclusive
    end: int | None = None    # 1-indexed inclusive

class StrReplaceParams(BaseModel, extra="forbid"):
    path: str
    old_str: str
    new_str: str

class CreateParams(BaseModel, extra="forbid"):
    path: str
    content: str

class InsertParams(BaseModel, extra="forbid"):
    path: str
    start: int  # 1-indexed line number to insert before
    content: str

class SWEGym(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = ValidatedSpec.model_validate(task_spec)

        self.instance = SWEbenchInstance(**self.validated.model_dump())
        self.base_commit = self.validated.base_commit
        self.test_spec = make_test_spec(self.instance)
        self.test_spec.arch = "x86_64"
        image = f"xingyaoww/{self.test_spec.instance_image_key}".replace("__", "_s_")

        api_key = secrets.get("OPENREWARD_API_KEY") or secrets.get("api_key")
        if not api_key:
            raise ValueError("OPENREWARD_API_KEY or api_key is not set")

        self.or_client = AsyncOpenReward(api_key=api_key)
        self.compute_settings = SandboxSettings(
            environment="jiayipan/SWE-Gym",
            image=image,
            machine_size="1:2"
        )
        self.computer = self.or_client.sandbox(self.compute_settings)

        # Scored submissions this session. answer() runs the SWE-bench eval --
        # the hidden FAIL_TO_PASS/PASS_TO_PASS tests -- and reports whether the
        # instance resolved. The agent also has bash and the editor tools, so an
        # uncapped answer() is a free CI loop against the held-out tests: edit,
        # score, read the result, edit again. The docstring already said this can
        # only be called once; this enforces it.
        self.submitted = 0


    async def setup(self) -> None:
        await self.computer.start()

        await self.computer.check_run("git config --system http.sslVerify false")
        await self.computer.check_run("git config --system user.email 'email@email.com'")
        await self.computer.check_run("git config --system user.name 'Name'")
        await self.computer.check_run("git config --system --add safe.directory /testbed")

    async def teardown(self) -> None:
        await self.computer.stop()

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=BASH_ONLY_INSTRUCTIONS.format(task=self.instance["problem_statement"]))]

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """
        Execute a bash command.
        """
        output, code = await self.computer.run(f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && {params.command.strip()}", timeout=600)
        max_len = self.task_spec.get("max_response_length")

        if isinstance(max_len, int):
            output = f"...(truncated)\n{output[-max_len:]}"

        return ToolOutput(
            metadata={"output": output, "exit_code": code},
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            reward=0.0,
            finished=False,
        )

    @tool
    async def answer(self) -> ToolOutput:
        """
        Computes the final score. This can only be called once, after all steps have been taken; only call
        this tool after you have finished all your steps and solved the coding issue.
        """
        if self.submitted > 0:
            return ToolOutput(
                metadata={"already_submitted": True, "submission_count": self.submitted},
                blocks=[TextBlock(text="This task has already been scored. The episode is over: "
                                       "the evaluation is not run again, and repeat submissions "
                                       "are penalised (reward -0.1).")],
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        report = await self._run_eval_with_retry()
        resolved = report[self.test_spec.instance_id]['resolved']

        # Incremented only after the eval succeeds; _run_eval_with_retry re-raises
        # a failing eval, which leaves the attempt retryable.
        self.submitted += 1

        return ToolOutput(
            metadata={"report": report},
            blocks=[TextBlock(text=f"Resolved: {resolved}")],
            reward=1 if resolved else 0,
            finished=True,
        )

    async def _run_eval_with_retry(self, *, max_attempts: int = 2) -> dict:
        """Run the SWE-bench eval in the sandbox and return the parsed report.

        The eval (apply patch, run tests, parse the report) is the grader's flaky
        external op; a failure is retried then re-raised so the SDK marks the call
        ToolFailed and ends the rollout. A legitimately failing patch is not an
        error — it yields a report with resolved=False and is scored normally.
        Grading is slow (~30 min), so attempts are kept low.
        """
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await self._run_eval()
            except Exception as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    print(f"SWE-Gym GRADING ERROR: {type(e).__name__}: {e} | retrying (attempt {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(5)
        assert last_exc is not None
        raise last_exc

    async def _run_eval(self) -> dict:
        """Apply the agent's patch, run the SWE-bench eval in the sandbox, and
        return the parsed report. Raises on any sandbox/eval/parse failure."""
        # extract patch for logging
        await self.computer.check_run(f"git add -A && git diff --cached {self.base_commit} > /testbed/model.patch")
        patch_bytes = await self.computer.download("/testbed/model.patch")
        patch = decode_patch_bytes(patch_bytes)

        # ---- Modify eval script to skip install commands ()
        eval_script_lines = self.test_spec.eval_script.split('\n')
        modified_eval_lines = []
        for line in eval_script_lines:
            # Skip lines that contain install commands
            if any(install_cmd in line for install_cmd in [
                "python -m pip install",
                "pip install",
                "conda install",
                "conda create"
            ]):
                continue
            modified_eval_lines.append(line)
        modified_eval_script = '\n'.join(modified_eval_lines)
        # ----

        with tempfile.TemporaryDirectory() as temp_dir:
            eval_file = Path(temp_dir) / "eval.sh"
            eval_file.write_text(modified_eval_script)
            await self.computer.upload(eval_file, str(PurePosixPath("/testbed/eval_script.sh")))

            # Write output to file to avoid SIGPIPE/max_bytes truncation
            _eval_output, _exit_code = await self.computer.run(
                "/bin/bash /testbed/eval_script.sh > /testbed/eval_output.txt 2>&1",
                timeout=1800,
            )
            eval_output_bytes = await self.computer.download("/testbed/eval_output.txt")
            test_output = eval_output_bytes.decode("utf-8", errors="replace")
            # Prepend patch marker expected by SWE-Bench-Fork's grading
            test_output = f">>>>> Applied Patch (pred)\n{test_output}"
            test_output_file = Path(temp_dir) / self.validated.instance_id / "test_output.txt"
            test_output_file.parent.mkdir(parents=True, exist_ok=True)
            # Handle potential encoding issues with test output
            try:
                test_output_file.write_text(test_output, encoding='utf-8')
            except UnicodeEncodeError:
                # If UTF-8 encoding fails, use utf-8 with error handling
                test_output_file.write_text(test_output, encoding='utf-8', errors='replace')

            # get the report from the test output
            report = get_eval_report(
                test_spec=self.test_spec,
                prediction={
                    "model_name_or_path": "None",
                    "model_patch": patch,
                    "instance_id": self.test_spec.instance_id,
                },
                log_path=str(PurePosixPath(test_output_file)),
                include_tests_status=True,
            )
        return report

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "all":
            validated_spec = [ValidatedSpec(**r) for r in EXAMPLES]
            return [v.model_dump() for v in validated_spec]
        elif split == "lite":
            validated_spec = [ValidatedSpec(**r) for r in EXAMPLES if r["instance_id"] in LITE_INSTANCE_IDS]
            return [v.model_dump() for v in validated_spec]
        else:
            raise ValueError(f"Unknown split: {split}")

    @classmethod
    def list_splits(cls) -> list[str]:
        # return ["all", "lite"]
        return [
            Split(name="all", type="train"),
            Split(name="lite", type="train"),
        ]

    # ---------- Text Editor tools (bash-only implementations) ----------

    @tool
    async def view(self, params: ViewParams) -> ToolOutput:
        """
        View file contents. Optionally specify a 1-indexed [start, end] line range.
        """
        p = quote(params.path)
        if params.start is not None or params.end is not None:
            start = params.start if params.start is not None else 1
            end = params.end if params.end is not None else '$'
            cmd = f"sed -n '{start},{end}p' {p}"
        else:
            cmd = f"cat {p}"
        output, code = await self.computer.run(cmd)
        max_len = self.validated.max_response_length
        if isinstance(max_len, int) and len(output) > max_len:
            output = f"...(truncated)\n{output[-max_len:]}"
        return ToolOutput(
            metadata={"content": output, "exit_code": code, "path": params.path},
            blocks=[TextBlock(text=output)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def str_replace(self, params: StrReplaceParams) -> ToolOutput:
        """
        Replace all occurrences of old_str with new_str in the given file. Use this tool to edit files.
        """
        path = params.path
        suffix = Path(path).suffix
        backup = f"{path}_old{suffix}"

        py = (
            "from pathlib import Path\n"
            f"p = Path({json.dumps(path)})\n"
            f"old = {json.dumps(params.old_str)}\n"
            f"new = {json.dumps(params.new_str)}\n"
            "text = p.read_text()\n"
            "p.write_text(text.replace(old, new))\n"
        )

        cmd = (
            f"set -e\n"
            f"cp {quote(path)} {quote(backup)}\n"
            f"python3 - << 'PY'\n{py}PY\n"
            f"git diff --no-index {quote(backup)} {quote(path)} || true"
        )

        output, exit_code = await self.computer.run(cmd)
        max_len = self.validated.max_response_length
        if isinstance(max_len, int) and len(output) > max_len:
            output = f"...(truncated)\n{output[-max_len:]}"
        return ToolOutput(
            metadata={"diff": output, "exit_code": exit_code, "backup_path": backup, "path": path},
            blocks=[TextBlock(text=output)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def insert(self, params: InsertParams) -> ToolOutput:
        """
        Insert content at the given 1-indexed line number. Use this tool to edit files.
        """
        path = params.path
        suffix = Path(path).suffix
        backup = f"{path}_old{suffix}"

        py = (
            "from pathlib import Path\n"
            "import sys\n"
            f"p = Path({json.dumps(path)})\n"
            f"start = int({json.dumps(params.start)})\n"
            f"content = {json.dumps(params.content)}\n"
            "if not p.exists():\n"
            "    p.parent.mkdir(parents=True, exist_ok=True)\n"
            "    p.write_text('')\n"
            "text = p.read_text()\n"
            "lines = text.splitlines(keepends=True)\n"
            "idx = max(0, min(start - 1, len(lines)))\n"
            "new_text = ''.join(lines[:idx]) + content + ''.join(lines[idx:])\n"
            "p.write_text(new_text)\n"
        )

        cmd = (
            f"set -e\n"
            f"if [ -f {quote(path)} ]; then cp {quote(path)} {quote(backup)}; "
            f"else mkdir -p $(dirname {quote(path)}); : > {quote(path)}; cp {quote(path)} {quote(backup)}; fi\n"
            f"python3 - << 'PY'\n{py}PY\n"
            f"git diff --no-index {quote(backup)} {quote(path)} || true"
        )

        output, _ = await self.computer.run(cmd)
        max_len = self.validated.max_response_length
        if isinstance(max_len, int) and len(output) > max_len:
            output = f"...(truncated)\n{output[-max_len:]}"
        return ToolOutput(
            metadata={"diff": output, "exit_code": 0, "backup_path": backup, "path": path, "start": params.start},
            blocks=[TextBlock(text=output)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def create(self, params: CreateParams) -> ToolOutput:
        """
        Create a file with the given content.
        """
        path = params.path
        path_q = quote(path)
        b64 = base64.b64encode(params.content.encode()).decode()
        # Use base64 to avoid here-doc delimiter collisions and escaping issues
        cmd = (
            f"set -e; "
            f"mkdir -p $(dirname {path_q}); "
            f"printf '%s' {quote(b64)} | base64 -d > {path_q}; "
            f"printf 'Created {path} (%s bytes)\\n' $(wc -c < {path_q})"
        )
        output, code = await self.computer.run(cmd)
        msg = output.strip()
        return ToolOutput(
            metadata={"message": msg, "path": path, "bytes": len(params.content), "exit_code": code},
            blocks=[TextBlock(text=msg)],
            reward=0.0,
            finished=False,
        )

import json
from typing import Any
import logging
import os
import tempfile
import traceback
from pathlib import Path, PurePosixPath
from shlex import quote
import base64

from datasets import Dataset, load_dataset
from pydantic import BaseModel

from construct.client import Computer, ComputeSettings
from matrix.server import Environment, JSONObject, Server, TextOutput, ToolOutput, tool
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec import make_test_spec
from environments.swegym.instructions import BASH_ONLY_INSTRUCTIONS
from environments.swegym.utils import decode_patch_bytes

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "indigo-idea-457514-b5")
GCP_REPOSITORY = os.getenv("GCP_REPOSITORY", "r2e-proxy")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")

with open(Path(__file__).parent / "missing_images.txt", "r") as f:
    MISSING_IMAGE_INSTANCE_IDS = [line.strip() for line in f.readlines()]

DATASET = load_dataset("SWE-Gym/SWE-Gym", split="train")
EXAMPLES: list[dict[str, Any]] = DATASET.to_pandas().to_dict(orient="records")  # type: ignore
EXAMPLES = [i for i in EXAMPLES if i["instance_id"] not in MISSING_IMAGE_INSTANCE_IDS]

LITE_DATASET = load_dataset("SWE-Gym/SWE-Gym-Lite", split="train")
LITE_INSTANCE_IDS: list[str] = LITE_DATASET.to_pandas()["instance_id"].tolist()  # type: ignore
LITE_INSTANCE_IDS = [i for i in LITE_INSTANCE_IDS if i not in MISSING_IMAGE_INSTANCE_IDS]

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
    def __init__(self, task_spec: JSONObject) -> None:
        super().__init__(task_spec)        
        self.validated = ValidatedSpec.model_validate(task_spec)
        
        self.instance = SWEbenchInstance(**self.validated.model_dump())
        self.base_commit = self.validated.base_commit
        self.test_spec = make_test_spec(self.instance)
        self.test_spec.arch = "x86_64"
        image = f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/{GCP_REPOSITORY}/xingyaoww/{self.test_spec.instance_image_key}".replace("__", "_s_")

        self.compute_settings = ComputeSettings(
            image=image,
            cpu_request="1",
            cpu_limit="2",
            memory_request="1G",
            memory_limit="2G",
        )
        self.computer = Computer(self.compute_settings)

    async def setup(self) -> None:
        await self.computer.__aenter__()
        
        await self.computer.check_run("git config --system http.sslVerify false")
        await self.computer.check_run("git config --system user.email 'email@email.com'")
        await self.computer.check_run("git config --system user.name 'Name'")
        await self.computer.check_run("git config --system --add safe.directory /testbed")

    async def teardown(self) -> None:
        await self.computer.__aexit__(None, None, None)

    def get_prompt(self) -> str:
        return BASH_ONLY_INSTRUCTIONS.format(task=self.instance["problem_statement"])

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
            data={"output": output, "exit_code": code},
            raw=TextOutput(text=f"{output}\n\n(exit {code})"),
            reward=0.0,
            finished=False,
        )

    @tool
    async def answer(self) -> ToolOutput:
        """
        Computes the final score. This can only be called once, after all steps have been taken; only call
        this tool after you have finished all your steps and solved the coding issue.
        """
        try:
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
            
                test_output = await self.computer.check_run("/bin/bash /testbed/eval_script.sh", timeout=1800)
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
            return ToolOutput(
                data={"report": report},
                reward=1 if report[self.test_spec.instance_id]['resolved'] else 0,
                finished=True,
            )
        except Exception:
            return ToolOutput(
                data={"error": traceback.format_exc()},
                reward=0.0,
                finished=True,
            )

    @classmethod
    def get_tasks(cls, split: str) -> list[JSONObject]:
        if split == "all":
            validated_spec = [ValidatedSpec(**r) for r in EXAMPLES]
            return [v.model_dump() for v in validated_spec]
        elif split == "lite":
            validated_spec = [ValidatedSpec(**r) for r in EXAMPLES if r["instance_id"] in LITE_INSTANCE_IDS]
            return [v.model_dump() for v in validated_spec]
        else:
            raise ValueError(f"Unknown split: {split}")

    @classmethod
    def get_splits(cls) -> list[str]:
        return ["all", "lite"]

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
            data={"content": output, "exit_code": code, "path": params.path},
            raw=TextOutput(text=output),
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
            data={"diff": output, "exit_code": exit_code, "backup_path": backup, "path": path},
            raw=TextOutput(text=output),
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
            data={"diff": output, "exit_code": 0, "backup_path": backup, "path": path, "start": params.start},
            raw=TextOutput(text=output),
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
            data={"message": msg, "path": path, "bytes": len(params.content), "exit_code": code},
            raw=TextOutput(text=msg),
            reward=0.0,
            finished=False,
        )


if __name__ == "__main__":
    Server(environments=[SWEGym]).run()
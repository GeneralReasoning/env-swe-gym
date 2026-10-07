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

# Rebuilds the repository in /testbed, and each submodule's, from base_commit
# (a submodule: its checked-out commit), its history and the tags in that
# history. The image's clone also holds upstream commits made after base_commit
# (in tags, the reflog and the object store), and those contain the fix.
# History up to base_commit stays, and core.abbrev keeps the short-hash length,
# so that `git describe`, and the version that versioneer-based packages such
# as pandas compute from it at import time, does not change. The working tree,
# .git/info and the repository config (without remotes) are kept; the index
# is rebuilt.
SNAPSHOT_REPO_SCRIPT = r"""set -euo pipefail
snapshot() {
  cd "$1"
  local gitdir commit fresh branch
  gitdir=$(git rev-parse --absolute-git-dir)
  commit=$(git rev-parse --verify "$2^{commit}")
  fresh="$gitdir.snapshot"
  rm -rf "$fresh"
  git init -q --bare "$fresh"
  git for-each-ref --merged "$commit" --format='%(objectname) %(refname)' refs/tags > "$fresh/kept-tags"
  { echo "$commit"; cut -d' ' -f1 "$fresh/kept-tags"; } \
    | git rev-list --objects --stdin \
    | git pack-objects -q --stdout \
    | git -C "$fresh" index-pack --stdin > /dev/null
  if [ -f "$gitdir/shallow" ]; then cp "$gitdir/shallow" "$fresh/shallow"; fi
  sed 's/^\([0-9a-f]*\) \(.*\)$/create \2 \1/' "$fresh/kept-tags" | git -C "$fresh" update-ref --stdin
  rm "$fresh/kept-tags"
  if branch=$(git symbolic-ref -q HEAD); then
    git -C "$fresh" symbolic-ref HEAD "$branch"
    git -C "$fresh" update-ref "$branch" "$commit"
  else
    git -C "$fresh" update-ref --no-deref HEAD "$commit"
  fi
  cp "$gitdir/config" "$fresh/config"
  git config -f "$fresh/config" core.abbrev "$(git rev-parse --short "$commit" | awk '{ print length }')"
  { git config -f "$fresh/config" --name-only --get-regexp '^(remote|branch)\.' || true; } \
    | sed 's/\.[^.]*$//' | sort -u \
    | while read -r section; do git config -f "$fresh/config" --remove-section "$section"; done
  cp -R "$gitdir/info/." "$fresh/info/"
  if [ -d "$gitdir/modules" ]; then mv "$gitdir/modules" "$fresh/modules"; fi
  rm -rf "$gitdir"
  mv "$fresh" "$gitdir"
  git reset -q
}
snapshot /testbed "$BASE_COMMIT"
cd /testbed
git submodule foreach --recursive --quiet 'echo "$toplevel/$sm_path"' | while read -r path; do (snapshot "$path" HEAD); done
"""


# With the network blocked, outbound connections are dropped, so curl, pip and
# git wait minutes for a connect timeout. Proxy-aware tools are pointed at a
# closed local port instead, so they fail at once with "connection refused".
# Local connections bypass the proxy.
REFUSING_PROXY = "http://127.0.0.1:9"
NO_NETWORK_ENV = {
    "http_proxy": REFUSING_PROXY,
    "https_proxy": REFUSING_PROXY,
    "HTTP_PROXY": REFUSING_PROXY,
    "HTTPS_PROXY": REFUSING_PROXY,
    "no_proxy": "localhost,127.0.0.1,::1",
    "NO_PROXY": "localhost,127.0.0.1,::1",
}


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
            machine_size="1:2",
            # Every task comes from a public upstream pull request, so with
            # network access the agent could download the fix. Setup and
            # grading need no network: the images ship the repository and its
            # dependencies, the eval script's install lines are skipped, and
            # uploads/downloads go through the SDK.
            block_network=True,
            env=NO_NETWORK_ENV,
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
        await self.computer.check_run(
            f"BASE_COMMIT={quote(self.base_commit)}\n{SNAPSHOT_REPO_SCRIPT}", timeout=900
        )

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

        # tests_status names the grading tests, and metadata reaches the model
        # like the text does, so the report carries per-group counts instead.
        instance_report = dict(report[self.test_spec.instance_id])
        tests_status = instance_report.pop("tests_status", {})
        instance_report["tests_status_counts"] = {
            group: {outcome: len(tests) for outcome, tests in outcomes.items()}
            for group, outcomes in tests_status.items()
        }

        return ToolOutput(
            metadata={"report": {self.test_spec.instance_id: instance_report}},
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
            try:
                await self.computer.upload(eval_file, str(PurePosixPath("/testbed/eval_script.sh")))
            except RuntimeError as e:
                # A failed upload's message quotes the command, which embeds the
                # eval script and so the test patch; it reaches the model if raised.
                print(f"SWE-Gym eval script upload failed: {e}")
                raise RuntimeError("Uploading the eval script to the sandbox failed.") from None

            # Write output to file to avoid SIGPIPE/max_bytes truncation
            _eval_output, _exit_code = await self.computer.run(
                "/bin/bash /testbed/eval_script.sh > /testbed/eval_output.txt 2>&1",
                timeout=1800,
            )
            eval_output_bytes = await self.computer.download("/testbed/eval_output.txt")
            test_output = eval_output_bytes.decode("utf-8", errors="replace")
            # Prepend patch marker expected by SWE-Bench-Fork's grading
            test_output = f">>>>> Applied Patch (pred)\n{test_output}"
            # The fork derives the repo from this directory name and looks it up in a
            # lowercase-keyed parser map, so use the lowercased test_spec id
            # (Project-MONAI__MONAI-* would otherwise raise KeyError).
            test_output_file = Path(temp_dir) / self.test_spec.instance_id / "test_output.txt"
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

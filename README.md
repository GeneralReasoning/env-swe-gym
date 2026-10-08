# SWE-Gym

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/jiayipan/SWE-Gym) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/SWE-Gym/SWE-Gym)

## Description

SWE-Gym is a training and evaluation environment for software engineering agents. It contains 2,006 real-world Python task instances from 10 popular open-source repositories (including pandas, mypy, moto, dask and MONAI). Each task provides a codebase with an executable runtime, a natural language problem statement describing an issue, and unit tests that verify whether the issue has been resolved.

## Capabilities

- Real-world software engineering tasks on production Python codebases
- Python bug fixing and feature implementation
- Test-based verification of code changes
- File viewing, editing, and creation within a repository sandbox
- Bash command execution for codebase exploration and testing

## Compute Requirements

Each agent is given an isolated Docker sandbox with 1 CPU and 2GB of RAM, except dask tasks (1 CPU, 4GB), whose test suites run out of memory at 2GB, and MONAI tasks (2 CPU, 8GB), which need the memory to load their CUDA libraries. Per-task Docker images are used, with pre-installed dependencies specific to each repository and version.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are two splits in this environment:

- **all**: 2,006 task instances spanning 10 Python repositories. This is the SWE-Gym training set (2,438 instances), excluding the instances listed under Data.
- **lite**: 207 curated task instances, a subset of the full set selected for higher quality and diversity, with the same exclusions.

Each task provides:
- A **problem statement** describing the issue to be fixed (from the original GitHub issue or pull request).
- A **codebase** checked out at the relevant base commit in `/testbed`. At setup its git repository is rebuilt to hold only the base commit, its history and the tags in that history, with no remotes; upstream commits made after the base commit, which contain the fix, are removed.
- **Unit tests** (FAIL_TO_PASS and PASS_TO_PASS) that determine whether the fix is correct.

## Reward Structure

Rewards are **binary** (1.0 or 0.0) and **deterministic**. When the agent calls the `answer` tool, the environment:

1. Extracts the git diff of all changes made to the codebase.
2. Runs the evaluation test suite using `swebench.harness.grading`.
3. Returns a reward of 1.0 if the issue is resolved (all FAIL_TO_PASS tests now pass and all PASS_TO_PASS tests still pass), and 0.0 otherwise.

No LLM graders are used for this environment.

## Data

Task data is loaded at runtime from HuggingFace:
- [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) for the full dataset.
- [SWE-Gym/SWE-Gym-Lite](https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Lite) for the curated lite subset.

Excluded instances:
- 37 whose Docker images are unavailable (`missing_images.txt`).
- All 105 modin instances: their tests start Ray, whose object store does not fit in the sandbox, so almost no gold patch resolves.
- 290 whose gold patch does not resolve the instance in the sandbox with the network blocked (`gold_patch_failures.txt`, grouped by cause): it needs network access, fails with or without it, does not apply, or passes only in some runs.
- 2 whose hidden tests check behaviour the problem statement does not describe (`statement_mismatch.txt`, with the reason).

Task ids (`task_all_N`) are positions in the filtered list, so changing these exclusions renumbers them.

Some pandas fixes change Cython or C sources. Those take effect only after the extensions are rebuilt (`python setup.py build_ext --inplace`, about 10 minutes on 1 CPU), and grading does not rebuild them.

## Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `bash` | `command: str` | Execute bash commands in the sandbox (600s timeout). Runs within the testbed conda environment. |
| `view` | `path: str`, `start: int?`, `end: int?` | View file contents or a specific line range (1-indexed, inclusive). |
| `str_replace` | `path: str`, `old_str: str`, `new_str: str` | Replace all occurrences of a string in a file. Shows the resulting diff. |
| `insert` | `path: str`, `start: int`, `content: str` | Insert content at a given 1-indexed line number. Shows the resulting diff. |
| `create` | `path: str`, `content: str` | Create a new file with the given content. |
| `answer` | *(none)* | Extract the patch, run the test suite, and return the resolved status. Ends the episode. |

## Time Horizon

SWE-Gym is a multi-turn environment. The agent iteratively explores the codebase, identifies the root cause of the issue, implements a fix, and verifies it before submitting via the `answer` tool. The episode ends when `answer` is called.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

There are no external API keys required beyond OpenReward platform access. The per-task Docker images are managed by the OpenReward sandbox infrastructure.

## Safety

Agents operate in isolated Docker sandboxes provisioned per task. Each sandbox is resource-limited (see Compute Requirements). Outbound network access is blocked: every task comes from a public upstream pull request, so the fix would otherwise be downloadable. The images ship the repository and its dependencies, so setup and grading need no network. The agent cannot affect the host system or other running environments.

## Citation

```bibtex
@inproceedings{pan2025swegym,
  title={Training Software Engineering Agents and Verifiers with SWE-Gym},
  author={Pan, Jiayi and Wang, Xingyao and Neubig, Graham and Jaitly, Navdeep and Ji, Heng and Suhr, Alane and Zhang, Yizhe},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```

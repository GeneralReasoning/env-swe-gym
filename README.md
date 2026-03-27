# SWE-Gym

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/jiayipan/SWE-Gym) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/SWE-Gym/SWE-Gym)

## Description

SWE-Gym is a training and evaluation environment for software engineering agents. It contains 2,438 real-world Python task instances sourced from 11 popular open-source repositories (including django, flask, sympy, pandas, and others). Each task provides a codebase with an executable runtime, a natural language problem statement describing an issue, and unit tests that verify whether the issue has been resolved.

## Capabilities

- Real-world software engineering tasks on production Python codebases
- Python bug fixing and feature implementation
- Test-based verification of code changes
- File viewing, editing, and creation within a repository sandbox
- Bash command execution for codebase exploration and testing

## Compute Requirements

Each agent is given an isolated Docker sandbox with 1 CPU and 2GB of RAM. Per-task Docker images are used, with pre-installed dependencies specific to each repository and version.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are two splits in this environment:

- **all**: 2,438 task instances spanning 11 Python repositories. This is the full SWE-Gym training set, excluding instances with missing Docker images.
- **lite**: 230 curated task instances, a subset of the full set selected for higher quality and diversity. Also excludes instances with missing Docker images.

Each task provides:
- A **problem statement** describing the issue to be fixed (from the original GitHub issue or pull request).
- A **codebase** checked out at the relevant base commit in `/testbed`.
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

Instances whose Docker images are unavailable (36 instances) are automatically excluded.

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

Agents operate in isolated Docker sandboxes provisioned per task. Each sandbox is resource-limited (1 CPU, 2GB RAM) and network-restricted. The agent cannot affect the host system or other running environments.

## Citation

```bibtex
@inproceedings{pan2025swegym,
  title={Training Software Engineering Agents and Verifiers with SWE-Gym},
  author={Pan, Jiayi and Wang, Xingyao and Neubig, Graham and Jaitly, Navdeep and Ji, Heng and Suhr, Alane and Zhang, Yizhe},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```

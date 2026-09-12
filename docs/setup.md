# Setup checklist

Project: **QK Attribution** — implementing QK attributions and head loadings for attribution graphs
(`safety-research/circuit-tracer` issue #53).

---

## Already verified as in place

**worker-1 (10.10.3.83)** — the machine we work on
- Python 3.10.12, pip 26.1.2, `uv` installed at /usr/local/bin/uv
- git and git-lfs both present (git-lfs matters for pulling HF weights)
- torch 2.8.0, transformers 4.57.6, accelerate 1.14.0, datasets 4.3.0, einops 0.8.1,
  huggingface_hub 0.36.2, numpy 1.26.4
- 235 GB free on the volume; HF cache already holds 39 GB of Qwen2.5-Coder and DeepSeek-Coder
- GPU idle: 552 MiB of 24564 MiB used, 0% utilisation
- nvcc is CUDA 11.5 while torch is built for cu128. Irrelevant here, since torch ships its own
  runtime and we are not compiling custom CUDA.

**Local box**
- `gh` installed and authenticated as GitHub user VivianSobers
- git identity configured (Vivian Sobers / vivianedwardbangalore@gmail.com)
- No GPU, no `uv`. Development and git work only; all compute runs on worker-1.

**Upstream**
- circuit-tracer is published on PyPI, so `pip install circuit-tracer` works
- MIT licensed, no CLA required
- Repo docs state Gemma-2-2B runs on Colab's 15 GB GPUs, so 24 GB is comfortable

---

## Needed from you

### 1. Hugging Face account — required

Yes, you need one. Three reasons:

- `google/gemma-2-2b` is **gated**. The page says "This repository is publicly accessible, but you
  have to accept the conditions to access its files and content." Requests process immediately once
  you are logged in and accept Google's licence.
- `meta-llama/Llama-3.2-1B` is gated the same way, under Meta's licence.
- The transcoder weights themselves live on HF (`mntss/gemma-scope-transcoders`,
  `mwhanna/gemma-scope-2-*`, and the Llama transcoders trained by the circuit-tracer team).

Steps:
1. Create an account at huggingface.co
2. Visit huggingface.co/google/gemma-2-2b and accept the licence
3. Visit huggingface.co/meta-llama/Llama-3.2-1B and accept the licence
4. Create a **read** token under Settings, Access Tokens
5. On worker-1, run `huggingface-cli login` and paste it. Do not paste the token into chat.

A fourth reason applies later: if the work produces artifacts worth publishing, you will want an
account to host them.

### 2. Permission to install on worker-1

It is a shared lab machine. I would create an isolated virtualenv with `uv` under your home
directory and touch nothing system-wide. Confirm that is acceptable.

### 3. Confirm the GPU is yours to occupy

worker-1's GPU is idle now, but the account is shared and someone else may want it. The spike needs
it for under an hour. Longer runs later need a clearer arrangement.

### 4. GitHub repo

`gh` is already authenticated as you, so I can create it on request. Say whether you want me to, or
you would rather create it yourself.

### 5. Rotate the SSH password

`ccbd@123` is in this session's history and in the shell history on both boxes. Rotate it if it
protects anything that matters.

---

## Naming

**Project title:** QK Attribution: Explaining Why Language Models Attend Where They Do

Alternative if you want something shorter and more memorable: *Why Attend?*

**Repo name:** `qk-attribution`

Searchable, matches the terminology used in issue #53 and in Anthropic's write-up, and instantly
legible to anyone who reads "implemented QK attribution" on your CV.

Note that you need your own repo even though the goal is an upstream PR. Yours holds the
development work, experiments, figures, notebooks and the writeup. The upstream contribution is a
PR into circuit-tracer, and the two serve different purposes.

---

## Upstream process, per their CONTRIBUTING.md

- Contributions welcome under MIT, no CLA
- They ask that you **open an issue to discuss major changes before starting work**. Since #53
  already exists, comment on it to signal intent. This avoids duplicated effort and surfaces
  maintainer preferences early.
- Required before submitting: add tests, run `pytest`, run `ruff check` and `ruff format`, run
  `pyright`, and confirm the demo notebooks still execute
- Their stated caveat: "PR reviews may take time and we cannot guarantee timely responses or merges"
- Also: "This library is under active development and breaking changes are possible. The API is
  not stable."

---

## First action

A one-day spike before any commitment: install circuit-tracer in a venv on worker-1, generate one
attribution graph on Gemma-2-2B end to end, record peak VRAM and wall time. The repo claims 15 GB
suffices, so this should pass, but it converts an inference into a measurement.

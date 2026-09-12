# QK Attribution

Explaining *why* language models attend where they do, by decomposing attention scores into
interactions between query-side and key-side features.

## The problem

Attribution graphs, as implemented in [circuit-tracer](https://github.com/safety-research/circuit-tracer),
freeze attention patterns and treat them as constants. The graph therefore shows where information
flows through attention via the OV pathway, but says nothing about why the model attended to those
positions in the first place. The QK pathway is invisible.

Anthropic described a method that closes this gap in
[Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html),
calling QK attributions "a significant qualitative improvement on the original attribution graphs,
unlocking analyses that were previously impossible."

It is not available in the open-source tool.
[Issue #53](https://github.com/safety-research/circuit-tracer/issues/53) requested it in
November 2025 and it remains unimplemented.

## What this repository does

Implements QK attributions and head loadings on top of existing pre-trained transcoders, with the
aim of contributing the result upstream.

Attention scores are bilinear in the residual streams at the query and key positions, so inserting
sparse decompositions at both ends expands the score into feature-pair interactions:

    C_ij^h(q,k) = [a_i(q) * a_j(k) / sqrt(d_h)] * v_i^T W_QK^h v_j

and each head's contribution to an existing graph edge s -> t can be isolated as:

    L_h(s->t) = a_s * a_t * (v_t^T W_OV^h v_s) * A^h(p_t, p_s)

The cost is quadratic in context length times feature count, so making this tractable is as much of
the work as computing it correctly. Anthropic note that many QK attribution matrices are
approximately low-rank, which is the avenue this repository explores.

## Status

Early. Environment and infrastructure verified, implementation not yet started.

See [docs/method.md](docs/method.md) for the derivation, [docs/setup.md](docs/setup.md) for the
environment, and [experiments/](experiments/) for measured results.

## Installation

    uv venv --python 3.10 .venv
    . .venv/bin/activate
    uv pip install -e ".[dev]"

Transcoder weights and gated model weights come from Hugging Face, so `hf auth login` is required.

## Checks

Upstream circuit-tracer requires all of these to pass before a PR:

    pytest
    ruff check
    ruff format --check
    pyright

## Licence

MIT, matching upstream.

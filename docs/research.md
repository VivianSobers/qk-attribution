# Project selection research

Compiled 2026-09-12. Constraints: 3rd year BTech, industry-facing resume project (not a paper),
2x RTX 4090 in separate boxes (1 GbE between them, 111 MB/s measured), no local GPU.

---

## Recommendation

**Implement QK attribution and head loadings in `safety-research/circuit-tracer` (open issue #53).**

Anthropic published the method in "Tracing Attention Computation Through Feature Interactions"
(transformer-circuits.pub/2025/attention-qk). It closes the biggest known hole in circuit tracing.
The open-source tool does not implement it. Issue #53 requested it on 17 Nov 2025 and it remains
unassigned, unclaimed, with no branch, no PR, and no maintainer reply, ten months later.

### Why this survived when nothing else did

The method is already specified in public, so there is no research risk and nothing to be scooped on.
The demand is documented in the canonical repo rather than assumed by me. A merged PR is permanent
and attributable. And it is hard in a way that suits you: attention scores are bilinear in the query
and key residual streams, so decomposing them is quadratic in context length times feature count,
and making that tractable is a systems problem sitting inside an interpretability problem.

### The math is concrete

QK attribution, per head, for query position q and key position k, feature i at query and j at key:

    C_ij^h(q,k) = [a_i^(q) * a_j^(k) / sqrt(d_h)] * v_i^T W_QK^h v_j

Head loading for a graph edge s -> t:

    L_h(s->t) = a_s * a_t * (v_t^T W_OV^h v_s) * A^h_pt_ps

Both are matrix algebra over pre-trained transcoder features. Implementable.

### Anthropic's own flagged extension

"Many QK attribution matrices are approximately low-rank, which may permit a shorter description."
Unexplored. Exploiting that is the contribution beyond reproduction, and it is exactly the
engineering angle that makes the quadratic cost survivable.

### Stated limitations of the method (room to work in)

1. Bias terms are less semantically precise than learned features and can obscure computations
2. Head polysemanticity: labels reflect roles on single prompts
3. Explains semantic matching but not the positional preferences that break ties
4. Decomposes evidence entering softmax, not the softmax decision itself, so inhibitory terms are missed

---

## Hardware verdict: feasible, with one thing to verify first

**You do not need to train transcoders.** That was the blocker and it is removed. circuit-tracer
already ships them for Gemma-2-2B (426K and 2.5M features), Llama-3.2-1B, Qwen-3 (0.6B to 14B),
Gemma-3, GPT-OSS-20B and Llama-3.1-8B-Instruct.

Training cross-layer transcoders is what costs real compute. CLT-Forge reports **8x80GB GPUs** for
Llama-1B at expansion factor 48. That is far beyond 48 GB split across two boxes. Using pre-trained
transcoders makes this inference-scale work on a 1-2B model, which one 4090 handles.

Attribution graph analysis runs on short single prompts, and TopK transcoders keep only k active
features per position (order 32-128), so the practical cost per query-key pair is k^2, not
millions squared.

**Risk to check on day one:** issue #92 reports OOM on high-end hardware. 24 GB may be tight.
Run existing attribution graph generation on Qwen-3-0.6B on worker-1 before committing anything.

Measured hardware facts from this investigation:
- worker-1 (10.10.3.83): RTX 4090 24564 MiB, driver 595.84, torch 2.8.0+cu128, 32 cores, 125 GB RAM
- worker-2 (10.10.3.50): RTX 4090 24564 MiB, driver 580.178.04, torch 2.7.0+cu126, 26 cores, 125 GB RAM
- PCIe H2D 17.4 GB/s, D2H 17.2 GB/s (worker-1, pinned)
- Inter-box: 111 MB/s, 0.695 ms RTT. 157x slower than PCIe. Useless for anything tightly coupled.
- Local box: no GPU at all, 14 GB RAM. Development only.

---

## Career logic, stated honestly

Do **not** pick this expecting an interpretability job. Interp roles are heavily PhD-gated: postings
ask for a PhD plus 2+ years research experience, or a Bachelor's with 5+ years hands-on research,
and value publication records at NeurIPS/ICML/ICLR. That is not reachable from 3rd year BTech.

Pick it because a merged PR in a 2.9k-star Anthropic-affiliated repo, implementing a published
research method, is a top-tier *general* signal that transfers to any ML role. Per LinkedIn's 2025
Tech Hiring Report, 41% of ML hiring managers prioritise candidates with active GitHub activity over
equivalent experience without it, and for a new grad with no professional experience one quality
merged PR outweighs a pile of solo projects, because it proves you can read a complex codebase,
meet someone else's standards, and survive review.

It also keeps the interp door open if you later do a PhD.

---

## What was checked and eliminated

| Area | Verdict | Killed by |
|---|---|---|
| Weight offloading, batch amortization | Dead | FlexGen (arXiv 2303.06865), ICML 2023. Same thesis. |
| MoE expert offloading | Dead | FreeToken (Berkeley/MIT, Aug 2026) runs 753B on one workstation GPU. Plus MoBiLE, HybriMoE, FineMoE (EuroSys 26), vLLM RFC #38256. |
| Prefill/decode disaggregation across boxes | Impossible | 256 MB KV per 2048-token request at 111 MB/s = 2.3 s. |
| Tensor parallelism across boxes | Impossible | 64 collectives/token x 0.695 ms = 45 ms/token, capping ~22 tok/s. One GPU alone does 50-100. |
| KV cache offloading | Crowded | vLLM shipped a KV offloading connector Jan 2026; NVIDIA CPU-GPU memory sharing; DUAL-BLADE. |
| Speculative decoding | Occupied | EAGLE 1/2/3/3.1, P-EAGLE shipped in vLLM, HeteroSpec, RASD, SuffixDecoding. |
| Diffusion LLM serving | Filling fast | Fast-dLLM (NVIDIA), HERALD, DyLLM, Sangam, AdaBlock-dLLM, dLLM-Serve. |
| Quantization quality evaluation | Partly open, but taken | "Displacement Is Not Direction" (2606.19558) did the cheap-metric analysis. |
| Compression x interpretability | Taken | "Interpreting the Effects of Quantization on LLMs" (2508.16785), "Perplexity Can Miss SAE Feature Damage Under Quantization" (2606.03002), "How Pruning Reshapes Features" (2603.25325). |
| Training transcoders for uncovered models | Dead | circuit-tracer already covers Qwen-3, Gemma-3, GPT-OSS, Llama-3.1-8B. EleutherAI sparsify + CLT-Forge automate training. |
| Transcoder evaluation / sanity checks | Taken | "Sanity Checks for SAEs" (2602.14111) found random baselines match trained SAEs. "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers" (ICLR 2026). |
| CoT faithfulness interpretability | Crowded | FaithCoT-Bench, Reasoning Theater, Beyond the Commitment Boundary, and more through 2026. |

### The pattern

Every area that is both cutting edge and about to be big already has five to ten papers and usually
a frontier lab. That is structural, not bad luck. "About to become big" means well-funded groups are
already sprinting there. Airtight and cutting-edge-race pull in opposite directions.

Issue #53 resolves the tension because its value comes from execution and permanence, not from
being first to an idea.

---

## Neighbouring repos swept

- **TransformerLens** (3.9k stars): 8 open issues, 1 open PR. Biggest items are an SVD circuits
  proposal (#1767) and leakage-safe k-sparse probing (#1728). Slower moving, some issues 1-2 years old.
- **EleutherAI/sparsify** (739 stars): 5 open issues, mostly bugs and monitoring. Two labelled
  good first issue. Nothing comparable in scope.
- **circuit-tracer** (2.9k stars, 344 forks, 67 merged PRs): healthiest of the three. Active
  contributors include TransformerLens and Neuronpedia maintainers. 13 open PRs, none touching #53.

Fallback contributions in the same repo if #53 stalls: multi-GPU support (#40), TopK CLTs (#56).

---

## Risk register

**Someone else implements it first.** Anthropic, Decode, EleutherAI, Goodfire and DeepMind have a
replication collaboration running. Ten months of silence on #53 suggests it is nobody's priority,
but this is real. Mitigation: the learning and the implementation stand either way, and partial
credit exists via smaller PRs in the same repo.

**24 GB is too tight.** Issue #92 reports OOM on better hardware. Mitigation: day-one spike on the
smallest supported model before any commitment.

**Quadratic blowup makes it impractical beyond toy prompts.** Mitigation: this is the contribution,
not the obstacle. Anthropic flagged low-rank structure as the way through.

**The PR is never merged.** Maintainers may be slow or may want it done differently. Mitigation:
engage on the issue before writing code, and a public fork with a good writeup still demonstrates
everything even unmerged.

---

## First step

A one-day spike, no commitment. Install circuit-tracer on worker-1, generate an attribution graph
on Qwen-3-0.6B end to end, and record peak VRAM. If it fits, the project is viable and we design it
properly. If it OOMs at the smallest supported model, we know before investing anything.

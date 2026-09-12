# Method

Derivation notes for QK attribution and head loadings. The method is due to Anthropic,
[Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html).
This file records the working understanding used to implement it, not a new contribution.

## Why attribution graphs need this

An attribution graph is built from the model's own attention patterns. Attention probabilities are
cached and held constant, and the graph's edges are computed through the OV pathway only. The
consequence is a specific blind spot: the graph can say that information moved from position p_s to
position p_t, but not why the model chose to attend there. Query-key computation is outside the
graph entirely.

The same asymmetry weakens evaluation. Attribution runs with attention frozen while intervention
swaps run with attention free, which plausibly accounts for part of the residual error that the
original method reports.

## The bilinear structure

For head h, the pre-softmax attention score between a query at position q and a key at position k is

    s^h(q, k) = x_q^T W_Q^h(T) W_K^h x_k / sqrt(d_h)
              = x_q^T W_QK^h x_k / sqrt(d_h)

where W_QK^h = W_Q^h(T) W_K^h and x denotes the residual stream.

This is bilinear. It is linear in x_q when x_k is held fixed, and linear in x_k when x_q is held
fixed. That is the property the whole method rests on, because a bilinear form distributes over any
decomposition of its two arguments.

## What modern architectures change

The single matrix W_QK holds only for a model with learned absolute positions and no query/key
normalisation. Neither describes the models circuit-tracer supports. Rotary embeddings rotate the
query and key by a position-dependent angle before the dot product, and Qwen3 and Gemma-3 also apply
RMSNorm to the projected query and key.

Both are linear or scalar, so bilinearity survives and the form becomes

    s^h(q, k) = x_q^T M^h(q - k) x_k / ( sqrt(d_h) * sigma_q(q) * sigma_k(k) )

    M^h(d) = W_Q^h diag(w_q) R(d) diag(w_k) W_K^h(T)

The QK-norm gains w_q and w_k fold into the projections because they scale each channel by a fixed
amount. R(d) is the rotary rotation for a position offset d, so the operator depends on the offset
between the two positions rather than being one matrix. RoPE's relative property is what keeps this
to 2*n_pos - 1 distinct operators instead of one per position pair. The sigma terms are the QK-norm
RMS scales, which are per-position scalars and are frozen the way circuit-tracer already freezes
layernorm scales.

Verified on Qwen3-0.6B to 2.5e-07 median relative error across all 448 (layer, head) pairs; see
`experiments/003-exact-qk-form.md`. Implemented in `qk_attribution.scores`.

Attention-score soft-capping, which Gemma-2 uses, is the one feature that does not reduce this way.
It applies a tanh to the score itself, so it sits outside the form rather than rescaling it.
`scores.require_supported` refuses such models.

## Decomposing into feature pairs

Substitute a sparse decomposition of the residual stream at both positions. Writing the query-side
reconstruction as a sum over features i with activation a_i(q) and decoder direction v_i, and
likewise for key-side features j:

    x_q ~= sum_i a_i(q) v_i  + error_q + bias_q
    x_k ~= sum_j a_j(k) v_j  + error_k + bias_k

Expanding the bilinear form gives one term per feature pair, plus cross terms involving the error
and bias components:

    C_ij^h(q, k) = [a_i(q) * a_j(k) / (sqrt(d_h) * sigma_q(q) * sigma_k(k))]
                   * v_i^T M^h(q - k) v_j

C_ij is the contribution of the interaction between query-side feature i and key-side feature j to
the attention score for head h. It answers the question the original graph could not: this head
attended here because this feature at the query position matched that feature at the key position.

The term v_i^T M^h(d) v_j depends on the weights and the position offset, not on the input, so it
can be precomputed per head and offset. In practice the cheaper arrangement is to project each
feature vector into head space once, as v_i^T W_Q diag(w_q) and v_j^T W_K diag(w_k), and leave a
d_head x d_head rotation between them, since d_head is far smaller than d_model.

## Head loadings

Separately from QK, the existing graph edges can be split by which head carried them. For an edge
from source node s to target node t:

    L_h(s -> t) = a_s * a_t * (v_t^T W_OV^h v_s) * A^h(p_t, p_s)

where A^h(p_t, p_s) is the cached attention probability from the target position to the source
position for head h, and W_OV^h = W_O^h W_V^h. Summing L_h over heads recovers the edge weight, so
this is an exact decomposition of something the graph already computes rather than new information
about attention.

Head loadings are therefore the cheaper half and a sensible first milestone.

## Cost

The QK decomposition is quadratic in two places at once. Over all query-key position pairs it is
O(n_pos^2), and for each such pair it is O(k_q * k_k) in the number of active features at the two
positions. With TopK-style transcoders k is small, on the order of tens to low hundreds, so the
feature term is manageable. Long contexts are the real problem, and Anthropic state that pruning is
likely needed before full decomposition.

Their stated route through it: "Many QK attribution matrices are approximately low-rank, which may
permit a shorter description." Exploiting that is the open engineering question and the part of this
work that is not straight reimplementation.

The offset dependence adds a second axis to precompute over, but it does not multiply the cost of
the contraction itself: the rotation is applied to the projected feature vectors, which is
O(d_head^2) per position rather than O(d_model^2) per pair.

## Known limitations of the method

Carried over from the source, and worth stating plainly so results are not oversold.

1. Bias and error terms are less semantically precise than learned features, and can absorb
   computation that then goes unexplained.
2. Head labels reflect behaviour on a single prompt. A head's role across contexts is not
   established by one attribution.
3. The decomposition explains semantic matching but not the positional preferences that break ties
   between otherwise similar keys.
4. It decomposes the evidence entering the softmax, not the softmax decision itself, so inhibitory
   contributions are not fully captured.

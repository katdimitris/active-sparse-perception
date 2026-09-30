"""Recurrent sparse perception on top of SparseFormer.

The conv stem computes one feature map per image.  Inference then runs in rounds.
Round r instantiates its own grid of latent tokens (e.g. 9, then 16, then 25), which
gather evidence from the feature map in the Focus stage (box adjustment + point
sampling, repeated), are lifted to the cortex width and processed by the cortex
blocks.  The classifier head reads every round, so inference can stop after any of
them.

The only state carried from one round to the next is the previous round's cortex
output, M_{r-1} = H_{r-1} (the memory).  The new tokens read it with one cross-
attention before cortex block `memory_stage`:

    Z_r = V_r + alpha * CA( LN(V_r + p(R_r)),  LN(M_{r-1} + p(R_{r-1})),  LN(M_{r-1}) )

where p(.) is a linear encoding of each token's box (after Focus), so the new tokens
can address the old ones by where they looked, and alpha is a learned scalar
(initialised to 1).  Without memory the rounds are independent passes that share
weights; with a single round the model is a standard (static) SparseFormer.
"""
import torch
import torch.nn as nn

from .sparseformer import (AvgTokenHead, EarlyConvolution, LAYER_NORM, RestrictGradNorm,
                           SFUnit, init_linear_params, maybe_promote)


def tile_roi(n):
    """A non-overlapping grid of boxes that exactly covers the frame (rows x cols = n)."""
    rows = int(n ** 0.5)
    while rows > 1 and n % rows:
        rows -= 1
    cols = n // rows
    r, c = torch.arange(rows).float(), torch.arange(cols).float()
    roi = torch.zeros(rows, cols, 4)
    roi[..., 0] = (c / cols).view(1, cols)
    roi[..., 1] = (r / rows).view(rows, 1)
    roi[..., 2] = ((c + 1) / cols).view(1, cols)
    roi[..., 3] = ((r + 1) / rows).view(rows, 1)
    return roi.view(n, 4)


class RecurrentSparseFormer(nn.Module):
    def __init__(self, conv_dim=96, token_schedule=(9, 16, 25), token_sampling_points=36,
                 width_configurations=(192, 384), block_sizes=(1, 5), repeats=(4, 1),
                 drop_path_rate=0.2, num_classes=1000, head_dim=0, cortex_samples=True,
                 memory='none', memory_stage=0):
        super().__init__()
        assert memory in ('none', 'cortex_ca'), memory
        self.token_schedule = tuple(int(n) for n in token_schedule)
        self.num_rounds = len(self.token_schedule)
        self.memory = memory == 'cortex_ca' and self.num_rounds > 1
        self.memory_stage = int(memory_stage)
        self.max_round = None                # set to stop after that many rounds
        d, dc = width_configurations[0], width_configurations[-1]

        self.feat_extractor = EarlyConvolution(conv_dim)
        self.token_roi = nn.Embedding(self.token_schedule[0], 4)
        self.token_embedding = nn.Embedding(self.token_schedule[0], d)

        # [focus unit] [width transition] [cortex blocks]; the leading block of each width
        # samples the image and adjusts the boxes, the rest are plain transformer blocks
        self.layers = nn.ModuleList()
        rates = torch.linspace(0, drop_path_rate,
                               sum(b * r for b, r in zip(block_sizes, repeats))).tolist()
        k = 0
        for i, (width, n_blocks, rep) in enumerate(zip(width_configurations, block_sizes, repeats)):
            if i > 0 and width_configurations[i - 1] != width:
                self.layers.append(nn.Sequential(nn.Linear(width_configurations[i - 1], width),
                                                 LAYER_NORM(width)))
            assert rep == 1 or n_blocks == 1, 'a repeated stage has a single block'
            for b in range(n_blocks):
                self.layers.append(SFUnit(width, conv_dim, token_sampling_points,
                                          sampling_enabled=b == 0, adjusting_enabled=b == 0,
                                          repeats=rep, drop_path=rates[k:k + rep], head_dim=head_dim))
                k += rep
        self._foc = [i for i, l in enumerate(self.layers) if isinstance(l, SFUnit) and l.dim == d]
        self._trans = [i for i, l in enumerate(self.layers) if not isinstance(l, SFUnit)]
        self._ctx = [i for i, l in enumerate(self.layers) if isinstance(l, SFUnit) and l.dim != d]
        self.head = AvgTokenHead(dc, num_classes)
        self._init_weights()

        # later rounds: each has its own learned embeddings and its own grid of boxes
        if self.num_rounds > 1:
            self.spawn_embedding_g = nn.ModuleList(nn.Embedding(n, d) for n in self.token_schedule[1:])
            self.spawn_roi_g = nn.ModuleList(nn.Embedding(n, 4) for n in self.token_schedule[1:])
            with torch.no_grad():
                for e, r in zip(self.spawn_embedding_g, self.spawn_roi_g):
                    nn.init.trunc_normal_(e.weight, std=1.)
                    r.weight.copy_(tile_roi(e.weight.shape[0]))
        if self.memory:
            assert 0 <= self.memory_stage < len(self._ctx)
            n_heads = (dc // head_dim) if head_dim else (64 if dc >= 64 else dc)
            self.mem_pos = nn.Linear(4, dc)
            self.mem_q_norm, self.mem_k_norm, self.mem_v_norm = LAYER_NORM(dc), LAYER_NORM(dc), LAYER_NORM(dc)
            self.mem_attn = nn.MultiheadAttention(dc, n_heads)
            # the memory read is the only path between rounds, so its gate starts open
            self.mem_alpha = nn.Parameter(torch.ones(1))
        if not cortex_samples:
            for i in self._ctx:
                self.layers[i].strip_acquisition()

    @torch.no_grad()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear_params(m.weight, m.bias)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        for m in self.modules():
            if m is not self and hasattr(m, 'init_layer'):
                m.init_layer()
        nn.init.trunc_normal_(self.token_embedding.weight, std=1.)
        self.token_roi.weight.copy_(tile_roi(self.token_schedule[0]))

    def no_weight_decay(self):
        return ['token_roi', 'spawn_roi_g']

    def round_tokens(self, r, B):
        if r == 0:
            emb, roi = self.token_embedding.weight, self.token_roi.weight
        else:
            emb, roi = self.spawn_embedding_g[r - 1].weight, self.spawn_roi_g[r - 1].weight
        return (maybe_promote(emb.unsqueeze(1).repeat(1, B, 1)),
                maybe_promote(roi.unsqueeze(1).repeat(1, B, 1)))

    def read_memory(self, V, R, M, R_mem):
        e_cur = self.mem_pos(R.float().detach()).to(V.dtype)
        e_pre = self.mem_pos(R_mem.float().detach()).to(M.dtype)
        C = self.mem_attn(self.mem_q_norm(V + e_cur), self.mem_k_norm(M + e_pre), self.mem_v_norm(M))[0]
        return V + self.mem_alpha * C

    def forward(self, x, scale=1.):
        """Returns a list with one [B, classes] logit tensor per round."""
        RestrictGradNorm.GRAD_SCALE = scale          # AMP loss scale, for the sampling-grad clamp
        feat = maybe_promote(self.feat_extractor(x))
        B = feat.size(0)
        n_rounds = self.num_rounds if self.max_round is None else min(self.max_round, self.num_rounds)
        logits, M, R_mem = [], None, None
        for r in range(n_rounds):
            T, R = self.round_tokens(r, B)
            for i in self._foc:                           # Focus
                for rep in range(self.layers[i].repeats):
                    T, R = self.layers[i].step(feat, T, R, rep)
            H = T
            for i in self._trans:
                H = self.layers[i](H)
            R_ctx = R
            for s, i in enumerate(self._ctx):             # cortex, with the memory read
                if M is not None and s == self.memory_stage:
                    H = self.read_memory(H, R, M, R_mem)
                for rep in range(self.layers[i].repeats):
                    H, R_ctx = self.layers[i].step(feat, H, R_ctx, rep)
            logits.append(self.head(H))
            if self.memory:
                M, R_mem = H, R
        return logits

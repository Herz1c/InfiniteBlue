# The network (PLAN D6-D8).
#   encoder: transformer over tokens (global + pieces); attention gets a learned bias per head
#            from the pair relations (log dx/dy, line type, pieces between, attacks)
#   policy:  each candidate move is a query: the moving piece's state + move features, which
#            cross-attends to the tokens (bias from the destination square's relations); one logit each
#   value:   from the global token: 3 win/draw/loss logits + 1 logit for the search score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

TOK_CH, REL_CH, CAND_CH = 10, 5, 9  # must match core/src/encode.rs
VOCAB = 256  # every channel is a u8
NEG = -1e4  # attention bias for padding (finite, so bf16 cannot produce NaN)


class Block(nn.Module):
    """Pre-norm attention + MLP. With `kv` given, queries attend to `kv` (cross-attention)."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, bias, kv=None):
        b, n, d = x.shape
        h = self.n1(x)
        src = h if kv is None else kv
        split = lambda t: t.view(b, t.shape[1], self.heads, d // self.heads).transpose(1, 2)
        q = split(self.q(h))
        k, v = (split(t) for t in self.kv(src).chunk(2, dim=-1))
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):  # math is the fallback
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.o(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))


def embed_sum(tables, codes):
    """Sum of one embedding per channel: codes [..., C] -> [..., dim]."""
    return sum(t(codes[..., i]) for i, t in enumerate(tables))


# A relation (5 channels) is looked up as two combined codes, which is much cheaper on the large
# candidate x token tensor: (dx bucket, dy bucket) and (line type, pieces between, attack bits).
REL_GEO, REL_LINE = 41 * 41, 6 * 4 * 4


def rel_codes(rel):
    rel = rel.int()
    return rel[..., 0] * 41 + rel[..., 1], rel[..., 2] * 16 + rel[..., 3] * 4 + rel[..., 4]


class Net(nn.Module):
    def __init__(self, d=256, layers=8, heads=8, dec_layers=2):
        super().__init__()
        self.layers, self.dec_layers, self.heads = layers, dec_layers, heads
        self.tok_emb = nn.ModuleList(nn.Embedding(VOCAB, d) for _ in range(TOK_CH))
        self.rel_geo, self.rel_line = nn.Embedding(REL_GEO, heads * layers), nn.Embedding(REL_LINE, heads * layers)
        # Candidate features: a small embedding per channel, concatenated (channel 0 = piece index).
        self.cand_emb = nn.ModuleList(nn.Embedding(VOCAB, d // (CAND_CH - 1)) for _ in range(CAND_CH - 1))
        # One candidate bias shared by all decoder layers: the candidate x token tensor is large.
        self.crel_geo, self.crel_line = nn.Embedding(REL_GEO, heads), nn.Embedding(REL_LINE, heads)
        self.enc = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.dec = nn.ModuleList(Block(d, heads) for _ in range(dec_layers))
        self.cand_in = nn.Linear(2 * d, d)
        self.enc_norm = nn.LayerNorm(d)
        self.policy = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))

    def forward(self, tokens, pairs, cands, cand_rel, n_cands):
        """All inputs are integer tensors on the device (see core/src/encode.rs for layouts).
        Returns policy logits [B, C] (padding = NEG), WDL logits [B, 3], score logit [B]."""
        b, t, _ = tokens.shape
        c = cands.shape[1]
        dt = torch.bfloat16 if torch.is_autocast_enabled() else torch.float32
        lookup = lambda emb, idx: F.embedding(idx, emb.weight.to(dt))  # bias built directly in bf16
        key_pad = ((tokens[:, :, 0] == 0) * NEG).to(dt)[:, None, None, :]  # [B,1,1,T]

        geo, line = rel_codes(pairs)
        bias = (lookup(self.rel_geo, geo) + lookup(self.rel_line, line)).view(b, t, t, self.layers, self.heads)
        x = embed_sum(self.tok_emb, tokens)
        for layer, blk in enumerate(self.enc):
            x = blk(x, bias[:, :, :, layer].permute(0, 3, 1, 2) + key_pad)  # [B,H,T,T]
        x = self.enc_norm(x)

        mover = torch.gather(x, 1, cands[:, :, 0:1].expand(b, c, x.shape[-1]))
        feats = torch.cat([lookup(e, cands[..., i + 1]) for i, e in enumerate(self.cand_emb)], dim=-1)
        q = self.cand_in(torch.cat([mover.to(dt), feats], dim=-1))
        geo, line = rel_codes(cand_rel)
        cbias = (lookup(self.crel_geo, geo) + lookup(self.crel_line, line)).permute(0, 3, 1, 2) + key_pad  # [B,H,C,T]
        for blk in self.dec:
            q = blk(q, cbias, kv=x)
        logits = self.policy(q).squeeze(-1).float()
        valid = torch.arange(c, device=logits.device)[None, :] < n_cands[:, None]
        logits = logits.masked_fill(~valid, NEG)

        v = self.value(x[:, 0]).float()
        return logits, v[:, :3], v[:, 3]


def losses(out, target, wdl, score_target, wdl_weight=0.3):
    """Policy cross-entropy (positions without a target are skipped), BCE on the score logit
    against sigmoid(score / K), and a lightly weighted WDL cross-entropy."""
    logits, wdl_logits, score_logit = out
    has = target >= 0
    pol = F.cross_entropy(logits[has], target[has]) if has.any() else logits.sum() * 0
    val = F.binary_cross_entropy_with_logits(score_logit, score_target)
    res = F.cross_entropy(wdl_logits, wdl)
    return pol + val + wdl_weight * res, {"policy": pol.item(), "value": val.item(), "wdl": res.item()}

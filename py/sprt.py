# Pentanomial SPRT (PLAN §5). Games are played in colour-swapped pairs; a pair scores 0, 0.25,
# 0.5, 0.75 or 1 for engine A. With the pair scores' mean m and variance v over N pairs, the
# generalized SPRT log-likelihood ratio is approximately
#     LLR = N * (s1 - s0) * (2m - s0 - s1) / (2v)
# where s0, s1 are the expected scores under H0 (elo0) and H1 (elo1) in the logistic Elo model.
# Accept H1 when LLR >= log((1-beta)/alpha), H0 when LLR <= log(beta/(1-alpha)).
import math

PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)  # index = pentanomial bucket [LL, LD, WL+DD, WD, WW]


def expected_score(elo):
    return 1 / (1 + 10 ** (-elo / 400))


def elo_of(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def stats(penta):
    """Mean and variance of the pair score; a tiny prior keeps the variance positive."""
    counts = [c + 1e-3 for c in penta]
    n = sum(counts)
    mean = sum(c * s for c, s in zip(counts, PAIR_SCORES)) / n
    var = sum(c * (s - mean) ** 2 for c, s in zip(counts, PAIR_SCORES)) / n
    return mean, var, sum(penta)


def llr(penta, elo0, elo1):
    mean, var, n = stats(penta)
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def bounds(alpha=0.05, beta=0.05):
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def report(penta, elo0, elo1, alpha=0.05, beta=0.05):
    """One-line summary: score with 95% interval, Elo, LLR and the SPRT decision."""
    mean, var, n = stats(penta)
    half = 1.96 * math.sqrt(var / max(n, 1))
    lo, hi = bounds(alpha, beta)
    l = llr(penta, elo0, elo1)
    decision = "H1 accepted" if l >= hi else "H0 accepted" if l <= lo else "continue"
    return (f"score {mean:.3f} [{mean - half:.3f}, {mean + half:.3f}]  Elo {elo_of(mean):+.0f}  "
            f"LLR {l:+.2f} ({lo:.2f}, {hi:.2f}) [{elo0}, {elo1}]: {decision}"), decision

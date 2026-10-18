# Pre-registered checkpoint selection (PROGRESS E6): among the run's snapshots ckpt_<step>.pt,
# take the one with the lowest validation total loss (policy + value + 0.3 * wdl) logged at that
# step; ties go to the later step. Match results are never consulted.
#   python tools/select_ckpt.py ~/data/runs/m2a      -> prints the chosen path, writes selected.json
import glob
import json
import os
import sys

run = sys.argv[1]
evals = {}
for line in open(os.path.join(run, "log.jsonl")):
    rec = json.loads(line)
    if "eval" in rec:
        e = rec["eval"]
        evals[rec["step"]] = (e["policy"] + e["value"] + 0.3 * e["wdl"], e)
snaps = {int(os.path.basename(p)[5:12]): p for p in glob.glob(os.path.join(run, "ckpt_*.pt"))}
scored = [(evals[s][0], -s, s) for s in snaps if s in evals]
if not scored:
    sys.exit(f"MISSING: no snapshot in {run} has a validation eval, stopping")
loss, _, step = min(scored)
out = {"path": snaps[step], "step": step, "val_total_loss": loss, "val": evals[step][1],
       "candidates": {str(s): round(evals[s][0], 5) for s in sorted(snaps) if s in evals}}
json.dump(out, open(os.path.join(run, "selected.json"), "w"), indent=1)
print(snaps[step])

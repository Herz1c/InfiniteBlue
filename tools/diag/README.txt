Diagnostic and benchmark scripts used during development (kept for provenance; see PROGRESS.md).
They run inside WSL with ~/venv/ic. Some expect files such as /tmp/req.jsonl or a smoke checkpoint;
read the top of each script before running it.
  gpu_bench.py      transformer inference throughput by model size
  dg_bench.sh       Apeiron data_gen throughput at depth 9 and 12
  enc_test.py       icore.encode output check and speed
  speed.py          training-step time and profiler breakdown
  search_*.py, batch_test.py, eng_time*.py   MCTS / engine speed
  terminal_agree.py Apeiron sees the referee's final positions as terminal?
  rep_debug.py, miss_detail.py   debugging a missed repetition / coverage misses

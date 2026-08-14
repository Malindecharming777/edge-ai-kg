# Demo

A narrated walkthrough in six beats: what the graph holds, what silently falls
back to the CPU, what that costs, what quantization unlocks, the blast radius of
losing one kernel, and the full electrode-to-silicon path.

```bash
python -m demo.demo            # embedded engine — no server needed
python -m demo.demo --fast     # no typing pauses
python -m demo.demo --url http://127.0.0.1:8080
```

The demo builds its own graph in-process if one isn't already loaded, so it runs
from a clean checkout after `python -m etl.download_data`.

Record / regenerate the asciinema cast:

```bash
asciinema rec --overwrite --cols 92 --rows 32 --idle-time-limit 2.0 \
  -c "bash -c 'python -m demo.demo'" demo/edgeai.cast
agg demo/edgeai.cast demo/edgeai.gif
```

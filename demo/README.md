# Demo

Two walkthroughs, both able to run against an **embedded** engine with no
server (`python -m etl.download_data` first, to build `data/`).

## `demo.questions` — the whole catalog, as questions

![Edge AI KG — 12 questions answered](edgeai-questions.gif)

Walks all 12 queries in `benchmarks/queries.py`: the plain-English question, the
Cypher it becomes, the answer, and why the question is awkward without a graph.
This is what the GIF above records.

```bash
python -m demo.questions                          # embedded
python -m demo.questions --url http://127.0.0.1:8080
python -m demo.questions --only EA01 EA06         # just these
python -m demo.questions --fast                   # no pacing
```

## `demo.demo` — the story, in six beats

What the graph holds, what silently falls back to the CPU, what that costs,
what quantization unlocks, the blast radius of losing one kernel, and the full
electrode-to-silicon path.

```bash
python -m demo.demo            # embedded
python -m demo.demo --fast
python -m demo.demo --url http://127.0.0.1:8080
```

## Re-recording the GIF

Requires [`asciinema`](https://asciinema.org) and
[`agg`](https://github.com/asciinema/agg) (`cargo install --git
https://github.com/asciinema/agg`).

```bash
asciinema rec --overwrite --cols 100 --rows 34 --idle-time-limit 2.0 \
  -c "python -m demo.questions --url http://127.0.0.1:8080" \
  demo/edgeai-questions.cast

agg --theme github-dark --font-size 15 --line-height 1.35 \
    --speed 1.4 --idle-time-limit 1.2 --fps-cap 12 --last-frame-duration 4 \
    demo/edgeai-questions.cast demo/edgeai-questions.gif
```

The `.cast` is committed alongside the `.gif`, so the GIF can be re-rendered at
different sizes or themes without re-running the demo.

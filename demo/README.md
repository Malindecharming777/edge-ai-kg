# Demo

Two walkthroughs, both able to run against an **embedded** engine with no
server (`python -m etl.download_data` first, to build `data/`).

## `demo.questions` — the whole catalog, as questions

![Edge AI KG — 16 questions answered](edgeai-questions.gif)

Walks all 16 queries in `benchmarks/queries.py`: the plain-English question, the
Cypher it becomes, the answer, and why the question is awkward without a graph.
EA13-EA16 run on the real ONNX Runtime and MLPerf Tiny layers, so their answers
are checkable against the upstream sources. This is what the GIF above records.

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

The GIF is **long-form**: a tall terminal so the entire 16-question run renders
in one vertical image with nothing scrolling off — the convention shared with
`samyama-graph/case_studies/_lib/record_gif.sh`. 100 columns at font-size 18
gives the 1105 px width every other case-study and `-kg` GIF uses; the height
is whatever the run needs (currently 10,987 px for 16 questions).

```bash
PYTHON=.venv/bin/python SG_URL=http://127.0.0.1:8080 scripts/record_gif.sh
scripts/record_gif.sh                      # defaults: demo.questions -> demo/edgeai-questions.gif
GIF_ROWS=400 scripts/record_gif.sh         # taller terminal
SG_URL= scripts/record_gif.sh              # record against the embedded engine
PYTHON=.venv/bin/python scripts/record_gif.sh
```

The script measures the demo's output first and raises `GIF_ROWS` if the run is
taller than the terminal, since a short terminal silently scrolls the opening
off the top.

Requires [`asciinema`](https://asciinema.org) (`pip install asciinema`) and
[`agg`](https://github.com/asciinema/agg)
(`cargo install --git https://github.com/asciinema/agg` — note the `agg` on
crates.io is an unrelated library with no binary).

The `.cast` is committed alongside the `.gif`, so the GIF can be re-rendered at
a different size or theme without re-running the demo.

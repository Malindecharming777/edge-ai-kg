#!/usr/bin/env bash
# Record the question-driven demo to a long-form animated GIF.
#
# "Long-form" follows the house convention in
# samyama-graph/case_studies/_lib/record_gif.sh: record a TALL terminal so the
# whole narrated run renders in one vertical image with nothing scrolling off,
# then render at font-size 18 / theme asciinema (100 cols -> 1105 px wide,
# matching every other case-study and -kg GIF).
#
#   scripts/record_gif.sh [demo_module] [out.gif]
#
# Env:
#   GIF_ROWS  terminal height in rows; must be >= the demo's total output lines
#             or the top scrolls away. Default 332 (a full 12-question run).
#   PYTHON    interpreter with rich + samyama available.
#   SG_URL    engine URL; omit to run the demo against an embedded engine.
set -uo pipefail

MODULE="${1:-demo.questions}"
OUT="${2:-demo/edgeai-questions.gif}"
CAST="${OUT%.gif}.cast"
ROWS="${GIF_ROWS:-332}"
PYTHON="${PYTHON:-python3}"
SG_URL="${SG_URL:-http://127.0.0.1:8080}"

for tool in asciinema agg; do
  command -v "$tool" >/dev/null || {
    echo "[gif] '$tool' not installed — skipping."
    echo "[gif]   pip install asciinema"
    echo "[gif]   cargo install --git https://github.com/asciinema/agg"
    exit 0
  }
done

URL_ARG=""
[ -n "$SG_URL" ] && URL_ARG="--url $SG_URL"

# Sanity-check the height: if the demo prints more lines than the terminal has
# rows, the beginning scrolls off and the long-form format is defeated.
LINES_OUT=$(COLUMNS=100 $PYTHON -m "$MODULE" $URL_ARG --fast 2>/dev/null | wc -l)
if [ "$LINES_OUT" -gt "$ROWS" ]; then
  echo "[gif] demo prints $LINES_OUT lines but GIF_ROWS=$ROWS — raising to fit."
  ROWS=$((LINES_OUT + 8))
fi
echo "[gif] recording $MODULE at 100x${ROWS} -> $CAST"

rm -f "$CAST"
COLUMNS=100 LINES="$ROWS" TERM=xterm-256color asciinema rec --overwrite -q -i 6 \
  --cols 100 --rows "$ROWS" \
  -c "env COLUMNS=100 LINES=$ROWS TERM=xterm-256color $PYTHON -m $MODULE $URL_ARG" \
  "$CAST" || { echo "[gif] asciinema failed"; exit 1; }

echo "[gif] rendering $CAST -> $OUT"
# speed 1.0 (real time) so the demo's read-pauses survive; a looping GIF can't
# be paused, so reading time has to live in the frames.
agg --speed 1.0 --idle-time-limit 4.0 --last-frame-duration 8 \
    --font-size 18 --theme asciinema "$CAST" "$OUT" \
  || { echo "[gif] agg failed"; exit 1; }

echo "[gif] wrote $OUT ($(du -h "$OUT" | cut -f1))"

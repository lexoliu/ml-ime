#!/bin/bash
# Evaluate a finished route A v2 segment: fuse its score files with the run3
# trigram on every eval set, context on and off, and write one results table.
#
#   kaggle/finish.sh data/route-a-v2/s6
#
# The segment directory is a kernel output that reported "finished": true and
# holds scores-lattice*-context-{on,off}.jsonl.gz. For each (lattice, context)
# the fusion weight is swept on the dev slice and the chosen weight is reported
# on the test slice; the trigram-only and neural-only rows come from the same
# binary. Results go to <segment>/results.md and are also printed.
#
# Needs: target/release/ime-cli built from dev, data/run3/ngram.bin (the 41M-line
# run3 trigram), data/run3_pool/eval3{,-abbreviated,-mixed}.jsonl and
# data/route-a-assets-v2/emittable.txt. The abbreviated test slice takes ~20 min
# on the M1; the whole script about 1.5 h.
set -eu
REPO=${MLIME_REPO:-/Users/lexoliu/Coding/ml-ime}
SEG=${1:?segment directory}
BIN=$REPO/target/release/ime-cli
NGRAM=$REPO/data/run3/ngram.bin
EMIT=$REPO/data/route-a-assets-v2/emittable.txt
OUT=$SEG/results.md
WEIGHTS="0.5 0.75 1 1.5 2"

for f in "$BIN" "$NGRAM" "$EMIT"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done

top1() { grep "sentence, top-1" | awk '{print $3}'; }
row() { # label, report text -> markdown row
  local label=$1 text=$2
  printf '| %s | %s | %s | %s | %s |\n' "$label" \
    "$(printf '%s' "$text" | grep 'sentence, top-1' | awk '{print $3}')" \
    "$(printf '%s' "$text" | grep 'sentence, top-8' | awk '{print $3}')" \
    "$(printf '%s' "$text" | grep '^character' | awk '{print $2}')" \
    "$(printf '%s' "$text" | grep '^MRR' | awk '{print $2}')"
}

{
  echo "# Route A v2 results ($(basename "$SEG"), $(date '+%Y-%m-%d'))"
  echo
  echo "Test slice of each eval set (5,0xx records); fusion weight chosen on the dev slice."
  echo
  for lattice in lattice lattice-abbreviated lattice-mixed; do
    case $lattice in
      lattice) set=$REPO/data/run3_pool/eval3.jsonl; name=full ;;
      lattice-abbreviated) set=$REPO/data/run3_pool/eval3-abbreviated.jsonl; name=abbreviated ;;
      lattice-mixed) set=$REPO/data/run3_pool/eval3-mixed.jsonl; name=mixed ;;
    esac
    echo "## $name typing"
    echo
    echo "| configuration | top-1 | top-8 | char | MRR@8 |"
    echo "|---|---:|---:|---:|---:|"
    base=$("$BIN" fused-eval --model "$NGRAM" --eval-set "$set" --emittable "$EMIT" --slice test 2>/dev/null)
    row "trigram only" "$base"
    for context in off on; do
      scores=$SEG/scores-$lattice-context-$context.jsonl.gz
      [ -e "$scores" ] || { echo "| (no $scores) | | | | |"; continue; }
      neural=$("$BIN" fused-eval --no-transition --eval-set "$set" --emittable "$EMIT" --slice test --scores "$scores" 2>/dev/null)
      row "neural only, context $context" "$neural"
      sweep=$("$BIN" fused-eval --model "$NGRAM" --eval-set "$set" --emittable "$EMIT" --slice dev --scores "$scores" $(for w in $WEIGHTS; do printf -- '--weight %s ' "$w"; done) 2>/dev/null)
      best=$(printf '%s' "$sweep" | awk '/^emission/{w=$4} /sentence, top-1/{print $3, w}' | sort -rn | head -1 | awk '{print $2}' | tr -d ';')
      fused=$("$BIN" fused-eval --model "$NGRAM" --eval-set "$set" --emittable "$EMIT" --slice test --scores "$scores" --weight "$best" 2>/dev/null)
      row "fused, context $context, w=$best" "$fused"
    done
    echo
  done
} | tee "$OUT"

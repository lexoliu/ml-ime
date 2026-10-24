#!/usr/bin/env bash
# Run one bounded chunk of a streamed corpus source (#225), resuming where the last one stopped.
#
# Usage: scripts/stream-corpus.sh <source> <files> <data dir>
#
#   <source>    fineweb2-hq | mnbvc-forum | yyets | fineweb2-mined
#   <files>     the most upstream files this chunk downloads and processes
#   <data dir>  the data root holding run3/, run1_pool/ and run3_pool/;
#               samples and run state go to <data dir>/corpus-225
#
# Every chunk deduplicates against run3's targets and drops anything shared with
# the evaluation and held-out records. fineweb2-hq stops once it has written 14M
# targets; fineweb2-mined is the 10% pass, the first 37 of FineWeb-2's 370 Mandarin
# files in spread order, and reads #215's entry table from
# <data dir>/corpus-225/inputs/entries-215.tsv; its shards wait in
# <data dir>/corpus-225/ingest/fineweb2-mined/unjudged until `mlime corpus judge`
# writes the records it keeps into samples/. A chunk stops cleanly before a
# download would leave less than 60 GB free. Build first:
# cargo build --release -p ime-cli
set -euo pipefail

if [[ $# -ne 3 ]]; then
  sed -n '2,19p' "$0" >&2
  exit 2
fi
source=$1
files=$2
data=$(cd "$3" && pwd)
repo=$(cd "$(dirname "$0")/.." && pwd)
bin=$repo/target/release/ime-cli
root=$data/corpus-225

evals=()
for file in "$data"/run1_pool/eval.jsonl "$data"/run1_pool/eval2.jsonl "$data"/run1_pool/pool.jsonl \
  "$data"/run3_pool/eval3.jsonl "$data"/run3_pool/eval3-abbreviated.jsonl \
  "$data"/run3_pool/eval3-mixed.jsonl "$data"/run3_pool/pool.jsonl "$data"/run3_pool/exclusions.jsonl; do
  evals+=(--eval "$file")
done
common=(--data-dir "$root" --files "$files" --run3 "$data/run3/samples" --floor-gb 60 "${evals[@]}")

case $source in
  fineweb2-hq)
    exec "$bin" corpus ingest --source fineweb2-hq --targets 14000000 "${common[@]}"
    ;;
  mnbvc-forum | yyets)
    exec "$bin" corpus ingest --source "$source" "${common[@]}"
    ;;
  fineweb2-mined)
    exec "$bin" corpus mine --upstream-files 37 --entries "$root/inputs/entries-215.tsv" "${common[@]}"
    ;;
  *)
    echo "unknown source: $source" >&2
    exit 2
    ;;
esac

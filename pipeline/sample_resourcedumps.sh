#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# sample_resourcedumps.sh
# Randomly copy N unzipped resource dump folders from source to destination
# ==========================================================

# --- Defaults ---
N=20
SRC="/mnt/nas-lae/cinii-kg/resourcedump_rdf_product_20240404"
DEST="/home/fdmartino/work/cinii/CiNii_classification/data/raw"
LIST_FILE="selected_folders.txt"
JOBS=4          # parallel rsync jobs
SEED=""         # optional: reproducible sampling if set

usage() {
  cat <<EOF
Usage: $0 [--n N] [--src PATH] [--dest PATH] [--list FILE] [--jobs J] [--seed S]
EOF
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --list) LIST_FILE="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown parameter: $1" >&2; usage; exit 1 ;;
  esac
done

# --- Check paths ---
if [[ ! -d "$SRC" ]]; then
  echo "❌ Source directory not found: $SRC" >&2
  exit 1
fi
mkdir -p "$DEST"

LIST_PATH="$DEST/$LIST_FILE"

# --- Build candidate list (directories named resourcedump_*, excluding *.zip) ---
# Using find avoids ls parsing issues and handles weird filenames safely.
mapfile -t CANDIDATES < <(
  find "$SRC" -maxdepth 1 -type d -name 'resourcedump_*' -print0 \
    | tr '\0' '\n' \
    | grep -vE '\.zip$' \
    | sort
)

TOTAL="${#CANDIDATES[@]}"
if (( TOTAL == 0 )); then
  echo "❌ No matching directories found in: $SRC" >&2
  exit 1
fi

if (( N > TOTAL )); then
  echo "⚠️  Requested N=$N but only $TOTAL folders available. Using N=$TOTAL."
  N="$TOTAL"
fi

echo "📋 Selecting $N random unzipped folders from:"
echo "   $SRC"

# --- Random sampling ---
# If seed provided, make selection reproducible.
if [[ -n "$SEED" ]]; then
  printf "%s\n" "${CANDIDATES[@]}" \
    | awk -v seed="$SEED" 'BEGIN{srand(seed)} {print rand() "\t" $0}' \
    | sort -n \
    | head -n "$N" \
    | cut -f2- \
    > "$LIST_PATH"
else
  printf "%s\n" "${CANDIDATES[@]}" | shuf -n "$N" > "$LIST_PATH"
fi

echo "✅ List saved to: $LIST_PATH"
echo "------------------------------"
cat "$LIST_PATH"
echo "------------------------------"

read -r -p "⚠️  Proceed to copy these folders to $DEST ? (y/N): " CONFIRM
if [[ "${CONFIRM:-}" != "y" && "${CONFIRM:-}" != "Y" ]]; then
  echo "❌ Aborted."
  exit 0
fi

echo "🚀 Copying files with $JOBS parallel jobs..."

# --- Parallel rsync ---
# -a: archive, preserves structure
# --info=progress2: single overall progress per rsync process
# --human-readable: readable sizes
# --partial: keep partial transfers if interrupted
# --inplace: can help on some filesystems (optional; remove if you prefer)
xargs -P "$JOBS" -I {} rsync -a --human-readable --info=progress2 --partial "{}" "$DEST/" \
  < "$LIST_PATH"

echo "✅ Done! Copied $N folders to:"
echo "   $DEST"

#!/usr/bin/env bash
# Gal-DINO for the comparison: clone the official repo at the tested commit, apply two
# compatibility patches (newer torch), add our scripts, create a venv, build the deformable
# attention CUDA op and link the dataset in the layout main.py expects.
#
# Needs: git, uv, nvcc 12.x (CUDA_HOME), data/RadioGalaxyNET downloaded (python data/download.py).
# The Gal-DINO checkpoint is a manual download, see README.md.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
DATA=$(cd .. && pwd)/data/RadioGalaxyNET
COMMIT=d108208d1d51ef8d18333536a1bfb0b9b578cdea  # Nikhel1/Gal-DINO master, "Merge pull request #5" (2024)

[ -d "$DATA/test" ] || { echo "missing $DATA, run: python data/download.py"; exit 1; }
[ -d Gal-DINO ] || git clone https://github.com/Nikhel1/Gal-DINO.git
cd Gal-DINO
git checkout -q "$COMMIT"
if git apply --check ../patches.diff 2>/dev/null; then git apply ../patches.diff; fi
cp ../{galdino_common,trt_runner,build_trt,export_onnx,eval_backends,latency,dump_eval}.py .
cp ../../score.py .

[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r ../requirements.txt
uv pip install --python .venv/bin/python --no-build-isolation ./models/dino/ops

# main.py reads <coco_path>/{train2017,val2017,test2017} + annotations/instances_<split>2017.json
# and always evaluates the test split; RadioGalaxyNET_val_as_test points "test" at val.
mkdir -p RadioGalaxyNET/annotations RadioGalaxyNET_val_as_test/annotations outputs_gal
for s in train val test; do
  ln -sfn "$DATA/$s" "RadioGalaxyNET/${s}2017"
  ln -sfn "$DATA/annotations/$s.json" "RadioGalaxyNET/annotations/instances_${s}2017.json"
done
ln -sfn "$DATA/train" RadioGalaxyNET_val_as_test/train2017
ln -sfn "$DATA/val" RadioGalaxyNET_val_as_test/test2017
ln -sfn "$DATA/annotations/train.json" RadioGalaxyNET_val_as_test/annotations/instances_train2017.json
ln -sfn "$DATA/annotations/val.json" RadioGalaxyNET_val_as_test/annotations/instances_test2017.json

[ -f outputs_gal/checkpoint.pth ] || echo "NOTE: put the Gal-DINO checkpoint at $HERE/Gal-DINO/outputs_gal/checkpoint.pth"
echo "Gal-DINO ready in $HERE/Gal-DINO"

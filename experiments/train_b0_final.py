
"""B0-final: training on full train + val."""

import json
import runpy
import time
from pathlib import Path

from catboost import CatBoostRanker, Pool

ROOT = Path("/home/ubuntu/avito")
CACHE = ROOT / "experiments/cache/B0_full"
RUNS = ROOT / "experiments/runs"
OUT = RUNS / "B0_final_gpu"

assert not OUT.exists(), f"Output already exists: {OUT}"

train_file = CACHE / "train.tsv"
val_file = CACHE / "val.tsv"
cd_file = CACHE / "features.cd"

train_state = json.loads(
    (CACHE / "train.checkpoint.json").read_text()
)
val_state = json.loads(
    (CACHE / "val.checkpoint.json").read_text()
)

assert train_state["next"] == 512
assert val_state["next"] == 64

original_bytes = train_state["bytes"]

assert train_file.stat().st_size >= original_bytes
assert val_file.stat().st_size == val_state["bytes"]

# Берём оптимальное число деревьев из готового B0-full.
b0_run = json.loads(
    (RUNS / "B0_full_gpu/run.json").read_text()
)
trees = int(b0_run["metrics"]["trees"])
assert trees == 144

# Загружаем конфигурацию B0.
base = runpy.run_path(
    str(ROOT / "experiments/catboost_b0_full_gpu.py"),
    run_name="load_config",
)

params = dict(base["MODEL_PARAMS"])
params["num_trees"] = trees

# В финальной модели отдельной validation нет.
params.pop("early_stopping_rounds", None)
params.pop("eval_metric", None)

assert params["task_type"] == "GPU"
assert params["devices"] == "0:1"
assert params["data_partition"] == "DocParallel"
assert params["gpu_ram_part"] == 0.95

start = time.monotonic()

print("Appending val to train temporarily...", flush=True)

try:
    with train_file.open("r+b") as dst:
        # Восстанавливаем исходный размер, если раньше
        # произошло незавершённое объединение.
        dst.truncate(original_bytes)
        dst.seek(original_bytes)

        with val_file.open("rb") as src:
            count = 0

            for line in src:
                label, group, features = line.split(b"\t", 2)

                dst.write(
                    label + b"\tv_" + group + b"\t" + features
                )
                count += 1

    assert count == val_state["rows"]

    print("Loading train+val Pool...", flush=True)

    pool = Pool(
        str(train_file),
        column_description=str(cd_file),
        thread_count=8,
    )

finally:
    # Возвращаем train.tsv к первоначальному размеру.
    with train_file.open("r+b") as dst:
        dst.truncate(original_bytes)

    print("Original train.tsv restored.", flush=True)

expected = train_state["rows"] + val_state["rows"]
assert pool.num_row() == expected

print(
    f"TRAINING: {expected:,} candidates, "
    f"{trees} trees, GPUs 0:1",
    flush=True,
)

model = CatBoostRanker(**params)

model.fit(
    pool,
    verbose=20,
    use_best_model=False,
)

assert model.tree_count_ == trees

OUT.mkdir(parents=True, exist_ok=False)
model.save_model(str(OUT / "model.cbm"))

report = {
    "trained_on": "full train+val",
    "train_candidates": expected,
    "train_feeds": (
        train_state["feeds"] + val_state["feeds"]
    ),
    "trees": model.tree_count_,
    "features": base["FEATURES"],
    "model_params": model.get_all_params(),
    "elapsed_seconds": time.monotonic() - start,
    "validation_metric": None,
}

(OUT / "run.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

print("DONE:", OUT, flush=True)
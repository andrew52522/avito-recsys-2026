
"""Full B0 training with shard-by-shard feature preparation."""

import gc
import json
import os
import runpy
import shutil
import time

from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from catboost import CatBoostRanker, Pool


ROOT = Path("/home/ubuntu/avito")
BASE = runpy.run_path(
    str(ROOT / "experiments/catboost_b0_full_gpu.py"),
    run_name="load_baseline",
)

DATA = ROOT / "competition"
CACHE = ROOT / "experiments/cache/B0_full"
OUTPUT = ROOT / "experiments/runs/B0_full_gpu"

CACHE.mkdir(parents=True, exist_ok=True)

FEATURES = BASE["FEATURES"]
CANDIDATE_COLUMNS = BASE["CANDIDATE_COLUMNS"]
EMBEDDING_FEATURES = BASE["EMBEDDING_FEATURES"]

assert BASE["MODEL_PARAMS"]["task_type"] == "GPU"
assert BASE["DATA_PARAMS"] == {
    "train_feeds": 0,
    "val_feeds": 0,
}
assert not OUTPUT.exists(), "Модель уже существует"

# Сохраняем порядок признаков официального B0.
cd_path = CACHE / "features.cd"
lines = ["0\tLabel", "1\tGroupId"]

for index, feature in enumerate(FEATURES, start=2):
    kind = "Categ" if feature in BASE["CAT_FEATURES"] else "Num"
    lines.append(f"{index}\t{kind}\t{feature}")

cd_path.write_text("\n".join(lines) + "\n")


def prepare_split(split_name):
    split = DATA / split_name
    files = sorted((split / "impressions").glob("*.parquet"))

    assert files, f"Нет Parquet-файлов для {split_name}"

    output_file = CACHE / f"{split_name}.tsv"
    checkpoint = CACHE / f"{split_name}.checkpoint.json"

    # Можно продолжить подготовку после прерывания.
    state = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else {"next": 0, "bytes": 0, "rows": 0, "feeds": 0}
    )

    if state["next"] > 0:
        assert output_file.exists(), "Нет файла для checkpoint"

    meta = json.loads(
        (split / "item_embeddings/data.meta").read_text()
    )

    embeddings = np.memmap(
        split / "item_embeddings/data.memmap",
        mode="r",
        dtype=meta["dtype"],
        shape=tuple(meta["shape"]),
        order=meta["order"],
    )

    mode = "r+b" if output_file.exists() else "w+b"

    print(
        f"\nPREPARING {split_name}: "
        f"{len(files)} files, resume from {state['next']}",
        flush=True,
    )

    with output_file.open(mode) as handle:
        # Удаляем возможный незавершённый кусок прошлого запуска.
        handle.truncate(state["bytes"])
        handle.seek(state["bytes"])

        for i in range(state["next"], len(files)):
            source = pl.scan_parquet(str(files[i]))

            columns = [*CANDIDATE_COLUMNS, "target"]

            source = source.select(
                "feed_id", "user_embedding", *columns
            )

            candidates = (
                source.with_columns(
                    pl.struct(
                        "user_embedding",
                        "item_id__embedding_idx",
                    )
                    .map_batches(
                        partial(
                            BASE["embedding_features"],
                            embeddings=embeddings,
                        ),
                        return_dtype=EMBEDDING_FEATURES,
                        is_elementwise=True,
                    )
                    .alias("embedding_features")
                )
                .unnest("embedding_features")
                .explode([
                    *columns,
                    "embedding_dot",
                    "embedding_cosine",
                    "embedding_missing",
                ])
                .select("target", "feed_id", *FEATURES)
            )

            # В RAM находится только один исходный Parquet-файл.
            chunk = candidates.collect(streaming=True)

            feeds = chunk["feed_id"].n_unique()

            chunk.write_csv(
                handle,
                include_header=False,
                separator="\t",
            )

            handle.flush()
            os.fsync(handle.fileno())

            state = {
                "next": i + 1,
                "bytes": handle.tell(),
                "rows": state["rows"] + chunk.height,
                "feeds": state["feeds"] + feeds,
            }

            tmp = checkpoint.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(checkpoint)

            del chunk
            gc.collect()

            if (i + 1) % 8 == 0 or i + 1 == len(files):
                free_gb = shutil.disk_usage(CACHE).free / 2**30
                print(
                    f"{split_name}: {i+1}/{len(files)} files; "
                    f"{state['rows']:,} candidates; "
                    f"disk free {free_gb:.1f} GiB",
                    flush=True,
                )

                if free_gb < 3:
                    raise RuntimeError("Недостаточно места на диске")

    print(f"{split_name} READY: {state}", flush=True)
    return output_file, state


start = time.perf_counter()

train_path, train_info = prepare_split("train")
val_path, val_info = prepare_split("val")

print("\nLOADING CATBOOST POOLS FROM DISK", flush=True)

train_pool = Pool(
    str(train_path),
    column_description=str(cd_path),
    thread_count=8,
)

val_pool = Pool(
    str(val_path),
    column_description=str(cd_path),
    thread_count=8,
)

assert train_pool.num_col() == len(FEATURES)
assert val_pool.num_col() == len(FEATURES)

print(
    "TRAINING ON GPU",
    "train rows =", train_pool.num_row(),
    "val rows =", val_pool.num_row(),
    flush=True,
)

model = CatBoostRanker(**BASE["MODEL_PARAMS"])
model.fit(train_pool, eval_set=val_pool, **BASE["FIT_PARAMS"])

print("\nCALCULATING METRICS", flush=True)

quality = model.eval_metrics(
    val_pool,
    ["NDCG", "NDCG:top=30", "NDCG:top=10"],
    eval_period=model.tree_count_,
    thread_count=8,
)

metrics = {
    "val_ndcg": quality["NDCG:type=Base"][-1],
    "val_ndcg_at_30": quality["NDCG:top=30;type=Base"][-1],
    "val_ndcg_at_10": quality["NDCG:top=10;type=Base"][-1],
    "train_feeds": train_info["feeds"],
    "train_candidates": train_pool.num_row(),
    "val_feeds": val_info["feeds"],
    "val_candidates": val_pool.num_row(),
    "trees": model.tree_count_,
}

OUTPUT.mkdir(parents=True, exist_ok=False)
model.save_model(str(OUTPUT / "model.cbm"))

report = {
    "metrics": metrics,
    "features": FEATURES,
    "model_params": model.get_all_params(),
    "elapsed_seconds": time.perf_counter() - start,
}

(OUTPUT / "run.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

print("\nDONE:", metrics, flush=True)
print("MODEL:", OUTPUT / "model.cbm", flush=True)

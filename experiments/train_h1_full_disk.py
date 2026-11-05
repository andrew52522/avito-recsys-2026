"""Full Avito RecSys 2026 H1 training using disk-based candidate feature cache.

CPU-only feature preparation:
    PREPARE_ONLY=1 python -u experiments/train_h1_full_disk.py
GPU training (two RTX 4090):
    python -u experiments/train_h1_full_disk.py

Uses *all* train and val candidates, with no sampling.
Requires sibling experiments/catboost_h1_full_gpu.py; uses its config.
Does not touch or delete the existing B0 cache.
"""

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
H1_SCRIPT = ROOT / "experiments/catboost_h1_full_gpu.py"
DATA = ROOT / "competition"
CACHE = ROOT / "experiments/cache/H1_full"
OUTPUT = ROOT / "experiments/runs/H1_full_gpu"

H1 = runpy.run_path(str(H1_SCRIPT), run_name="h1_config")
FEATURES = H1["FEATURES"]
CANDIDATE_COLUMNS = H1["CANDIDATE_COLUMNS"]
EMBEDDING_FEATURES = H1["EMBEDDING_FEATURES"]

assert len(FEATURES) == 10 and "hist_cat_count" in FEATURES
assert H1["MODEL_PARAMS"]["task_type"] == "GPU"
assert H1["MODEL_PARAMS"]["devices"] == "0:1"
assert H1["MODEL_PARAMS"]["data_partition"] == "DocParallel"
assert H1["MODEL_PARAMS"]["gpu_ram_part"] == 0.95
assert H1["DATA_PARAMS"] == {"train_feeds": 0, "val_feeds": 0}


def current_free_gib() -> float:
    return shutil.disk_usage(ROOT).free / (2 ** 30)


def prepare_split(split_name: str, cd_path: Path):
    split = DATA / split_name
    files = sorted((split / "impressions").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Missing {split_name} impression files")

    output_file = CACHE / f"{split_name}.tsv"
    checkpoint = CACHE / f"{split_name}.checkpoint.json"
    state = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else {"next": 0, "bytes": 0, "rows": 0, "feeds": 0}
    )

    assert 0 <= state["next"] <= len(files), f"Bad {split_name} checkpoint"
    if state["next"] > 0:
        assert output_file.exists(), f"Missing TSV for {split_name} checkpoint"
    if output_file.exists():
        assert output_file.stat().st_size >= state["bytes"], (
            f"Truncated {split_name} TSV (smaller than checkpoint)"
        )

    meta = json.loads((split / "item_embeddings" / "data.meta").read_text())
    embeddings = np.memmap(
        split / "item_embeddings" / "data.memmap",
        mode="r",
        dtype=meta["dtype"],
        shape=tuple(meta["shape"]),
        order=meta["order"],
    )

    print(
        f"\nPREPARING {split_name}: {len(files)} files, "
        f"resume from {state['next']}",
        flush=True,
    )

    # A checkpoint is only advanced after a shard is fully written and fsync'ed.
    mode = "r+b" if output_file.exists() else "w+b"
    with output_file.open(mode) as handle:
        handle.truncate(state["bytes"])
        handle.seek(state["bytes"])

        for i in range(state["next"], len(files)):
            source = pl.scan_parquet(str(files[i])).select(
                "feed_id",
                "user_embedding",
                "history__category_id",
                *CANDIDATE_COLUMNS,
                "target",
            )
            # One list per feed; its k-th count belongs to the k-th candidate.
            source = source.with_columns(
                pl.struct("category_id", "history__category_id")
                .map_elements(
                    H1["history_category_features"],
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("hist_cat_count")
            )
            candidates = (
                source.with_columns(
                    pl.struct("user_embedding", "item_id__embedding_idx")
                    .map_batches(
                        partial(
                            H1["embedding_features"],
                            embeddings=embeddings,
                        ),
                        return_dtype=EMBEDDING_FEATURES,
                        is_elementwise=True,
                    )
                    .alias("embedding_features")
                )
                .unnest("embedding_features")
                .explode([
                    *CANDIDATE_COLUMNS,
                    "target",
                    "embedding_dot",
                    "embedding_cosine",
                    "embedding_missing",
                    "hist_cat_count",
                ])
                .select("target", "feed_id", *FEATURES)
            )
            chunk = candidates.collect(streaming=True)
            if chunk.is_empty():
                raise RuntimeError(f"Empty shard: {files[i]}")
            if chunk["hist_cat_count"].null_count() or chunk["hist_cat_count"].min() < 0:
                raise RuntimeError(f"Invalid hist_cat_count in shard: {files[i]}")
            feeds = chunk["feed_id"].n_unique()

            chunk.write_csv(handle, include_header=False, separator="\t")
            handle.flush()
            os.fsync(handle.fileno())
            state = {
                "next": i + 1,
                "bytes": handle.tell(),
                "rows": state["rows"] + chunk.height,
                "feeds": state["feeds"] + feeds,
            }
            checkpoint_tmp = checkpoint.with_suffix(".tmp")
            checkpoint_tmp.write_text(json.dumps(state))
            checkpoint_tmp.replace(checkpoint)
            del chunk, candidates, source
            gc.collect()

            if (i + 1) % 8 == 0 or i + 1 == len(files):
                free = current_free_gib()
                print(
                    f"{split_name}: {i + 1}/{len(files)} files; "
                    f"{state['rows']:,} candidates; "
                    f"disk free {free:.1f} GiB",
                    flush=True,
                )
                if free < 5.0:
                    raise RuntimeError(
                        "Free disk < 5 GiB. Extend disk before resuming; "
                        "DO NOT delete this partial H1 cache."
                    )

    print(f"{split_name} READY: {state}", flush=True)
    return output_file, state


def main():
    start = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments/logs").mkdir(parents=True, exist_ok=True)

    # Prevent an almost-full disk from being exhausted by a new ~20+ GiB cache.
    # If preparation is already partly done, checkpoints preserve progress.
    train_checkpoint = CACHE / "train.checkpoint.json"
    if not train_checkpoint.exists() and current_free_gib() < 30:
        raise RuntimeError(
            f"Only {current_free_gib():.1f} GiB free. "
            "For H1 prepare, first extend disk so at least 30 GiB are free. "
            "B0 files are untouched."
        )

    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")

    cd_path = CACHE / "features.cd"
    cd_lines = ["0\tLabel", "1\tGroupId"]
    for column_index, feature in enumerate(FEATURES, start=2):
        kind = "Categ" if feature in H1["CAT_FEATURES"] else "Num"
        cd_lines.append(f"{column_index}\t{kind}\t{feature}")
    cd_content = "\n".join(cd_lines) + "\n"
    if cd_path.exists() and cd_path.read_text() != cd_content:
        raise ValueError("H1 feature schema has changed; refuse to reuse old cache")
    cd_path.write_text(cd_content)

    train_path, train_info = prepare_split("train", cd_path)
    val_path, val_info = prepare_split("val", cd_path)

    # Compare the finished H1 dataset against the validated B0 row counts.
    b0_cache = ROOT / "experiments/cache/B0_full"
    for name, info in [("train", train_info), ("val", val_info)]:
        b0_ckpt = b0_cache / f"{name}.checkpoint.json"
        if b0_ckpt.exists():
            expected = json.loads(b0_ckpt.read_text())
            assert info["rows"] == expected["rows"], (
                f"{name} row count differs from B0: "
                f"H1={info['rows']}, B0={expected['rows']}"
            )
            assert info["feeds"] == expected["feeds"], (
                f"{name} feed count differs from B0"
            )

    if os.environ.get("PREPARE_ONLY") == "1":
        print("\nH1 CACHE READY. GPU training not started.", flush=True)
        return

    print("\nLOADING H1 CATBOOST POOLS FROM DISK", flush=True)
    train_pool = Pool(str(train_path), column_description=str(cd_path), thread_count=8)
    val_pool = Pool(str(val_path), column_description=str(cd_path), thread_count=8)
    assert train_pool.num_col() == len(FEATURES) == 10
    assert val_pool.num_col() == len(FEATURES)
    assert train_pool.num_row() == train_info["rows"]
    assert val_pool.num_row() == val_info["rows"]

    print(
        f"TRAINING H1 ON 2 GPUs: "
        f"train={train_pool.num_row():,}, val={val_pool.num_row():,}, "
        f"features={len(FEATURES)}",
        flush=True,
    )
    model = CatBoostRanker(**H1["MODEL_PARAMS"])
    model.fit(train_pool, eval_set=val_pool, **H1["FIT_PARAMS"])

    print("\nCALCULATING H1 METRICS", flush=True)
    quality = model.eval_metrics(
        val_pool,
        ["NDCG", "NDCG:top=30", "NDCG:top=10"],
        eval_period=model.tree_count_,
        thread_count=H1["MODEL_PARAMS"]["thread_count"],
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
        "source": "full train / full val / H1 (B0 + hist_cat_count)",
    }
    (OUTPUT / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nH1 DONE:", metrics, flush=True)
    print("MODEL:", OUTPUT / "model.cbm", flush=True)


if __name__ == "__main__":
    main()

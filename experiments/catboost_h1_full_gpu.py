"""Avito RecSys 2026 — H1 feature pipeline and test inference.

H1 = official B0 features + hist_cat_count (history category frequency).

For the *full* training run, use train_h1_full_disk.py, NOT a whole-split
candidate_dataset(...).collect(), which exceeded 128 GiB RAM previously.
This module is also imported by that disk-based training script.

Official environment: Python 3.11 / polars 1.14 / catboost 1.2.10.
"""

import json
from collections import Counter
from functools import partial
from pathlib import Path

import click
import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool
from loguru import logger


MODEL_PARAMS = {
    "boosting_type": "Plain",
    "max_ctr_complexity": 1,
    "objective": "YetiRank",
    "task_type": "GPU",
    "eval_metric": "NDCG:top=30;type=Base",
    "early_stopping_rounds": 100,
    "devices": "0:1",
    "data_partition": "DocParallel",
    "gpu_ram_part": 0.95,
    "learning_rate": 0.4,
    "num_trees": 2000,
    "random_state": 666,
    "l2_leaf_reg": 5,
    "nan_mode": "Min",
    "allow_writing_files": False,
    "thread_count": 8,
}
FIT_PARAMS = {"verbose": 100, "use_best_model": True, "early_stopping_rounds": 50}
DATA_PARAMS = {"train_feeds": 0, "val_feeds": 0}

BIN_FEATURES = ["f1", "f2", "f3", "f4", "f5"]
CAT_FEATURES = ["category_id"]
FLOAT_FEATURES = [
    *BIN_FEATURES,
    "embedding_dot",
    "embedding_cosine",
    "embedding_missing",
    "hist_cat_count",
]
FEATURES = [*FLOAT_FEATURES, *CAT_FEATURES]
CANDIDATE_COLUMNS = [
    "item_id", *CAT_FEATURES, *BIN_FEATURES, "item_id__embedding_idx"
]
EMBEDDING_FEATURES = pl.Struct({
    "embedding_dot": pl.List(pl.Float32),
    "embedding_cosine": pl.List(pl.Float32),
    "embedding_missing": pl.List(pl.Float32),
})


def history_category_features(row: dict) -> list[int]:
    """One count per candidate; original history contains at most 256 events."""
    counts = Counter(row["history__category_id"] or [])
    return [int(counts.get(category, 0)) for category in row["category_id"]]


def embedding_features(batch: pl.Series, embeddings: np.memmap) -> pl.Series:
    """Copy of official B0 embedding features, calculated per complete feed."""
    features = {name: [] for name in EMBEDDING_FEATURES.to_schema()}
    for row in batch:
        user = np.asarray(row["user_embedding"], dtype=np.float32)
        indices = np.asarray(row["item_id__embedding_idx"])
        missing = indices == -1
        items = embeddings[np.maximum(indices, 0)].astype(np.float32)
        items[missing] = 0
        dot = np.einsum("ij,j->i", items, user)
        denominator = np.linalg.norm(items, axis=1) * np.linalg.norm(user)
        cosine = np.divide(
            dot, denominator, out=np.zeros_like(dot), where=denominator > 0
        )
        features["embedding_dot"].append(dot)
        features["embedding_cosine"].append(cosine)
        features["embedding_missing"].append(missing.astype(np.float32))
    return pl.DataFrame(
        features, schema=EMBEDDING_FEATURES.to_schema()
    ).to_struct()


def candidate_dataset(split: Path, max_feeds: int = 0) -> pl.LazyFrame:
    """Expand feed records to candidate records. Used for test prediction."""
    split = Path(split)
    files = sorted((split / "impressions").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No impression Parquet files in {split}")
    source = pl.scan_parquet([str(p) for p in files])
    columns = CANDIDATE_COLUMNS + (
        ["target"] if "target" in source.collect_schema() else []
    )
    source = source.select("feed_id", "user_embedding", "history__category_id", *columns)
    if max_feeds:
        source = source.head(max_feeds)

    source = source.with_columns(
        pl.struct("category_id", "history__category_id")
        .map_elements(
            history_category_features,
            return_dtype=pl.List(pl.Int64),
        )
        .alias("hist_cat_count")
    )

    meta = json.loads((split / "item_embeddings" / "data.meta").read_text())
    embeddings = np.memmap(
        split / "item_embeddings" / "data.memmap",
        mode="r",
        dtype=meta["dtype"],
        shape=tuple(meta["shape"]),
        order=meta["order"],
    )
    return (
        source.with_columns(
            pl.struct("user_embedding", "item_id__embedding_idx")
            .map_batches(
                partial(embedding_features, embeddings=embeddings),
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
            "hist_cat_count",
        ])
        .drop("user_embedding", "item_id__embedding_idx", "history__category_id")
    )


def predict_scores(batch: pl.DataFrame, model: CatBoostRanker) -> pl.DataFrame:
    features = Pool(batch.select(FEATURES), cat_features=CAT_FEATURES)
    scores = model.predict(features, thread_count=MODEL_PARAMS["thread_count"])
    return batch.select("feed_id", "item_id").with_columns(
        pl.Series("score", scores, dtype=pl.Float32)
    )


@click.group()
def main():
    """H1 prediction and pipeline checks. Train with train_h1_full_disk.py."""


@main.command("check")
@click.option("--data", type=click.Path(path_type=Path), required=True)
def check(data: Path):
    """Compute H1 for a *single* train feed; no model training."""
    sample = candidate_dataset(data / "train", max_feeds=1).collect(streaming=True)
    if sample.is_empty():
        raise RuntimeError("No candidates in first feed")
    assert sample["hist_cat_count"].null_count() == 0
    assert sample["hist_cat_count"].min() >= 0
    assert len(FEATURES) == 10
    logger.info("H1 CHECK OK: {} candidates, {} features", sample.height, len(FEATURES))
    print(sample.select("item_id", "category_id", "hist_cat_count", "target").head(10))


@main.command("predict")
@click.option("--data", type=click.Path(path_type=Path), required=True)
@click.option("--model", "model_path", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def predict(data: Path, model_path: Path, output: Path):
    """Score the complete test split and produce submission.parquet."""
    if output.exists():
        raise FileExistsError(output)
    model = CatBoostRanker()
    model.load_model(str(model_path))
    if model.feature_names_ != FEATURES:
        raise ValueError(
            f"Model features differ from H1 FEATURES: {model.feature_names_} != {FEATURES}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = candidate_dataset(data / "test")
    schema = candidates.select("feed_id", "item_id").collect_schema()
    schema["score"] = pl.Float32
    predictions = candidates.map_batches(
        partial(predict_scores, model=model),
        schema=schema,
        projection_pushdown=False,
        streamable=True,
    )
    predictions.sink_parquet(output, compression="zstd", maintain_order=True)
    logger.info("Predictions saved to {}", output)


if __name__ == "__main__":
    main()

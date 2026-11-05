# Avito Tech RecSys 2026 — reranking experiments

Эксперименты по [Avito Tech RecSys 2026](https://ods.ai/competitions/avitotechrecsys2026): официальные CatBoost и Transformer baselines, feature engineering для истории пользователя и проверка качества ранжирования.

**Задача:** retrieval уже сделан организаторами. Для каждой ленты (`feed_id`) нужно присвоить `score` каждому кандидату (`item_id`), чтобы хорошие объявления оказались выше. Оценивается порядок **внутри** ленты.

> **Состояние на 6 ноября 2026.** Три CatBoost-модели обучены и отправлены на ODS; лучший известный Public NDCG — **0.3461360866 (H1-full)**. Официальный Transformer пока прошёл только маленький CPU smoke test. Полный neural train и neural submission **не выполнены**. Результаты leaderboard приведены по отправкам команды, а не автоматически получены из GitHub.

## 1. Быстрый ориентир

| Что нужно | Где искать |
|---|---|
| Посмотреть данные | `official_baseline/01_data_exploration.ipynb`, `examples/` |
| Разобрать официальный CatBoost | `official_baseline/catboost_baseline.py` |
| Разобрать Transformer | `official_baseline/neural_baseline.py` |
| Посмотреть H1 и новые признаки | `experiments/catboost_h1_full_gpu.py` |
| Понять полное обучение B0/H1 | `experiments/train_b0_full_disk.py`, `experiments/train_h1_full_disk.py` |
| Проверить нейросеть на val | `experiments/eval_neural_ndcg.py` |
| Увидеть результаты | Раздел «Результаты» ниже; полные `run.json` хранятся на сервере |

## 2. Задача и данные

Одна строка `impressions/*.parquet` = **одна лента**, в которой лежат массивы кандидатов. У кандидатов есть `item_id`, `category_id`, пять числовых признаков `f1`–`f5`, `item_id__embedding_idx` и `target` (только train/val). Для ленты доступны `feed_id`, `user_id`, `timestamp`, `user_embedding` (128) и история `history__*` (до 256 событий, **от новых к старым**).

| Split | Даты 2026 | Impression-файлов | Особенности |
|---|---|---:|---|
| Train | 6–19 сентября | 512 | Есть `target` и `pos` |
| Val | 22–24 сентября | 64 | Есть `target`, кандидаты перемешаны перед показом |
| Test | 26–29 сентября | 64 | Нет `target`, кандидаты перемешаны |

Разделение **хронологическое**: исследовать temporal distribution shift очень важно. `pos` недоступна при inference и не должна напрямую использоваться как test-feature.

Item embeddings — `int8` в отдельных memmap для каждого split, размерность 128. Индекс `-1` означает отсутствие embedding; его надо маскировать, а не индексировать напрямую (в NumPy `-1` — последняя строка). Для dot/cosine переводить `int8` в `float32`. **Индексы не переносимы между train/val/test**.

Ожидаемое размещение полного датасета *локально*, вне Git:

```text
competition/
├── train/
│   ├── impressions/*.parquet
│   └── item_embeddings/{data.memmap,data.meta}
├── val/
│   ├── impressions/*.parquet
│   └── item_embeddings/{data.memmap,data.meta}
├── test/
│   ├── impressions/*.parquet
│   └── item_embeddings/{data.memmap,data.meta}
└── sample_submission.parquet
```

В `examples/` — первые **20 полных лент** каждого split, чтобы изучить schema/list-поля. Эти маленькие фрагменты **не репрезентативная EDA-выборка**; embedding-матрицы в них не включены. Возможность распространять конкурсные данные нужно проверить по правилам ODS.

## 3. Структура репозитория

Проверено по GitHub `main` 6 ноября 2026:

```text
.
├── official_baseline/
│   ├── 01_data_exploration.ipynb  # официальный обзор датасета
│   ├── 02_baseline.ipynb          # официальный notebook по baseline
│   ├── catboost_baseline.py      # официальный B0 / YetiRank
│   ├── neural_baseline.py        # официальный candidate-set Transformer
│   ├── calculate_metrics.py      # scorer при наличии answers + leaderboard_split
│   ├── pyproject.toml            # зависимости Python
│   └── uv.lock                   # зафиксированные версии
├── experiments/
│   ├── catboost_smoke.py         # B0 на маленьком срезе, CPU
│   ├── catboost_h1_smoke.py      # H1 на маленьком срезе, CPU
│   ├── catboost_b0_full_gpu.py   # фичи B0, test inference
│   ├── catboost_h1_full_gpu.py   # фичи H1, проверка, test inference
│   ├── train_b0_full_disk.py     # B0: подготовка TSV по shard + GPU training
│   ├── train_h1_full_disk.py     # H1: подготовка TSV по shard + GPU training
│   ├── train_b0_final.py         # B0 обученный на train+val
│   └── eval_neural_ndcg.py       # конкурсный NDCG для neural checkpoint
├── examples/
│   ├── train_20feeds.parquet
│   ├── val_20feeds.parquet
│   └── test_20feeds.parquet
└── .gitignore
```

**На GitHub нет:** полных данных, `experiments/cache/`, `experiments/runs/`, `experiments/logs/`, моделей (`.cbm`, `.pt`), исходных больших TSV и submission. Они сохранены на сервере; не перезаписывайте и не удаляйте их без необходимости.

**Отдельно:** `train_neural_streaming.py` обсуждался, но **не был в `main` на момент проверки**. Его нельзя считать готовой частью репозитория.

## 4. Модели и результаты

### B0 — CatBoostRanker

Официальный ранкер с YetiRank и **9 признаками**: `f1`–`f5`, `category_id`, `embedding_dot`, `embedding_cosine`, `embedding_missing` (сходство user/item embeddings и флаг missing).

Прямое разворачивание всех кандидатов в RAM приводило к OOM. Поэтому полный train готовили **по отдельным Parquet-шардам** в TSV на диске, сохраняя `GroupId = feed_id`, затем строили CatBoost Pool и обучали на GPU. Это **полный train, без negative sampling**.

### H1 — B0 + история категорий

**10-й признак**: `hist_cat_count`, количество событий из `history__category_id`, категория которых совпадает с категорией кандидата. Он не учитывает recency, тип события и сходство эмбеддингов истории. Полное обучение — `experiments/train_h1_full_disk.py`, фичи и inference — `experiments/catboost_h1_full_gpu.py`.

### Результаты полных CatBoost-запусков

| Model | Обучение | Full val NDCG | Public NDCG | Деревьев |
|---|---|---:|---:|---:|
| B0-full | Train | 0.35334025 | ~0.34607386 | 144 |
| B0-final | Train + val | — | ~0.34600771 | 144 |
| **H1-full** | Train | **0.35403538** | **0.3461360866** | 256 |

Полный train: **3 151 208 лент / 246 745 836 кандидатов**. Полный val: **14 423 ленты / 1 389 971 кандидат**. B0-final не имеет отдельного holdout val. Улучшение H1 над B0-full: **+0.00069513 full val**, около **+0.00006223 Public**; разница на Public очень мала и не доказывает улучшение на Private.

### Official Transformer — пока только smoke

`official_baseline/neural_baseline.py` — **candidate-set Transformer**, **не** SASRec/BERT4Rec и **не использует `history__*`**. Вход: user token + candidate tokens (числовые фичи, категория, item embedding), self-attention между кандидатами и user token. 2 encoder layers, hidden width 64, 4 heads. Loss — **BCEWithLogitsLoss**; лучшая эпоха выбирается по **val Logloss**, не NDCG. По умолчанию 5 эпох и batch size 32 ленты.

Проверенный CPU smoke: 200 train-лент, 100 val-лент / 8 129 кандидатов, 1 эпоха. **Val Logloss = 0.0944001663, конкурсный NDCG = 0.2830840747.** Это smoke-оценка и **не сравнима** с full-val CatBoost.

**Ключевой инженерный риск:** официальный `read_feeds(...).collect()` материализует весь split в CPU RAM перед DataLoader, несмотря на последующее обучение батчами. Для полного train необходим проверенный потоковый загрузчик или достаточная CPU RAM; большой VRAM H100 сам по себе проблему CPU OOM не решает.

## 5. Запуск и окружение

Зависимости зафиксированы в `official_baseline/pyproject.toml` и `uv.lock` (Python 3.11, CatBoost 1.2.10, Polars 1.14.0, NumPy 1.26.4, PyTorch 2.10.0). Репозиторий **не содержит полные данные**.

Для новой машины, если установлены Python 3.11 и `uv`:

```bash
git clone git@github.com:andrew52522/avito-recsys-2026.git
cd avito-recsys-2026/official_baseline
uv sync --locked
source .venv/bin/activate
python neural_baseline.py --help
```

На исходном сервере окружение уже есть, повторно создавать/скачивать его не нужно:

```bash
cd ~/avito
source official_baseline/.venv/bin/activate
```

**Безопасная проверка одного H1 feed (без тренировки):**

```bash
python experiments/catboost_h1_full_gpu.py check --data competition
```

**Проверка имеющегося neural checkpoint на маленьком val:**

```bash
python experiments/eval_neural_ndcg.py \
  --model experiments/runs/neural_smoke_cpu_200f/model.pt \
  --split competition/val \
  --max-feeds 100
```

**Test inference уже обученного H1, только если submission ещё нет:**

```bash
python experiments/catboost_h1_full_gpu.py predict \
  --data competition \
  --model experiments/runs/H1_full_gpu/model.cbm \
  --output experiments/runs/H1_full_gpu/submission.parquet
```

**Не запускайте full train из любопытства.** Скрипты полного B0/H1 используют абсолютные серверные пути (`/home/ubuntu/avito`) и конфигурацию **2 × RTX 4090**, `devices="0:1"`, `DocParallel`, `gpu_ram_part=0.95`. Перед запуском проверять `df -h`, `free -h`, `nvidia-smi`, текущие процессы, наличие кэшей и output. `train_b0_final.py` временно изменяет размер TSV-кэша, затем восстанавливает его: не запускать параллельно с задачами, использующими тот же файл.

## 6. Как считается метрика и что нужно сдать

Submission — Parquet с **ровно** `feed_id: int32`, `item_id: int32`, `score: float32` для **2 904 806** test-кандидатов. Нельзя терять пары, создавать дубли или NaN/Inf. Порядок строк в Parquet неважен.

**Конкурсный NDCG:**

- внутри каждого `feed_id` сортируем кандидатов по score;
- **DCG считаем по всей длине ленты** (не только top-30);
- нормируем на **IDCG@30**;
- затем усредняем NDCG **по лентам**, с равным весом;
- при равных score негативные кандидаты идут впереди позитивных; feed без positive получает 1.

**Важно:** конкурсный full NDCG **не равен** `NDCG:top=30` из CatBoost и **не равен** `val_logloss` у Transformer. `official_baseline/calculate_metrics.py` требует answers с `leaderboard_split` (`public`/`private`), поэтому для обычного val checkpoint удобнее использовать `experiments/eval_neural_ndcg.py`.

## 7. Следующие гипотезы и правила экспериментов

**Ещё не выполнено:** полный Transformer с потоковой загрузкой, ансамбль CatBoost + Transformer, EDA temporal shift, более богатые признаки истории. Возможные направления:

1. **EDA temporal shift:** категории, позиции, изменения состава лент и взаимодействий по времени; срезы раннего/позднего val.
2. **История:** cosine candidate vs history item embeddings (max/top-k/recent), event-aware признаки, recency и категории.
3. **LightGBM LambdaRank** для независимого ранкера.
4. **DIN / candidate-conditioned history attention**, потому что официальный Transformer не использует последовательность событий.
5. **Ансамбль:** получить одинаковые per-candidate val scores и смешать их после выравнивания по `(feed_id, item_id)`; веса подбирать только на val, а не многократными Public-сабмитами.

**Дисциплина:** каждому эксперименту нужны гипотеза, изменения относительно baseline, seed и гиперпараметры, объём данных, full-val NDCG, ресурсы/стоимость и вывод. Не путать smoke и full run. Не удалять кэши и сохранённые модели без обсуждения. Перед коммитом смотреть `git status` и `git diff --cached --stat`. Полные данные, модели, логи, ключи и секреты в Git не отправлять.

**Безопасность данных:** в текущем репозитории размещены фрагменты исходного конкурсного датасета. Проверьте правила ODS на их распространение и видимость GitHub-репозитория; если это запрещено, удаление из текущей ветки **не** стирает их из истории Git.

## Полезные ссылки

- [Соревнование](https://ods.ai/competitions/avitotechrecsys2026)
- [Данные и baseline](https://ods.ai/competitions/avitotechrecsys2026/dataset)
- [Leaderboard](https://ods.ai/competitions/avitotechrecsys2026/leaderboard)

---

_Структура файлов проверена по ветке `main` 6 ноября 2026; метрики и детали состоявшихся запусков — по журналу экспериментов команды. Модели, кэши и полный датасет в GitHub не хранятся._

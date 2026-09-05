# Fraud Spike Detector — Code Flow

This document describes the current execution flow of the project, the data passed between components, and known placeholders. It supersedes earlier versions of this file, which described an earlier state of the pipeline (single fraud pattern, no IP/BIN features, Random Forest only, and an evaluation bug where the test pipeline retrained on test data).

## 1. High-level flow

```text
PaySim CSV
    |
    v
simulate.build_buckets
    |  inject 5 fraud patterns (A-E) and genuine demand spikes
    |  aggregate transactions by step
    |  compute IP/BIN concentration features
    v
data/processed/buckets_labeled.csv
data/processed/raw_augmented.csv   (row-level, with ip_address / bin_number)
    |
    v
train.train
    |  train XGBoost classifier on training split only
    |  save models/stage2_model.pkl
    v
detect.pipeline
    |  Stage 1: z-score / EWMA / CUSUM anomaly detection
    |  Stage 2: load saved model, classify every bucket
    |  final_flag = 1 iff stage2_pred == "fraud_attack"  (NOT gated by Stage 1)
    |  attach reason_codes
    v
evaluate.run_eval
    |  time-split, load the ALREADY-TRAINED model, predict on test only
    |  fraud-catch report (fraud_attack only)
    |  three-way classification report (all classes)
    |  cluster action report (IP/BIN recommendations)
    |  charts: stage1 anomalies, stage2 classification, fraud clusters,
    |          class metrics, confusion matrix
    v
evaluation report + charts
```

Quick-start commands:

```bash
python -m simulate.build_buckets
python -m train.train
python -m evaluate.run_eval
python -m detect.pipeline
```

**Important fix from earlier versions:** evaluation now trains the model exactly once (via `train.train`), saves it to disk, and `run_eval.py` loads that saved model to predict on the held-out test set only. The earlier version of this pipeline retrained a fresh model inside the test-evaluation call itself, which meant "test" metrics were partly measuring training performance. This is corrected — see `evaluate/run_eval.py`.

## 2. Simulation and bucket creation

### `simulate/config.py`

`InjectionConfig` defaults:

- Random seed: `42`
- Extra transactions per fraud step: `150`
- Extra transactions per demand step: `150`
- Fraud steps: `80` (raised from an initial `10` — too few examples for the classifier to learn from)
- Demand steps: `80` (same reasoning)
- Fraud IP pool size: `n_fraud_ips`
- Fraud BIN pool size: `n_fraud_bins`

### `simulate/build_buckets.py`

`build_labeled_buckets()`:

1. Loads the PaySim CSV from `data/raw/` (via `load_raw_data()`).
2. Sorts unique `step` values and splits them ~80/20 chronologically into train/test step sets.
3. Selects fraud and demand steps separately within train and test portions (via a seeded RNG), so both classes are represented in both splits.
4. Labels each selected step `fraud_attack` or `genuine_demand`; everything else is `normal`.
5. Calls `apply_injections()` to add synthetic rows for the selected steps.
6. Calls `create_bucket_features()` to aggregate the augmented data by `step`.
7. Saves the labeled bucket table to `data/processed/buckets_labeled.csv`, and the row-level augmented data (with `ip_address` / `bin_number`) to `data/processed/raw_augmented.csv`.

Bucket-level columns:

| Column | Meaning |
| --- | --- |
| `step` | PaySim time-step identifier |
| `txn_count` | Number of transactions in the bucket |
| `unique_senders` | Number of distinct `nameOrig` values |
| `avg_amount` | Mean transaction amount |
| `total_amount` | Sum of transaction amounts |
| `fraud_count` | Sum of `isFraud` values (PaySim's own native fraud, distinct from our injected labels) |
| `payment_count` | Number of `PAYMENT` transactions |
| `transfer_count` | Number of `TRANSFER` transactions |
| `unique_ratio` | `unique_senders / txn_count` |
| `amount_ratio` | `total_amount / txn_count` |
| `fraud_rate` | `fraud_count / txn_count` |
| `top_ip_share` | Share of transactions from the single most common IP in the bucket |
| `top_bin_share` | Share of transactions using the single most common card BIN in the bucket |
| `label` | Synthetic target: `normal`, `genuine_demand`, or `fraud_attack` |

**Note:** `top_ip_share` / `top_bin_share` are computed and saved, and used by the cluster action report, but are *not* included in the Stage 2 classifier's training features (see Section 4) — this was a deliberate choice after testing showed including them made the classifier's IP/BIN reliance oversized. They remain valuable for downstream, human-facing investigation output.

### `simulate/inject_spikes.py`

Five fraud patterns are simulated, chosen at random per fraud step:

| Type | Function | Pattern |
| --- | --- | --- |
| A | `inject_fraud_type_a` | Many senders → one destination (card-testing bot farm) |
| B | `inject_fraud_type_b` | Few senders → many destinations (mule network) |
| C | `inject_fraud_type_c` | Many small transactions (probe amounts) |
| D | `inject_fraud_type_d` | Sudden high-value transfers (account drain), fewer but larger rows |
| E | `inject_fraud_type_e` | Rapid repeated transactions from one session |

Each fraud type draws its IP/BIN from a small "core" pool `purity`% of the time (default 75%), and a fully random IP/BIN otherwise — this avoids a perfectly clean, unrealistic IP/BIN signal.

`inject_demand_spike()` creates `PAYMENT` rows from a mix of brand-new customer IDs and a smaller pool of "repeat customers" (~30% of rows), drawing IP/BIN from a shared pool of ~40 IPs / 15 BINs — this represents realistic overlap (shared ISPs, popular banks) rather than perfectly unique values, so `unique_ratio` and IP/BIN concentration aren't artificially perfect separators.

`apply_injections()` concatenates all generated rows onto the original PaySim DataFrame; it does not modify original rows.

## 3. Stage 1: anomaly detection

### `detect/stage1_anomaly.py`

`stage1_flag(bucket)` monitors `txn_count` by default:

1. `compute_baseline()` — global mean/std of the metric.
2. `zscore` — bucket's distance from the global baseline.
3. `ewma_zscore()` — exponentially weighted local z-score (`alpha=0.2`).
4. `cusum_signal()` — **now run on the z-scored series, not raw transaction counts.** This was a bug fix: running CUSUM on raw counts meant the cumulative sum grew by hundreds per step and blew past the `3.0` threshold almost immediately, flagging nearly all traffic. Running it on the standardized z-score puts the CUSUM threshold on the same "standard deviations from normal" scale as the other two signals.
5. `anomaly_flag = 1` when any of the three signals exceeds its threshold (`|zscore| > 2.5`, `|ewma_z| > 2.5`, or `cusum > 3.0`).

After the CUSUM fix, Stage 1 flags only the most statistically extreme volume spikes — a small minority of buckets — rather than nearly everything.

## 4. Stage 2: classification

### `detect/stage2_classifier.py`

`prepare_features()` selects these features for the classifier:

```text
txn_count, unique_senders, avg_amount, total_amount,
unique_ratio, amount_ratio
```

(`top_ip_share` / `top_bin_share` were tested as additional features; they created a near-perfect, unrealistically clean separation and were excluded from the classifier for that reason — see Section 2.)

Labels are mapped `normal -> 0`, `genuine_demand -> 1`, `fraud_attack -> 2`.

`train_classifier()`:

1. Trains on the full DataFrame passed to it (the caller, `train/train.py`, is responsible for passing only the training split).
2. Supports `model_type`: `"random_forest"` (default), `"logistic"`, `"gradient_boosting"`, and `"xgboost"`. **XGBoost was selected as the final model** after comparing all three tree-based options on identical data — it had the best balance of fraud recall, false-positive rate, and weighted cost.
3. Saves the trained model to `models/stage2_model.pkl`.

`predict_stage2()` loads the saved model if none is supplied, predicts on the given buckets, and adds a `stage2_pred` column with the three label names.

### `train/train.py`

Entry point that builds buckets, time-splits them, and calls `train_classifier()` on the **training split only** — this is what produces the saved model used by both `detect.pipeline` and `evaluate.run_eval`.

## 5. Combined detection pipeline

### `detect/pipeline.py`

`run_detection_pipeline(bucket, model=None)`:

1. Calls `stage1_flag()` — adds `anomaly_flag`.
2. Loads the saved model if none was passed in.
3. Calls `predict_stage2()` — adds `stage2_pred`, run on every bucket regardless of Stage 1's result.
4. Sets `final_flag = 1` iff `stage2_pred == "fraud_attack"`. **This is no longer gated by `anomaly_flag`** — an earlier version required both stages to agree, which caused most real fraud (fraud that doesn't show up as a raw volume anomaly) to be silently dropped from the final decision.
5. Sets `genuine_demand_flag = 1` iff `stage2_pred == "genuine_demand"`, independent of `final_flag`.
6. Adds `reason_codes` — a dict with the Stage 1 flag, Stage 2 prediction, and ground-truth label (when available), for explainability.

Returns the full DataFrame with all added columns.

## 6. Evaluation flow

### `evaluate/split.py`

`time_split()` sorts unique steps and treats the last `test_fraction` of them (by step number, not row count) as the test set — strictly chronological, no shuffling, no leakage between splits.

### `evaluate/run_eval.py`

`run_evaluation()`:

1. Rebuilds labeled buckets and time-splits them.
2. Loads the **already-trained** model from disk (`models/stage2_model.pkl`) — does not retrain here.
3. Runs `run_detection_pipeline()` on the test split only, using the loaded model.
4. Loads the row-level `raw_augmented.csv` and runs `generate_cluster_action_report()` for IP/BIN-based recommendations.
5. Computes and prints **two separate reports**, replacing an earlier single combined report that conflated fraud and demand into one "positive" class:
   - **Fraud catch performance**: `fraud_catch_metrics()` treats only `fraud_attack` as positive; `genuine_demand` and `normal` are both negative. This isolates fraud-catching ability without demand spikes distorting the score.
   - **Three-way classification**: `sklearn.metrics.classification_report` comparing `stage2_pred` directly against `label` for all three classes.
6. Generates charts via `evaluate/plots.py`: Stage 1 anomalies, Stage 2 classification, fraud clusters, class metrics bar chart, and confusion matrix.

### `evaluate/metrics.py`

- `confusion_counts()`, `precision_recall_f1()`, `compute_cost()` — generic binary metric helpers (false-positive cost `1.0`, false-negative cost `5.0`, reflecting that missing real fraud is weighted as more costly than a false alarm).
- `fraud_catch_metrics()` — fraud-only binary metrics (see above).
- `evaluate_predictions()` — retained from the original design but no longer used as the headline report, since it treats fraud and demand as the same "positive" class.

### `evaluate/cluster_report.py`

`generate_cluster_action_report()`:

1. Calls `find_fraud_clusters()` to group fraud steps that occur within `max_gap` steps of each other into a single cluster.
2. For each cluster, looks up the row-level IP addresses and BIN numbers involved (from `raw_augmented.csv`).
3. If one IP or BIN accounts for more than 50% of a cluster's fraud transactions, recommends reporting it (IP → cyber-crime unit, BIN → issuing bank).
4. Otherwise, flags the cluster as a likely distributed attack.

## 7. API flow (unchanged, still placeholder)

### `api/main.py`

Exposes only:

```text
GET /health -> {"status": "ok"}
```

Route modules exist but are not mounted by the application as written.

### `api/schemas.py` and `api/routes/score.py`

`ScoreRequest` / `ScoreResponse` schemas are defined, but the `/score` handler is a placeholder — it does not call `detect.pipeline` or load `stage2_model.pkl`. Wiring this up (loading the trained model and running `run_detection_pipeline` on incoming requests) is the natural next step if the API is to become functional.

### `api/routes/events.py` and `api/routes/review.py`

Still placeholders: `GET /events` returns an empty list; `POST /review` returns a fixed acceptance response without storing anything.

## 8. Explanation flow (unchanged)

### `explain/fallback.py`

`explain_fallback()` builds a deterministic, plain-English explanation from a bucket's stats and Stage 2 prediction — no LLM call, always available.

### `explain/prompt_template.py`

`PROMPT_TEMPLATE` is defined for an external LLM (Ollama, per project notes) but is not currently invoked by the batch pipeline or API. Kept out of the detection path deliberately, to preserve deterministic precision/recall — the LLM is intended purely as an explanation layer, not a decision-maker.

## 9. Storage and other project files

- `data/raw/` — source PaySim CSV.
- `data/processed/buckets_labeled.csv` — bucket-level features and labels.
- `data/processed/raw_augmented.csv` — row-level data including `ip_address` / `bin_number`, used by the cluster action report.
- `models/stage2_model.pkl` — latest trained Stage 2 model (XGBoost).
- `db/schema.sql` — database schema definitions; current Python code does not connect to or write to this database.
- `docker/` — container configuration for deployment.

## 10. End-to-end example

```bash
pip install -r requirements.txt

python -m simulate.build_buckets   # inject fraud types A-E + demand, build buckets
python -m train.train              # train XGBoost on the training split, save model
python -m evaluate.run_eval        # load saved model, evaluate on test split,
                                    # print fraud-catch + 3-way reports,
                                    # print cluster action report, save charts
python -m detect.pipeline          # run full pipeline end-to-end, print sample output
```

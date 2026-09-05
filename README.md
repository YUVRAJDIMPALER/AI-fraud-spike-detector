# AI Fraud Spike Detector

AI Fraud Spike Detector is a small end-to-end fraud detection project built around a practical question:

> Is this sudden burst of activity a real business spike, or an attack hiding inside normal traffic?

It uses PaySim transaction data, adds realistic synthetic fraud and demand events, classifies each time bucket, and can publish the resulting fraud events to Kafka for downstream teams.

## What it does

The detector uses two complementary stages:

1. **Anomaly detection** looks for unusual transaction volume using z-scores, EWMA, and CUSUM.
2. **Classification** labels every bucket as `normal`, `genuine_demand`, or `fraud_attack` using transaction count, sender diversity, and amount patterns.

The final fraud decision comes from Stage 2. This is intentional: an attack can have an ordinary-looking volume while still having a suspicious transaction shape. Stage 1 remains available as useful context for investigation.

```text
PaySim data
   -> inject realistic fraud and demand examples
   -> aggregate transactions into time buckets
   -> detect unusual volume
   -> classify normal traffic, demand, or fraud
   -> predict fraud type and publish events
```

## Why synthetic data is included

Public datasets rarely contain enough labeled examples of both fraud attacks and legitimate demand spikes. The project starts with the PaySim CSV and injects labeled examples for five attack patterns:

| Type | Pattern | Example interpretation |
| --- | --- | --- |
| A | Many senders to one destination | Card-testing bot farm |
| B | Few senders to many destinations | Mule or account-takeover network |
| C | Many small transactions | Card-testing probes |
| D | Sudden high-value transfers | Account drain |
| E | Rapid repeated transactions | Compromised session |

It also creates `genuine_demand` periods with many customers, shared IPs, and repeat activity. That makes the hard case visible: a busy period should not automatically be treated as fraud.

## Results

On the current time-based holdout evaluation, the detector reports:

| Fraud-only metric | Score |
| --- | ---: |
| Precision | 0.84 |
| Recall | 0.90 |
| F1 | 0.87 |

Three-way accuracy across `normal`, `genuine_demand`, and `fraud_attack` is **0.93**. These numbers come from synthetic events added to PaySim data, so they are useful for comparing experiments, not a substitute for production validation.

### Why is the fraud-type classifier about 80% accurate?

The fraud-type model answers a harder question than the main detector: it must distinguish **which kind** of fraud happened after the bucket has already been identified as fraud. Its current evaluation is **80% accuracy on 40 held-out fraud buckets**, which means 32 predictions were correct and 8 were wrong.

The errors are understandable from the confusion matrix:

- Types **D** and **E** are separated well in this run because sudden high-value transfers and rapid repeated activity create strong signals.
- Types **A**, **B**, and **C** overlap more. For example, some A events are predicted as C, while B and C are occasionally confused with each other.
- The classifier only sees aggregate bucket features such as transaction count, sender count, average amount, amount totals, IP concentration, and BIN concentration. Different attack patterns can produce similar aggregates, so the model cannot always recover the exact story behind a bucket.
- The evaluation set is small: one different set of injected steps could move the percentage noticeably. The fraud patterns are also simulated, and their parameters include randomness and intentional noise to avoid making them unrealistically easy to recognize.

This 80% score does **not** mean the complete fraud detector is only 80% effective. The main Stage 2 model first separates `fraud_attack` from `genuine_demand` and `normal`; the fraud-type prediction is a second, more detailed classification used to guide investigation and Kafka routing. Improving it would require more varied labeled examples, richer row-level or sequence features, and evaluation on real production fraud data.

## Quick start

The commands below are written for Windows PowerShell. Run them from the project folder: `C:\code\Fraud-Spike detector`.

### 1. Install dependencies

Run these commands from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install fastapi uvicorn kafka-python xgboost
```

### 2. Download the PaySim dataset

The project uses the PaySim dataset from Kaggle:

[https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)

You need a Kaggle account and Kaggle API credentials for the command-line method. Install the Kaggle CLI, authenticate it using the instructions in Kaggle's account settings, then run:

```powershell
python -m pip install kaggle
kaggle datasets download -d ealaxi/paysim1 -p data/raw
Expand-Archive -Path data/raw/paysim1.zip -DestinationPath data/raw -Force
Remove-Item data/raw/paysim1.zip
```

The pipeline expects this CSV file:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

If Kaggle downloads a different filename, rename the CSV to the filename above. Do not commit the dataset to GitHub; it is intentionally excluded by `.gitignore`.

### 3. Build data and train models

For the complete workflow used by the dashboard, run:

```powershell
python train_all_models.py
python -m train.train_fraud_type
```

What these commands do:

- `python train_all_models.py` creates labeled buckets, trains Random Forest, Gradient Boosting, and XGBoost models, evaluates them on a chronological holdout set, and writes the model comparison used by the dashboard.
- `python -m train.train_fraud_type` trains the second model that predicts fraud types A-E.

For the smaller single-model workflow, use:

```powershell
python -m simulate.build_buckets
python -m train.train
```

`simulate.build_buckets` adds synthetic fraud and genuine-demand spikes to PaySim and writes generated files under `data/processed/`. `train.train` trains the basic Stage 2 classifier.

### 4. Evaluate the models

```powershell
python -m evaluate.run_eval
```

This rebuilds the labeled buckets, loads the trained model, evaluates only the chronological test split, prints fraud and three-class metrics, and creates evaluation charts.

### 5. Run a batch detection

```powershell
python -m detect.pipeline
```

This loads the saved model, runs both detection stages, predicts fraud types for fraud buckets, and prints the latest detection results.

`train.train` uses Gradient Boosting and trains only on the chronological training split. Evaluation then runs against the later holdout split, helping avoid time leakage.

## Run the API and dashboard

Start the API with:

```powershell
uvicorn api.main:app --reload
```

Then open [http://localhost:8000/](http://localhost:8000/) in a browser. Useful endpoints include:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Check that the API is running |
| `GET /api/models` | List available models and the active model |
| `GET /api/models/{model_id}/evaluation` | Read evaluation details for a model |
| `POST /api/models/{model_id}/select` | Select the model used by the pipeline |
| `POST /api/pipeline/run` | Build data, run detection, and publish fraud events |
| `GET /docs` | Browse the interactive FastAPI documentation |

The dashboard can stream Kafka events over `WS /ws/events` when Kafka is running.

To use the model picker and the dashboard endpoints, make sure `train_all_models.py` has been run first. Restart the API after retraining so it reloads the model files and metrics.

## Kafka and department messages

Kafka, Zookeeper, PostgreSQL, and the API can be started together with Docker Compose:

```powershell
docker compose -f docker/docker-compose.yml up --build
```

Kafka is expected at `localhost:9092`. After the models are trained, publish detected fraud events with:

```powershell
python run_and_publish.py
```

Events are sent to topics such as `fraud.typeA.problem`. A department consumer can acknowledge and act on one fraud type:

```powershell
python -m events.department_consumer A
```

Use `B`, `C`, `D`, or `E` for the other fraud types.

## Project layout

```text
simulate/       Create labeled buckets and inject fraud/demand patterns
detect/         Run anomaly detection, classification, and fraud-type prediction
train/          Train the saved models
evaluate/       Time split, metrics, charts, and cluster reports
api/             FastAPI endpoints and the dashboard event bridge
events/          Kafka publishing and department consumers
dashboard/      Browser dashboard
data/raw/       Original PaySim input
data/processed/ Generated bucket and augmented transaction data
models/         Generated model and metric artifacts
docker/         Docker Compose and API image configuration
```

## Generated files

The main commands create or update:

- `data/processed/buckets_labeled.csv`: bucket-level features and labels.
- `data/processed/raw_augmented.csv`: augmented transaction rows with IP and BIN data.
- `models/stage2_model.pkl`: three-way Stage 2 classifier.
- `models/fraud_type_model.pkl`: classifier for fraud types A-E.
- Evaluation charts generated by `evaluate.run_eval`.

These files are local working artifacts and are excluded from GitHub where appropriate. The raw PaySim CSV and processed CSV files are never required in the repository because another developer can download and regenerate them by following the steps above.

## Cluster action reports

For detected fraud, the evaluation code can group nearby events into attack clusters and inspect IP and card BIN concentration. A concentrated IP or BIN produces a more actionable recommendation; a cluster without a dominant source is treated as a possible distributed attack. These signals are kept out of the classifier so the model does not overfit to artifacts from the synthetic data.

## Important limitations

- The labels are simulated on top of PaySim data. Real production traffic may behave differently.
- The current benchmark uses a relatively small time-based holdout, so results can change with new simulation settings.
- Stage 1 uses global thresholds rather than per-merchant baselines.
- Kafka must be available at `localhost:9092` for publishing and live dashboard events.
- PostgreSQL configuration and schema are included, but the Python application does not currently persist events there.
- The explanation path is deterministic; the LLM prompt files are not part of the active detection path.

## Push this project to GitHub

The GitHub repository should be named `AI-fraud-spike-detector`.

After creating an empty repository with that name on GitHub, run these commands from the project root. Replace `YOUR-USERNAME` with your GitHub username:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial commit: AI fraud spike detector"
git remote add origin https://github.com/YOUR-USERNAME/AI-fraud-spike-detector.git
git push -u origin main
```

What each command does:

- `git init` creates local Git history for the project.
- `git branch -M main` names the default branch `main`.
- `git add .` stages files that are not excluded by `.gitignore`.
- `git status` lets you check that private files, virtual environments, models, and datasets are not staged.
- `git commit` saves the first version locally.
- `git remote add origin` connects this folder to your GitHub repository.
- `git push -u origin main` uploads the project and remembers the remote branch.

Before committing, confirm that `git status` does **not** show `.venv`, `data/raw`, `data/processed`, `.env`, or `.pkl` files.

## License and data

This repository is intended for experimentation and demonstration. Make sure you have the right to use any PaySim data distributed with your local copy before sharing or deploying the project.

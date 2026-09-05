# Fraud Spike Detector Runbook

This runbook explains how to install, train, evaluate, and run the Fraud Spike Detector on a new computer.

The commands below assume Windows PowerShell because that is the primary development environment for this project. macOS/Linux equivalents are included where virtual-environment commands differ.

## 1. What You Need

### Required software

Install these before starting:

- Git
- Python 3.11 (recommended; the Docker image also uses Python 3.11)
- Docker Desktop, only if you want Kafka, PostgreSQL, or the containerized API
- A Kaggle account, to download the PaySim dataset

Check the installations:

```powershell
python --version
git --version
docker --version
docker compose version
```

Python 3.10 or newer should work, but Python 3.11 is the tested target. On some Windows machines, use `py --version` and replace `python` with `py` in the commands below if the `python` command is not available.

## 2. Get the Project

Clone the repository and enter its directory:

```powershell
git clone <REPOSITORY_URL>
Set-Location "Fraud-Spike detector"
```

If the project is already on the computer:

```powershell
Set-Location "C:\path\to\Fraud-Spike detector"
```

All commands in this document must be run from the project root, the directory containing `requirements.txt`, `train_all_models.py`, and the `api` folder.

## 3. Create and Activate a Virtual Environment

A virtual environment keeps this project's packages separate from other Python projects.

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

The terminal prompt should show `(.venv)` after activation.

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools after activation:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

For macOS/Linux, use the same command with the activated environment.

## 4. Install Python Dependencies

Install the packages listed by the repository:

```powershell
python -m pip install -r requirements.txt
```

The current `requirements.txt` contains the data-science and notebook packages. The application and publishing paths also require these runtime packages, so install them as well:

```powershell
python -m pip install fastapi uvicorn kafka-python xgboost
```

Install the Kaggle CLI for downloading PaySim:

```powershell
python -m pip install kaggle
```

Verify the important imports:

```powershell
python -c "import pandas, sklearn, joblib, matplotlib, seaborn, fastapi, uvicorn, kafka, xgboost; print('Python dependencies are installed')"
```

If `xgboost` fails to install, confirm that the virtual environment is active and that Python is 64-bit:

```powershell
python -c "import platform; print(platform.python_version()); print(platform.architecture()[0])"
```

## 5. Download the PaySim Dataset

The project uses the PaySim dataset from Kaggle. Download it from:

<https://www.kaggle.com/datasets/ealaxi/paysim1>

The expected input file is:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

### Option A: Kaggle CLI

1. Create or locate a Kaggle API token from your Kaggle account settings.
2. Configure the Kaggle CLI using Kaggle's current authentication instructions.
3. From the repository root, run:

```powershell
New-Item -ItemType Directory -Force data/raw
kaggle datasets download -d ealaxi/paysim1 -p data/raw
Expand-Archive -Path data/raw/paysim1.zip -DestinationPath data/raw -Force
Remove-Item data/raw/paysim1.zip
```

Check that the expected file exists:

```powershell
Test-Path "data/raw/PS_20174392719_1491204439457_log.csv"
```

The command should return `True`.

### Option B: Manual download

Download the dataset ZIP from Kaggle, extract it into `data/raw`, and rename the CSV if necessary so its final path is exactly:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

Do not commit the raw dataset or generated data to the repository.

## 6. Train the Models

There are two supported training workflows.

### Recommended: train all model candidates

This creates the model files used by the model picker and dashboard:

```powershell
python train_all_models.py
python -m train.train_fraud_type
```

The first command:

- Builds synthetic fraud and genuine-demand examples from PaySim.
- Creates time-based training and holdout splits.
- Trains Random Forest, Gradient Boosting, and XGBoost models.
- Writes the model files under `models/`.
- Writes `models/model_metrics.json`.

The second command trains the fraud-type classifier for patterns A through E and writes its model and metrics.

Expected important outputs include:

```text
models/stage2_random_forest.pkl
models/stage2_gradient_boosting.pkl
models/stage2_xgboost.pkl
models/model_metrics.json
models/fraud_type_model.pkl
models/fraud_type_metrics.json
```

### Smaller single-model workflow

Use this when you only need the basic pipeline and do not need the dashboard model picker:

```powershell
python -m simulate.build_buckets
python -m train.train
```

This writes the basic Stage 2 model to:

```text
models/stage2_model.pkl
```

## 7. Evaluate the Models

Run the chronological holdout evaluation:

```powershell
python -m evaluate.run_eval
```

This rebuilds the labeled buckets, loads the already-trained model, evaluates the later time-based holdout, prints fraud-catch and three-class metrics, and generates evaluation charts.

To check the fraud-type classifier separately:

```powershell
python check/check_fraud_type_accuracy.py
```

Useful validation commands are:

```powershell
python check/check_fraud.py
python check/check_fraud_type.py
```

## 8. Run Batch Detection Without Kafka

After training the single model or the model artifacts needed by the pipeline, run:

```powershell
python -m detect.pipeline
```

This runs Stage 1 anomaly detection and Stage 2 classification, then prints detection results.

To create and publish fraud events, Kafka must be running first. The publishing script is:

```powershell
python run_and_publish.py
```

## 9. Run the API and Dashboard Locally

The API imports Kafka components during startup, so Kafka should be running even if you only want to open the dashboard.

Start the API from an activated virtual environment:

```powershell
uvicorn api.main:app --reload
```

Open these URLs in a browser:

- Dashboard: <http://localhost:8000/>
- Health check: <http://localhost:8000/health>
- Interactive API documentation: <http://localhost:8000/docs>
- Model list: <http://localhost:8000/api/models>

The API loads model metadata at startup. After retraining, stop it with `Ctrl+C` and start it again:

```powershell
uvicorn api.main:app --reload
```

The dashboard pipeline endpoint runs detection and publishes detected fraud events:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/pipeline/run
```

A successful response includes the selected model, bucket counts, fraud counts, and the number of published events.

## 10. Start Kafka and PostgreSQL with Docker Compose

Docker Compose starts these services:

- API on port `8000`
- PostgreSQL on port `5432`
- Zookeeper on port `2181`
- Kafka on port `9092`

Start everything from the repository root:

```powershell
docker compose -f docker/docker-compose.yml up --build
```

Leave this terminal running. In a second terminal, activate the virtual environment before running local Python commands:

```powershell
Set-Location "C:\path\to\Fraud-Spike detector"
.\.venv\Scripts\Activate.ps1
```

Check running containers:

```powershell
docker compose -f docker/docker-compose.yml ps
```

Stop the services:

```powershell
docker compose -f docker/docker-compose.yml down
```

Stop the services and remove their persisted volumes, if any are later added:

```powershell
docker compose -f docker/docker-compose.yml down -v
```

Kafka is configured for local clients at:

```text
localhost:9092
```

The Python Kafka producer, dashboard bridge, and department consumer all use this address.

## 11. Publish and Consume Department Events

With Kafka running and the models trained, publish detected fraud events:

```powershell
python run_and_publish.py
```

Consume events for a fraud type in another terminal:

```powershell
python -m events.department_consumer A
```

Valid fraud types are `A`, `B`, `C`, `D`, and `E`:

```powershell
python -m events.department_consumer B
python -m events.department_consumer C
python -m events.department_consumer D
python -m events.department_consumer E
```

Events are published to topics such as:

```text
fraud.typeA.problem
fraud.typeB.problem
fraud.typeC.problem
fraud.typeD.problem
fraud.typeE.problem
```

## 12. Optional Notebook Workflow

The repository includes `report.ipynb`. With the virtual environment activated, start Jupyter with:

```powershell
jupyter notebook
```

Alternatively:

```powershell
jupyter lab
```

Open `report.ipynb` and select the `.venv` Python interpreter as the notebook kernel.

## 13. Complete First-Time Command Sequence

For a normal Windows setup, the complete sequence is:

```powershell
Set-Location "C:\path\to\Fraud-Spike detector"
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install fastapi uvicorn kafka-python xgboost kaggle
kaggle datasets download -d ealaxi/paysim1 -p data/raw
Expand-Archive -Path data/raw/paysim1.zip -DestinationPath data/raw -Force
Remove-Item data/raw/paysim1.zip
python train_all_models.py
python -m train.train_fraud_type
python -m evaluate.run_eval
```

After this sequence, use the API, dashboard, or Kafka instructions above.

## 14. Troubleshooting

### `python` is not recognized

Install Python 3.11 and enable **Add Python to PATH** during installation. Alternatively, use the Python launcher:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.11 -m pip install -r requirements.txt
```

### PowerShell blocks `Activate.ps1`

Allow scripts only for the current PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### `ModuleNotFoundError`

Confirm that the environment is active and install the missing runtime packages:

```powershell
python -m pip install -r requirements.txt
python -m pip install fastapi uvicorn kafka-python xgboost
```

Check which Python is being used:

```powershell
Get-Command python
python -c "import sys; print(sys.executable)"
```

The executable should be inside the project's `.venv` directory.

### PaySim file not found

Verify the exact path and filename:

```powershell
Get-ChildItem data/raw
Test-Path "data/raw/PS_20174392719_1491204439457_log.csv"
```

Download or rename the CSV so the expected path exists.

### No model selected in the API

Run the recommended training workflow:

```powershell
python train_all_models.py
python -m train.train_fraud_type
```

Then restart Uvicorn. The API reads `models/model_metrics.json` when it starts.

### Kafka connection refused

Make sure Docker Compose is running and that port `9092` is available:

```powershell
docker compose -f docker/docker-compose.yml ps
Test-NetConnection localhost -Port 9092
```

Start or restart the services if necessary:

```powershell
docker compose -f docker/docker-compose.yml up --build
```

### Port `8000`, `5432`, or `9092` is already in use

Find the process using a port:

```powershell
Get-NetTCPConnection -LocalPort 8000,5432,9092 -ErrorAction SilentlyContinue
```

Stop the conflicting application or change the relevant Docker port mapping. If the API alone is affected, run it on another port:

```powershell
uvicorn api.main:app --reload --port 8001
```

Then open <http://localhost:8001/>.

### Docker API build cannot import FastAPI or Kafka

The Dockerfile installs the packages from `requirements.txt`, while the API also needs the runtime packages listed in Section 4. Install those packages in `requirements.txt` before building the API image, or build/run the API from the activated local virtual environment. The local non-Docker workflow is the most direct path for training and development.

### Training is slow or uses too much memory

The PaySim CSV is large and the all-model workflow trains three candidates. Use the single-model workflow for a quicker run:

```powershell
python -m simulate.build_buckets
python -m train.train
```

## 15. Stop and Clean Up

Deactivate the Python environment:

```powershell
deactivate
```

Remove the virtual environment if you need to recreate it:

```powershell
Remove-Item -Recurse -Force .venv
```

Generated data and models can be regenerated from the raw PaySim input. Remove only generated artifacts when you intentionally want a clean rebuild:

```powershell
Remove-Item -Force data/processed/*.csv -ErrorAction SilentlyContinue
Remove-Item -Force models/*.pkl, models/*.json -ErrorAction SilentlyContinue
```

Do not remove the raw PaySim CSV unless you are prepared to download it again.

## 16. Important Notes

- Synthetic fraud and demand labels are added to PaySim data; benchmark results are not production guarantees.
- The classifier uses a chronological holdout to reduce time leakage.
- Kafka is required for event publishing and live dashboard event streaming.
- PostgreSQL is provided by Docker Compose, but the current Python application does not persist events to PostgreSQL.
- Do not commit `data/raw`, generated processed CSV files, model binaries, API credentials, or Kaggle credentials.

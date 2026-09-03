# Airflow Task Documentation — Netflix Engineering

> Project: `netflix_engineering` | Airflow 2.7.1 | LocalExecutor | Postgres + Redis
> DAGs location: `airflow/dags/`
> Config: `airflow/config/airflow.cfg` | Compose: `docker-compose.yml`

---

## 1. Overview

This project contains **2 DAGs** that form a two-layer orchestration for a Netflix content-based recommendation model.

| DAG File | DAG ID | Role | Schedule | Tasks |
|---|---|---|---|---|
| `netflix_pipeline.py` | `netflix_pipeline` | **Orchestrator / Trigger** — entry point, triggers the training pipeline and waits | `None` (manual) | 1 |
| `training_pipeline.py` | `netflix_model_training_pipeline` | **Worker / MLOps Pipeline** — ETL + train + evaluate + promote + report + notify | `None` (manual, designed for weekly retraining) | 7 |

Relationship:

```
netflix_pipeline (TriggerDagRunOperator, wait_for_completion=True)
    └──> netflix_model_training_pipeline
              extract_data → preprocess_data → train_model → evaluate_model → promote_model → generate_report → send_notification
```

Both DAGs have `is_paused_upon_creation=False` and `catchup=False`, so they appear unpaused immediately and never backfill.

---

## 2. DAG 1 — `netflix_pipeline.py` (`netflix_pipeline`)

**Path:** `airflow/dags/netflix_pipeline.py` (30 lines)

**Purpose:** Thin wrapper. No data logic. Exists so external callers (API, UI, scheduler, future ETL DAG) have a stable entry point that delegates to the real training DAG.

```python
default_args = { owner: data_team, depends_on_past: False, start_date: 2024-01-01, retries: 1, retry_delay: 5m }
DAG(schedule=None, catchup=False, is_paused_upon_creation=False, tags=[netflix])
```

**Task:**

| Task ID | Operator | Config | What it does |
|---|---|---|---|
| `trigger_training` | `TriggerDagRunOperator` | `trigger_dag_id=netflix_model_training_pipeline`, `wait_for_completion=True` | Triggers the training DAG as a DagRun. This DAG stays in `running` state until the child DAG finishes (success/fail). If child fails, this task fails. |

**Dependencies:** Single node — `trigger_training` (no upstream/downstream beyond itself).

**When to use:** Trigger this DAG instead of the training DAG directly when you want a parent to block/wait or when chaining future pipelines. For ad-hoc retraining you can trigger either DAG — result is the same.

---

## 3. DAG 2 — `training_pipeline.py` (`netflix_model_training_pipeline`)

**Path:** `airflow/dags/training_pipeline.py` (215 lines)
**Docstring:** *Netflix Content Model Training Pipeline — Weekly retraining with MLflow tracking*

```python
default_args = {
  owner: data_team, depends_on_past: False, start_date: 2024-01-01,
  email: [cubudida@gmail.com], email_on_failure: True, retries: 2, retry_delay: 5m, max_active_runs: 1
}
DAG(schedule=None, catchup=False, is_paused_upon_creation=False, tags=[netflix, mlflow, training])
```

**Airflow Variables (with defaults):**

| Variable | Default | Used in |
|---|---|---|
| `mlflow_tracking_uri` | `http://mlflow:5000` | `train_model`, `evaluate_model` (inside Docker network `mlflow:5000` is correct; host maps to `5001:5000`) |
| `model_name` | `netflix_content_model` | `train_model` (register), `evaluate_model`/`promote_model` (load/promote) |
| `data_path` | `/opt/airflow/data/raw/netflix_titles.csv` | `extract_data` → host `./data/raw/netflix_titles.csv` |
| `processed_data_path` | `/opt/airflow/data/processed` | `preprocess_data`, `evaluate_model` → host `./data/processed/` |
| `model_path` | `/opt/airflow/data/models` | `train_model`, `generate_report` → host `./data/models/` |

Set via UI: **Admin → Variables** or via env/CLI. If not set, defaults apply.

**Imports:** `pandas`, `numpy`, `mlflow`, `mlflow.sklearn`, `sklearn (TfidfVectorizer, cosine_similarity)`, `pickle`, `json`, `src.data.preprocess.preprocess_netflix_data` (requires `PYTHONPATH=/opt/airflow`, set in compose).

### 3.1 Task Chain

```
extract_data >> preprocess_data >> train_model >> evaluate_model >> promote_model >> generate_report >> send_notification
```

All 6 PythonOperators use `provide_context=True` (deprecated in Airflow 2.7 but still functional; passed `**context` for XCom).

#### Task 1 — `extract_data` → `extract_data(**context)`

- Reads `DATA_PATH` if `os.path.exists`, else returns `False`.
- Converts DataFrame to JSON string via `df.to_json()` and returns it (pushed to XCom automatically as return value).
- **Failure mode:** If file missing, returns `False` → next task `pd.read_json(False)` will crash.

#### Task 2 — `preprocess_data` → `preprocess_data(**context)`

- Pulls XCom from `extract_data`, `pd.read_json(...)`.
- Calls `src/data/preprocess.py::preprocess_netflix_data(df)`:
  - `director/cast/country → fill Unknown`
  - `rating/duration → fill mode`
  - `date_added → pd.to_datetime → year_added, month_added`
  - `combined_features` initial creation (director+cast+listed_in+description)
- **Re-creates** `combined_features` as `' '.join([str(r[c]) for c in [director,cast,listed_in,description] if r[c] != Unknown])`
- Saves `PROCESSED_PATH/netflix_processed.csv`
- Pushes XCom `key=processed_data` with `df.to_json()` and also returns it.

Source of `preprocess_netflix_data`: `src/data/preprocess.py` (29 lines).

#### Task 3 — `train_model` → `train_model(**context)`

- `mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)`, `mlflow.set_experiment("netflix_content_recommendation")`
- Pulls processed JSON from XCom `preprocess_data`.
- `with mlflow.start_run(run_name=training_YYYYMMDD_HHMMSS)`:
  - `TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1,2))`
  - `tfidf_matrix = tfidf.fit_transform(df['combined_features'])`
  - `similarity = cosine_similarity(tfidf_matrix)` (in-memory, O(n²))
  - `mlflow.log_params({max_features:5000, train_size: len(df)})`
  - `mlflow.log_metrics({sparsity: 1 - (similarity>0).sum()/(n²)})`
  - Saves `MODEL_PATH/tfidf_vectorizer.pkl` (pickle) and `MODEL_PATH/processed_data.csv`
  - `mlflow.sklearn.log_model(tfidf, "tfidf_model", registered_model_name=MODEL_NAME)` → creates/versions model in Registry
  - Pushes `mlflow_run_id` to XCom `key=mlflow_run_id`.

#### Task 4 — `evaluate_model` → `evaluate_model(**context)`

- Pulls `mlflow_run_id` from `train_model`; if missing → log error and skip.
- Loads `PROCESSED_PATH/netflix_processed.csv` and re-creates `combined_features`.
- `client = MlflowClient(); versions = client.get_latest_versions(name=MODEL_NAME)`; if none → skip.
- `model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{versions[0].version}")`
- Samples `min(100, len(df))` rows (seed 42), `model.transform`, `cosine_similarity`
- Computes **Precision@5** proxy: for each sample, checks if its `listed_in` genre appears in the 5 nearest neighbors' `listed_in`:
  ```python
  precision = mean([mean([1 if listed_in in top5 else 0]) for each row])
  ```
- Re-opens run `with mlflow.start_run(run_id=run_id): mlflow.log_metric('evaluation_precision', precision)`
- Pushes `avg_precision` to XCom.

#### Task 5 — `promote_model` → `promote_model(**context)`

- Pulls `avg_precision`; if `>= 0.6`:
  - `versions = client.get_latest_versions(name=MODEL_NAME, stages=['None'])`
  - `client.transition_model_version_stage(MODEL_NAME, versions[0].version, stage='Production')`
- Otherwise does nothing (no demotion, no failure).

#### Task 6 — `generate_report` → `generate_report(**context)`

- Collects XComs: `run_id`, `precision`, `promoted` (truthy check on `promote_model` return).
- Writes JSON to `MODEL_PATH/metrics_report_YYYYMMDD.json`:
  ```json
  { "timestamp": "ISO8601", "run_id": "...", "precision": 0.72, "promoted": "Yes/No" }
  ```

#### Task 7 — `send_notification` → `EmailOperator`

- `to: cubudida@gmail.com`, `subject: Netflix Training Pipeline Complete`, `html_content: <p>Training completed at {{ execution_date }}</p>`
- **Requires SMTP configured** in `airflow.cfg` / env. Currently not configured → this task will fail in default setup, but all prior artifacts are already persisted.

---

## 4. Infrastructure Context

**Compose services** (`docker-compose.yml`):

| Service | Image/Build | Ports (host:container) | Volumes | Notes |
|---|---|---|---|---|
| `postgres` | `postgres:15` | `5432:5432` | `postgres_data:/var/lib/postgresql/data` | Airflow + MLflow backend |
| `redis` | `redis:alpine` | `6379:6379` | — | — |
| `redisinsight` | `redis/redisinsight` | `5540:5540` | — | — |
| `mlflow` | `./mlflow` | `5001:5000` | `./data/models:/mlflow` | `BACKEND_STORE_URI=postgresql+.../mlflow`, `ARTIFACT_ROOT=/mlflow/artifacts` |
| `airflow` (webserver) | `./airflow` | `8080:8080` | `./airflow/dags:/opt/airflow/dags`, `./data:/opt/airflow/data`, `./src:/opt/airflow/src`, `./config:/opt/airflow/config` | `command: airflow db init && users create && webserver` |
| `airflow-scheduler` | `./airflow` | — | same as airflow | `command: airflow scheduler` |
| `api` | `./api` | `8000:8000` | `./data/models:/app/models` | `MLFLOW_TRACKING_URI=http://mlflow:5000` |
| `streamlit` | `./streamlit` | `8501:8501` | `./data:/app/data` | `API_URL=http://api:8000` |
| `prometheus` | `prom/prometheus` | `9090:9090` | `./monitoring/prometheus:/etc/prometheus` | — |
| `grafana` | `./monitoring/grafana` | `3000:3000` | `grafana_data` | — |

**Airflow image requirements** (`airflow/requirements.txt`): `apache-airflow==2.7.1`, `providers-postgres`, `providers-redis`, `pandas==2.0.3`, `numpy`, `scikit-learn==1.5.2`, `mlflow==3.10.1`, `psycopg2-binary`, `python-dotenv`, `pyyaml`.

**Airflow config** (`airflow/config/airflow.cfg`): `executor=LocalExecutor`, `load_examples=False`, `catchup_by_default=False`, `dag_dir_list_interval=15s`, `auth_backends=basic_auth,session` (needed for API trigger).

**Networking:** All services on `netflix_network` bridge. Inside containers use `postgres`, `redis`, `mlflow` hostnames. From host use `localhost:5432/6379/5001/8080/8000/8501`.

---

## 5. Step-by-Step — How to Run Properly

No code change needed. This is code-checkup mode — read-only verification.

### 5.1 Prerequisites

1. **Repo root:** `cd /Users/miftahhadiyannoor/Documents/netflix_engineering`
2. **.env file** (not committed, required by compose — check if `.env.example` exists and copy):
   ```
   POSTGRES_USER=...
   POSTGRES_PASSWORD=...
   POSTGRES_DB=...
   MLFLOW_TRACKING_URI=http://mlflow:5000
   AIRFLOW_WWW_USER=...
   AIRFLOW_WWW_PASSWORD=...
   AIRFLOW_WWW_FIRSTNAME=...
   AIRFLOW_WWW_LASTNAME=...
   AIRFLOW_WWW_EMAIL=...
   GF_SECURITY_ADMIN_USER=...
   GF_SECURITY_ADMIN_PASSWORD=...
   ```
3. **Data:** Place Kaggle Netflix dataset at `./data/raw/netflix_titles.csv` (columns expected: show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description). Create folders `./data/processed` and `./data/models` if missing — DAGs also `os.makedirs(..., exist_ok=True)`.
4. **Docker + Docker Compose** installed.

### 5.2 Start the Stack

```bash
docker compose up --build -d
docker ps                          # all services up
docker logs airflow --tail 50
docker logs airflow-scheduler --tail 50
docker logs mlflow --tail 50
```

Wait ~30s for `airflow db init` and scheduler heartbeat. Airflow webserver creates admin user from `.env` (or fails silently with `|| true` if exists).

### 5.3 Verify Airflow

1. Open `http://localhost:8080` → login with `AIRFLOW_WWW_USER / AIRFLOW_WWW_PASSWORD`.
2. **DAGs list** should show both DAGs unpaused (green). If paused, toggle on.
3. **Admin → Variables** — optionally override defaults:
   - `mlflow_tracking_uri` → `http://mlflow:5000`
   - `model_name` → `netflix_content_model`
   - `data_path` → `/opt/airflow/data/raw/netflix_titles.csv`
   - `processed_data_path` → `/opt/airflow/data/processed`
   - `model_path` → `/opt/airflow/data/models`
4. **No variables needed** if you accept defaults.

### 5.4 Trigger the Pipeline

Choose one:

**A. Via UI (recommended for checkup):**
- Trigger `netflix_pipeline` → it will trigger `netflix_model_training_pipeline` and show as running until child finishes.
- Or trigger `netflix_model_training_pipeline` directly.

**B. Via CLI:**
```bash
docker exec airflow airflow dags trigger netflix_pipeline
# or
docker exec airflow airflow dags trigger netflix_model_training_pipeline
docker exec airflow airflow dags list-runs -o json | head
```

**C. Via REST API** (auth enabled in `airflow.cfg`):
```bash
curl -X POST http://localhost:8080/api/v1/dags/netflix_pipeline/dagRuns \
  -u "$AIRFLOW_WWW_USER:$AIRFLOW_WWW_PASSWORD" -H "Content-Type: application/json" -d '{}'

curl -X POST http://localhost:8080/api/v1/dags/netflix_model_training_pipeline/dagRuns \
  -u "$AIRFLOW_WWW_USER:$AIRFLOW_WWW_PASSWORD" -H "Content-Type: application/json" -d '{}'
```

### 5.5 Monitor Execution

- **Graph / Grid view** in UI → watch the 7 tasks go `queued → running → success`.
- **Logs:** Click any task → Log → check for `✅ Model evaluation completed. Precision@5: ...` and `Model promoted to Production` messages.
- **CLI:**
  ```bash
  docker exec airflow airflow tasks state netflix_model_training_pipeline train_model <execution_date>
  docker logs airflow-scheduler --follow
  ```

### 5.6 Verify Results

| Check | Where | Expected |
|---|---|---|
| Processed data | Host `./data/processed/netflix_processed.csv` and `./data/models/fprocessed_data.csv` | CSV with `year_added`, `month_added`, `combined_features` |
| Vectorizer | `./data/models/tfidf_vectorizer.pkl` | Pickle file, 5000 features |
| MLflow run | `http://localhost:5001` → Experiment `netflix_content_recommendation` | Params `max_features`, `train_size`; Metrics `sparsity`, `evaluation_precision`; Artifact `tfidf_model` |
| Model Registry | MLflow UI → Models → `netflix_content_model` | New version, stage `None` or `Production` if precision ≥ 0.6 |
| Metrics report | `./data/models/metrics_report_YYYYMMDD.json` | JSON with `timestamp`, `run_id`, `precision`, `promoted` |
| Email | Inbox `cubudida@gmail.com` | Only if SMTP configured; otherwise `send_notification` will be `failed` (prior tasks still succeed) |

```bash
ls -lh ./data/processed/ ./data/models/
cat ./data/models/metrics_report_*.json
# MLflow: open http://localhost:5001
```

### 5.7 Stop

```bash
docker compose down        # keep volumes
docker compose down -v     # also wipe postgres/mlflow data
```

---

## 6. What Counts as Success / Failure

**Success:** All tasks `extract → preprocess → train → evaluate → promote → report` are green. `send_notification` may be red if SMTP not set — treat as non-blocking.

**Expected failure points (code-level):**

| Symptom | Cause | Fix |
|---|---|---|
| `extract_data` returns `False`, `preprocess_data` crashes `pd.read_json` | `netflix_titles.csv` missing at `DATA_PATH` | Place file at `./data/raw/netflix_titles.csv` or set Variable `data_path` correctly |
| `ModuleNotFoundError: src.data.preprocess` | `PYTHONPATH` not set or `src` not mounted | Ensure compose volumes include `./src:/opt/airflow/src` and env `PYTHONPATH=/opt/airflow` |
| `mlflow.exceptions.MlflowException: Unable to connect` | `mlflow` service not ready or wrong URI | Use `http://mlflow:5000` inside containers; host is `http://localhost:5001`; check `docker logs mlflow` |
| `No registered model versions found. Skipping evaluation.` | First run, model not yet registered or `train_model` failed | Re-run; check MLflow Registry |
| `send_notification` failed `smtplib` | No SMTP in `airflow.cfg` | Configure `[smtp]` section or mark task as optional / remove from chain for local dev |
| `FutureWarning: fillna(inplace)` / `provide_context` deprecation | Using old pandas/airflow APIs | Non-blocking, but should migrate to `df = df.fillna(...)` and remove `provide_context=True` |

---

## 7. Quick File Reference

```
airflow/dags/netflix_pipeline.py          # DAG netflix_pipeline (1 task: TriggerDagRunOperator)
airflow/dags/training_pipeline.py         # DAG netflix_model_training_pipeline (7 tasks)
src/data/preprocess.py                    # preprocess_netflix_data(df)
airflow/config/airflow.cfg                # LocalExecutor, catchup_by_default=False, auth_backends
airflow/requirements.txt                  # airflow 2.7.1, mlflow 3.10.1, sklearn 1.5.2
docker-compose.yml                        # 10 services, netflix_network
data/raw/netflix_titles.csv               # input (host)
data/processed/netflix_processed.csv      # output (host, via /opt/airflow/data/processed)
data/models/tfidf_vectorizer.pkl          # output
data/models/metrics_report_*.json         # output
```

---

## 8. Notes for Code Checkup (No Run)

- Both DAGs parse correctly with `python -m py_compile`; scheduler will detect them within `dag_dir_list_interval=15s`.
- `schedule=None` means they never run automatically — must be triggered manually or via `TriggerDagRunOperator` / API.
- To validate without Docker: `docker exec airflow airflow dags list` / `airflow dags test netflix_model_training_pipeline 2024-01-01` (runs tasks sequentially without scheduler).
- Do not commit `.env` or `data/` artifacts.


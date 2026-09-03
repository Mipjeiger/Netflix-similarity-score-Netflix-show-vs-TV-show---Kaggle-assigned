from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle
import json
import os
import logging
import sys
from io import StringIO

sys.path.append('/opt/airflow')
from src.data.preprocess import preprocess_netflix_data

"""
Netflix Content Model Training Pipeline
Weekly retraining with MLflow tracking (Optional)
This Airflow DAG defines training and evaluate the task manually by airflow trigger CLI command
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email': ['cubudida@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}

# Configuration - robust Variable.get with env fallback (variables may not exist at parse time)
def _get_var(name, default):
    try:
        return Variable.get(name, default_var=default)
    except Exception:
        return os.getenv(name.upper(), default)

MLFLOW_TRACKING_URI = _get_var('mlflow_tracking_uri', os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
MODEL_NAME = _get_var('model_name', os.getenv('MODEL_NAME', 'netflix_content_model'))
DATA_PATH = _get_var('data_path', '/opt/airflow/data/raw/netflix_titles.csv')
PROCESSED_PATH = _get_var('processed_data_path', '/opt/airflow/data/processed')
MODEL_PATH = _get_var('model_path', '/opt/airflow/data/models')

def extract_data(**context):
    """Extract data from the source CSV file."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        logger.info(f"✅ Data head been read as: \n{df.head()}")

    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Input dataset not found at {DATA_PATH}. "
            "Mount the dataset at ./data/raw/netflix_titles.csv "
            "or set the Airflow data_path variable to the correct path."
        )

    return pd.read_csv(DATA_PATH).to_json()

def preprocess_data(**context):
    """Preprocess the extracted data."""
    ti = context['task_instance']
    extracted_json = ti.xcom_pull(task_ids='extract_data')
    if not extracted_json:
        raise ValueError("extract_data returned no dataset.")

    df = pd.read_json(StringIO(extracted_json))
    df_processed = preprocess_netflix_data(df)
    df_processed['combined_features'] = df_processed.apply(
        lambda r: ' '.join([str(r[c]) for c in ['director','cast','listed_in','description'] if r[c] != 'Unknown']), axis=1
    )
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    df_processed.to_csv(f"{PROCESSED_PATH}/netflix_processed.csv", index=False)
    ti.xcom_push(key='processed_data', value=df_processed.to_json())
    return df_processed.to_json()

def train_model(**context):
    """Train the content-based recommendation model."""
    # Resolve tracking URI at runtime (env overrides Variable)
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Ensure artifact mount exists and is writable
    for p in ["/mlflow", "/mlflow/artifacts", MODEL_PATH]:
        try:
            os.makedirs(p, exist_ok=True)
        except PermissionError as e:
            logger.warning(f"Could not mkdir {p}: {e}")

    mlflow.set_experiment("netflix_content_recommendation")
    ti = context['task_instance']
    raw = ti.xcom_pull(task_ids='preprocess_data', key='processed_data')

    if raw is None:
        raw = ti.xcom_pull(task_ids='preprocess_data')
    if raw is None:
        raise ValueError("preprocess_data returned no data via XCom")

    # Read json data into DataFrame
    df = pd.read_json(StringIO(raw))

    # Mlflow training
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])
        similarity = cosine_similarity(tfidf_matrix)

        # Mlflow logging
        mlflow.log_params({'max_features': 5000, 'train_size': len(df)})
        mlflow.log_metrics({'sparsity': 1 - (similarity > 0).sum().sum() / (similarity.shape[0]**2)})

        os.makedirs(MODEL_PATH, exist_ok=True)
        with open(f"{MODEL_PATH}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(tfidf, f)
        df.to_csv(f"{MODEL_PATH}/processed_data.csv", index=False)

        # Mlflow sklearn model logging
        mlflow.sklearn.log_model(tfidf, name="tfidf_model", registered_model_name=MODEL_NAME)
        run_id = run.info.run_id
        ti.xcom_push(key='mlflow_run_id', value=run_id)
        return run_id

def evaluate_model(**context):
    """Evaluate the trained model"""
    ti = context['task_instance']
    run_id = ti.xcom_pull(task_ids='train_model', key='mlflow_run_id')

    if not run_id:
        logger.error("No MLflow run ID found. Skipping evaluation.")
        return

    df = pd.read_csv(f"{PROCESSED_PATH}/netflix_processed.csv")
    df['combined_features'] = df.apply(
        lambda r: ' '.join([str(r[c]) for c in ['director', 'cast', 'listed_in', 'description'] if r[c] != 'Unknown']), axis=1
    )
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', MLFLOW_TRACKING_URI))
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(name=MODEL_NAME)
    except Exception:
        # MLflow 3.x fallback
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)[:1]

    if not versions:
        logger.error("No registered model versions found. Skipping evaluation.")
        return
    
    ver = versions[0].version
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{ver}")
    sample = df.sample(min(100, len(df)), random_state=42)
    tfidf_matrix = model.transform(sample['combined_features'])
    sim = cosine_similarity(tfidf_matrix)
    precision = np.mean([np.mean([1 if sample.iloc[i]['listed_in'] in sample.iloc[np.argsort(sim[i])[-6:-1]]['listed_in'].values else 0]) for i in range(len(sample))])

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric('evaluation_precision', precision)
        logger.info(f"✅ Model evaluation completed. Precision@5: {precision:.4f}")
    ti.xcom_push(key='avg_precision', value=precision)
    return precision

def promote_model(**context):
    """Promote the model to production"""
    ti = context['task_instance']
    precision = ti.xcom_pull(task_ids='evaluate_model', key='avg_precision')

    if precision and precision >= 0.6:
        client = mlflow.tracking.MlflowClient()
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', MLFLOW_TRACKING_URI))
        try:
            versions = client.get_latest_versions(name=MODEL_NAME, stages=['None'])
        except Exception:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            # filter None stage
            versions = [v for v in versions if v.current_stage == 'None']
            versions = sorted(versions, key=lambda v: int(v.version), reverse=True)[:1]

        if versions:
            try:
                client.transition_model_version_stage(MODEL_NAME, versions[0].version, stage='Production')
            except Exception:
                # MLflow 3.x uses set_registered_model_alias or update stage via client
                try:
                    client.set_registered_model_alias(MODEL_NAME, "production", versions[0].version)
                except Exception as e:
                    logger.warning(f"Could not promote model: {e}")
            logger.info(f"✅ Model promoted to Production. Version: {versions[0].version}")

def generate_report(**context):
    """Generate a report of the training and evaluation"""
    ti = context['task_instance']
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'run_id': ti.xcom_pull(task_ids='train_model', key='mlflow_run_id'),
        'precision': ti.xcom_pull(task_ids='evaluate_model', key='avg_precision'),
        'promoted': 'Yes' if ti.xcom_pull(task_ids='promote_model') else 'No'
    }
    with open(f"{MODEL_PATH}/metrics_report_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
        json.dump(report, f)

# Define the DAG
dag = DAG(
    'netflix_model_training_pipeline',
    default_args=default_args,
    description='A weekly training pipeline for Netflix content recommendation model',
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=['netflix', 'mlflow', 'training']
)

# Define the tasks
extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    provide_context=True,
    dag=dag
)

preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag
)

train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)

evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag
)

promote = PythonOperator(
    task_id='promote_model',
    python_callable=promote_model,
    provide_context=True,
    dag=dag
)

report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    provide_context=True,
    dag=dag
)

email = EmailOperator(
    task_id='send_notification',
    to='cubudida@gmail.com',
    subject='Netflix Training Pipeline Complete',
    html_content='<p>Training completed at {{ execution_date }}</p>',
    dag=dag
)

# Define task dependencies
extract >> preprocess >> train >> evaluate >> promote >> report >> email
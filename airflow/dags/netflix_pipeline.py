from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'netflix_pipeline',
    default_args=default_args,
    description='Trigger training pipeline',
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=['netflix']
)

trigger_training = TriggerDagRunOperator(
    task_id='trigger_training',
    trigger_dag_id='netflix_model_training_pipeline',
    wait_for_completion=True,
    dag=dag
)

# Define the task dependencies
trigger_training
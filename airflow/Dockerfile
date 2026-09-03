FROM apache/airflow:2.7.1-python3.11

USER root

RUN apt-get update && apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY dags/ /opt/airflow/dags/
COPY config/ /opt/airflow/config/

ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
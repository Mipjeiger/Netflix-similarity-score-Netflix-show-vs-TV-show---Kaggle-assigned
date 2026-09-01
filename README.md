# 🎬 Netflix Content Analysis MLOps Platform

[![CI Pipeline](https://github.com/yourusername/netflix_engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/netflix_engineering/actions/workflows/ci.yml)
[![CD Pipeline](https://github.com/yourusername/netflix_engineering/actions/workflows/cd.yml/badge.svg)](https://github.com/yourusername/netflix_engineering/actions/workflows/cd.yml)
[![Security Scan](https://github.com/yourusername/netflix_engineering/actions/workflows/security_scan.yml/badge.svg)](https://github.com/yourusername/netflix_engineering/actions/workflows/security_scan.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/yourusername/netflix-api)](https://hub.docker.com/r/yourusername/netflix-api)

## 🚀 Overview

End-to-end MLOps platform for Netflix content analysis with a **content recommendation engine**. This project demonstrates production-grade ML engineering with:

- 📊 **Content Analysis**: EDA, network analysis, and trend detection
- 🤖 **Recommendation Engine**: TF-IDF + Cosine Similarity for content matching
- 🔄 **MLOps Pipeline**: Automated training, tracking, and deployment
- 📈 **Monitoring**: Real-time metrics and performance tracking

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| 📊 **EDA Dashboard** | Interactive visualizations with Plotly | ✅ |
| 🔍 **Content Recommendations** | Similar content based on TF-IDF | ✅ |
| 🕸️ **Network Analysis** | Director-Actor collaboration graph | ✅ |
| 🔄 **Auto Retraining** | Weekly model updates via Airflow | ✅ |
| 📈 **Experiment Tracking** | MLflow for version control | ✅ |
| 📉 **Performance Monitoring** | Grafana + Prometheus dashboards | ✅ |
| 🚀 **CI/CD Pipeline** | GitHub Actions for automation | ✅ |
| 🐳 **Containerized** | Docker Compose for easy deployment | ✅ |

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Orchestration** | Apache Airflow |
| **ML Tracking** | MLflow |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit + Plotly |
| **Monitoring** | Grafana + Prometheus |
| **Database** | PostgreSQL + Redis |
| **CI/CD** | GitHub Actions |
| **Container** | Docker + Docker Compose |
| **ML** | scikit-learn, pandas, numpy, networkx |


## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/netflix_engineering.git
cd netflix_engineering

# 2. Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Deploy all services
chmod +x scripts/deploy.sh
./scripts/deploy.sh

Service	URL	Credentials
🖥️ Streamlit Dashboard	http://localhost:8501	-
📡 FastAPI Docs	http://localhost:8000/docs	-
🔧 Airflow UI	http://localhost:8080	admin/admin
📊 MLflow UI	http://localhost:5000	-
📈 Grafana	http://localhost:3000	admin/admin
🔄 CI/CD Pipeline

Continuous Integration (CI)

✅ Code linting (Black, Flake8, isort, mypy, pylint)
✅ Unit & integration tests (pytest)
✅ Security scanning (Bandit, Trivy, Safety)
✅ Docker image building
✅ Model validation with MLflow
Continuous Deployment (CD)

🚀 Automatic deployment to staging
🚀 Production deployment after approval
🔄 Automatic rollback on failure
📧 Slack notifications

Access Services

Service	URL	Credentials
🖥️ Streamlit Dashboard	http://localhost:8501	-
📡 FastAPI Docs	http://localhost:8000/docs	-
🔧 Airflow UI	http://localhost:8080	admin/admin
📊 MLflow UI	http://localhost:5000	-
📈 Grafana	http://localhost:3000	admin/admin
🔄 CI/CD Pipeline

Continuous Integration (CI)

✅ Code linting (Black, Flake8, isort, mypy, pylint)
✅ Unit & integration tests (pytest)
✅ Security scanning (Bandit, Trivy, Safety)
✅ Docker image building
✅ Model validation with MLflow
Continuous Deployment (CD)

🚀 Automatic deployment to staging
🚀 Production deployment after approval
🔄 Automatic rollback on failure
📧 Slack notifications
Workflows

yaml
.github/workflows/
├── ci.yml                 # Runs on PR and push to main
├── cd.yml                 # Deploys to staging/production
├── model_validation.yml   # Validates model performance
└── security_scan.yml      # Security vulnerability scanning

🧪 Testing

bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
📊 Model Performance

The recommendation model uses TF-IDF vectorization with cosine similarity:

Metric	Value
Model Type	TF-IDF + Cosine Similarity
Features	5,000 max features
Training Data	8,807 Netflix titles
Evaluation Metric	Precision@5
Threshold	0.6 for production promotion

🔍 Monitoring

Grafana Dashboards

API request rate and latency
Model accuracy and drift
System resource usage
Error rates and alerts
Prometheus Metrics

python
# Available metrics
- api_requests_total
- api_latency_seconds
- model_accuracy
- model_prediction_count
- error_rate

📝 API Endpoints

yaml
GET /health:
  Description: Health check
  Response: {"status": "healthy"}

POST /recommend:
  Description: Get content recommendations
  Request: {"title": "Stranger Things", "n_recommendations": 5}
  Response: [{"title": "...", "type": "...", "genre": "...", "similarity_score": 0.85}]

GET /stats/content:
  Description: Get content statistics
  Response: {"total_titles": 8807, "movies": 6131, "tv_shows": 2676}
🐳 Docker Commands

bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f api

# Scale services
docker-compose up -d --scale api=3

# Rebuild and restart
docker-compose up -d --build

🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open a Pull Request
Development Guidelines

✅ Write tests for new features
✅ Update documentation
✅ Follow PEP 8 style guide
✅ Use pre-commit hooks: pre-commit install
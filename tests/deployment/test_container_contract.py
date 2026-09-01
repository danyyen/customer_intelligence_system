"""Static checks for the container contract when Docker is unavailable in CI."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_runs_as_non_root_and_has_readiness_check():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim" in dockerfile
    assert "USER appuser" in dockerfile
    assert "/health/ready" in dockerfile
    assert 'CMD ["python", "-m", "customer_intelligence.api.server"]' in dockerfile


def test_runtime_requirements_exclude_analysis_stack():
    requirements = (ROOT / "requirements-api.txt").read_text(encoding="utf-8")
    assert "scikit-learn==1.9.0" in requirements
    assert "fastapi==0.116.1" in requirements
    assert "jupyter" not in requirements.lower()
    assert "matplotlib" not in requirements.lower()
    assert "seaborn" not in requirements.lower()


def test_compose_waits_for_database_and_bootstrap():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"database", "bootstrap", "api"}
    assert services["bootstrap"]["depends_on"]["database"]["condition"] == "service_healthy"
    assert services["api"]["depends_on"]["bootstrap"]["condition"] == "service_completed_successfully"
    assert services["api"]["environment"]["PORT"] == "8000"


def test_docker_context_excludes_raw_and_development_data():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "uci data" in ignored
    assert "notebooks" in ignored
    assert "tests" in ignored
    assert ".venv" in ignored
    assert "!data/processed/latest_customer_decisions.csv" in ignored

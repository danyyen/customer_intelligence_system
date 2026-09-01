# Docker deployment guide

## Why Docker exists

The model artifact was created with a specific Python and scikit-learn
environment. Docker packages that environment with the API, preventing a host
machine from silently using incompatible library versions.

The image contains the API code, minimal serving dependencies, approved model
artifact and verified bootstrap decisions. It deliberately excludes raw
transactions, notebooks, tests, development environments and IDE files.

The packaged decision CSV is acceptable here only because this public portfolio
dataset uses anonymized customer identifiers. A real company should not bake
customer decisions into an image; the bootstrap job would read them from
controlled object storage or produce them directly into PostgreSQL.

## Components

```text
PostgreSQL container
        ↑
one-time bootstrap container
        ↑
verified decision CSV

PostgreSQL container ← FastAPI container ← HTTP client
```

`bootstrap` completes before the API starts. It publishes the decision snapshot
transactionally into PostgreSQL. The API then serves customer lookups without
running feature engineering during HTTP requests.

## Install Docker Desktop

Docker is not currently installed on the development machine. Install Docker
Desktop for Windows, enable its WSL 2 backend when prompted, start Docker
Desktop, and verify:

```powershell
docker --version
docker compose version
```

## Build and start locally

From the repository root:

```powershell
docker compose up --build
```

Then open:

```text
Swagger UI: http://localhost:8000/docs
Readiness:   http://localhost:8000/health/ready
Customers:   http://localhost:8000/v1/customers?limit=5
```

Use another terminal for a smoke check:

```powershell
Invoke-RestMethod http://localhost:8000/health/ready
Invoke-RestMethod "http://localhost:8000/v1/customers?strategy=value_protection&limit=2"
docker compose ps
```

Stop containers while retaining PostgreSQL data:

```powershell
docker compose down
```

Stop containers and delete the local database volume:

```powershell
docker compose down --volumes
```

The second command is destructive and should be used only when intentionally
resetting local Docker data.

## Image-only alternative

An API image can run without Compose only when `DATABASE_URL` points to an
already available database containing a published decision snapshot:

```powershell
docker build -t customer-intelligence-api:local .
docker run --rm -p 8000:8000 `
  -e DATABASE_URL="postgresql://user:password@host:5432/customer_intelligence" `
  customer-intelligence-api:local
```

## Trade-offs

- `python:3.13-slim` is smaller than the full Python image but easier to debug
  than ultra-minimal/distroless images.
- A single API process is appropriate for the current portfolio scale. Cloud
  platforms can run additional container replicas later.
- The local password in `compose.yaml` is intentionally non-secret and must
  never be reused outside local development.
- PostgreSQL adds setup complexity but accurately represents a shared,
  persistent production serving store. SQLite remains the simpler alternative
  for Python-only local development.

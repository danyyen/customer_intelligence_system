# Render deployment guide

## What the Blueprint creates

`render.yaml` is infrastructure as code: a reviewable recipe for the cloud
resources required by this application. Render reads it and creates:

1. `customer-intelligence-db`, a managed PostgreSQL database.
2. `customer-intelligence-api`, a public Docker-based FastAPI service.
3. A private `DATABASE_URL` connection from the API to PostgreSQL.

The database password is generated and managed by Render. It is never written
to this repository.

## First deployment

1. Sign in to Render and connect the GitHub account that can access this repo.
2. Choose **New > Blueprint**.
3. Select `danyyen/customer_intelligence_system`.
4. Confirm the Blueprint path is `render.yaml`.
5. Review the two free resources and select **Apply**.
6. Follow the Events logs until the database, image build, snapshot bootstrap,
   and API startup complete.

After deployment, Render provides an `onrender.com` service URL. Verify:

```text
https://YOUR-SERVICE.onrender.com/health/live
https://YOUR-SERVICE.onrender.com/health/ready
https://YOUR-SERVICE.onrender.com/docs
https://YOUR-SERVICE.onrender.com/v1/customers?limit=5
```

## Why startup loads the decision snapshot

Render's `preDeployCommand` is available only to paid services. This portfolio
starts on the free plan, so its Docker command first publishes the bundled,
verified customer-decision snapshot and then starts FastAPI. Publication is a
single database transaction: clients see a complete snapshot, never half of
one.

This is an intentional free-tier compromise. It is suitable here because there
is one small API instance and anonymized public portfolio data. In a real
production system, use a paid pre-deploy command or scheduled scoring workflow,
store the input in controlled object storage, and use versioned database
migrations.

## Free-tier limitations

- The API sleeps after 15 minutes without traffic, so the first request can take
  around a minute.
- A free PostgreSQL database expires after 30 days and has no backups.
- Free PostgreSQL is limited to 1 GB.
- This deployment is a demonstration, not a production SLA.

Upgrade the web service and database before treating the application as an
always-available production system. After upgrading, move the bootstrap command
to `preDeployCommand` and remove the `dockerCommand` override.

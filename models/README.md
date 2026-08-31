# Model artifacts

`churn_logistic_v1.joblib` is produced by:

```powershell
python -m scripts.package_churn_model
```

Only load model artifacts created by this repository. Joblib/pickle artifacts
can execute code during loading and must never be accepted from untrusted users.

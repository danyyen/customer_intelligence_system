"""Start the hosted API after preparing its serving database.

The free Render plan cannot run a separate pre-deploy command. This launcher
therefore performs the two required operations in a predictable order without
depending on shell quoting or ``&&`` behavior:

1. Publish the verified customer-decision snapshot to PostgreSQL.
2. Start the FastAPI server only after publication succeeds.

If database publication fails, the exception stops the process. Render then
marks the deployment unhealthy instead of serving an API with missing data.
"""

from __future__ import annotations

from scripts.load_decisions_to_database import main as publish_decisions

from .server import main as start_api


def main() -> None:
    """Prepare the serving store and then run the HTTP service."""
    publish_decisions()
    start_api()


if __name__ == "__main__":
    main()

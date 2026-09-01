"""Container-friendly Uvicorn entry point.

Cloud platforms provide the listening port through ``PORT``. Keeping that
logic here avoids shell expansion tricks in the Dockerfile's startup command.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "customer_intelligence.api.main:app",
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()

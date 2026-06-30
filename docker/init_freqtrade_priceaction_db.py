"""Initialize the parallel freqtrade_priceaction database schema."""
from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine


sys.path.insert(0, "/freqtrade/user_data/strategies")

from price_action.models import _Base  # noqa: E402


def main() -> None:
    db_url = os.environ["PA_DB_URL"]
    engine = create_engine(db_url)
    _Base.metadata.create_all(engine)
    engine.dispose()

    print("freqtrade_priceaction db initialized")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import bisect
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COINBASE_API_BASE = "https://api.exchange.coinbase.com"
MAX_TRADES_PER_REQUEST = 1000
DEFAULT_MAX_TRADES = 500_000
GRANULARITIES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}


def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("datetime value cannot be empty")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid datetime {value!r}. Use ISO format like 2026-07-01T00:00:00Z."
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def isoformat_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def product_id(crypto: str, quote: str = "USD") -> str:
    return f"{crypto.strip().upper()}-{quote.strip().upper()}"


def lookback_window(days: int) -> Tuple[datetime, datetime]:
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=max(1, days))
    return window_start, window_end


class CryptoPriceDatasetBuilder:
    """Build Y-train/Y-test target rows from Coinbase Exchange trade prices."""

    def __init__(
        self,
        api_base_url: str = COINBASE_API_BASE,
        timeout: int = 30,
        max_trades: int = DEFAULT_MAX_TRADES,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.max_trades = max(1, max_trades)
        self.session = requests.Session()

    def build_dataset(
        self,
        crypto: str,
        days: int,
        test_ratio: float = 0.2,
        granularity: str = "1m",
        align_with: Optional[str] = None,
    ) -> Dict[str, Any]:
        window_start, window_end = lookback_window(days)
        product = product_id(crypto)
        trades = self.fetch_trades(product, window_start, window_end)
        if len(trades) < 2:
            raise RuntimeError(
                f"At least two {product} trades are required in the requested window."
            )

        if align_with:
            target_timestamps = self._load_alignment_timestamps(align_with)
            price_rows = self._rows_at_timestamps(trades, target_timestamps, window_start, window_end)
            notes = [
                "Y rows are aligned to timestamps from the Kalshi X dataset.",
                "Each price is the most recent trade at or before the aligned timestamp.",
            ]
        else:
            price_rows = self._rows_on_grid(trades, window_start, window_end, GRANULARITIES[granularity])
            notes = [
                "Y rows are sampled on a regular time grid.",
                "Each price is the most recent trade at or before the row timestamp.",
            ]

        if len(price_rows) < 2:
            raise RuntimeError("At least two target rows are required to create y_train/y_test splits.")

        splits = split_targets(price_rows, test_ratio)
        return {
            "crypto": crypto.upper(),
            "quote_currency": "USD",
            "product_id": product,
            "requested_days": days,
            "window_start": isoformat_z(window_start),
            "window_end": isoformat_z(window_end),
            "test_ratio": test_ratio,
            "granularity": granularity if not align_with else None,
            "aligned_with": align_with,
            "source": f"Coinbase Exchange {product} trades",
            "target_definition": (
                "Each row contains a timestamp and the USD price of the most recent "
                f"{crypto.upper()} trade at or before that timestamp."
            ),
            "y_train": splits["y_train"],
            "y_test": splits["y_test"],
            "y_train_count": len(splits["y_train"]),
            "y_test_count": len(splits["y_test"]),
            "train_start": splits["train_start"],
            "train_end": splits["train_end"],
            "test_start": splits["test_start"],
            "test_end": splits["test_end"],
            "notes": notes,
        }

    def fetch_trades(
        self,
        product: str,
        window_start: datetime,
        window_end: datetime,
    ) -> List[Dict[str, Any]]:
        url = f"{self.api_base_url}/products/{product}/trades"
        headers = {"User-Agent": "sigwork-crypto-price-dataset/1.0"}
        trades: List[Dict[str, Any]] = []
        before: Optional[int] = None

        while len(trades) < self.max_trades:
            params: Dict[str, Any] = {"limit": MAX_TRADES_PER_REQUEST}
            if before is not None:
                params["before"] = before

            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break

            reached_window_start = False
            for row in payload:
                if not isinstance(row, dict):
                    continue
                trade_time = parse_datetime(str(row["time"]))
                if trade_time > window_end:
                    continue
                if trade_time < window_start:
                    reached_window_start = True
                    break

                trades.append(
                    {
                        "timestamp": isoformat_z(trade_time),
                        "unix_time": int(trade_time.timestamp()),
                        "price": float(row["price"]),
                        "trade_id": row.get("trade_id"),
                    }
                )

            if reached_window_start:
                break

            oldest_trade_id = payload[-1].get("trade_id")
            if oldest_trade_id is None or oldest_trade_id == before:
                break
            before = int(oldest_trade_id)

        trades.sort(key=lambda item: item["unix_time"])
        logger.info("Fetched %s trades for %s", len(trades), product)
        return trades

    def _load_alignment_timestamps(self, path: str) -> List[datetime]:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)

        rows = payload.get("training_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"{path!r} must contain a non-empty training_rows list from kalshi_crypto_pipeline."
            )

        timestamps = [parse_datetime(str(row["timestamp"])) for row in rows if "timestamp" in row]
        if len(timestamps) < 2:
            raise ValueError(f"{path!r} did not provide enough timestamps for Y alignment.")
        timestamps.sort()
        return timestamps

    def _rows_at_timestamps(
        self,
        trades: Sequence[Dict[str, Any]],
        target_timestamps: Sequence[datetime],
        window_start: datetime,
        window_end: datetime,
    ) -> List[Dict[str, Any]]:
        trade_times = [trade["unix_time"] for trade in trades]
        trade_prices = [trade["price"] for trade in trades]
        rows: List[Dict[str, Any]] = []

        for target_time in target_timestamps:
            if target_time < window_start or target_time > window_end:
                continue
            index = bisect.bisect_right(trade_times, int(target_time.timestamp())) - 1
            if index < 0:
                continue
            rows.append(
                {
                    "timestamp": isoformat_z(target_time),
                    "price": trade_prices[index],
                }
            )
        return rows

    def _rows_on_grid(
        self,
        trades: Sequence[Dict[str, Any]],
        window_start: datetime,
        window_end: datetime,
        granularity_seconds: int,
    ) -> List[Dict[str, Any]]:
        trade_times = [trade["unix_time"] for trade in trades]
        trade_prices = [trade["price"] for trade in trades]
        rows: List[Dict[str, Any]] = []

        cursor = window_start
        while cursor <= window_end:
            index = bisect.bisect_right(trade_times, int(cursor.timestamp())) - 1
            if index >= 0:
                rows.append(
                    {
                        "timestamp": isoformat_z(cursor),
                        "price": trade_prices[index],
                    }
                )
            cursor += timedelta(seconds=granularity_seconds)
        return rows


def split_targets(price_rows: List[Dict[str, Any]], test_ratio: float) -> Dict[str, Any]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be greater than 0 and less than 1")

    split_index = max(1, min(len(price_rows) - 1, int(len(price_rows) * (1.0 - test_ratio))))
    train_records = price_rows[:split_index]
    test_records = price_rows[split_index:]

    return {
        "train_start": train_records[0]["timestamp"],
        "train_end": train_records[-1]["timestamp"],
        "test_start": test_records[0]["timestamp"],
        "test_end": test_records[-1]["timestamp"],
        "y_train": train_records,
        "y_test": test_records,
    }


def build_dataset(
    crypto: str,
    days: int,
    output_path: str,
    test_ratio: float = 0.2,
    granularity: str = "1m",
    align_with: Optional[str] = None,
    max_trades: int = DEFAULT_MAX_TRADES,
) -> Dict[str, Any]:
    builder = CryptoPriceDatasetBuilder(max_trades=max_trades)
    dataset = builder.build_dataset(
        crypto=crypto,
        days=days,
        test_ratio=test_ratio,
        granularity=granularity,
        align_with=align_with,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Y-only crypto price dataset (y_train/y_test) using Coinbase trade prices. "
            "Use kalshi_crypto_pipeline.py to generate the corresponding X dataset."
        )
    )
    parser.add_argument("--crypto", default="BTC", help="Crypto asset symbol, e.g. BTC or ETH")
    parser.add_argument("--days", type=int, default=60, help="Rolling lookback window in days")
    parser.add_argument(
        "--granularity",
        choices=sorted(GRANULARITIES),
        default="1m",
        help="Sample interval when not using --align-with",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Chronological fraction of rows assigned to y_test",
    )
    parser.add_argument(
        "--align-with",
        help="Kalshi JSON export path; Y timestamps match training_rows from that X dataset",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=DEFAULT_MAX_TRADES,
        help="Maximum Coinbase trades to fetch while building the dataset",
    )
    parser.add_argument("--output", default="btc_price_dataset.json", help="Path to write the JSON dataset")
    args = parser.parse_args()

    dataset = build_dataset(
        crypto=args.crypto,
        days=args.days,
        output_path=args.output,
        test_ratio=args.test_ratio,
        granularity=args.granularity,
        align_with=args.align_with,
        max_trades=args.max_trades,
    )
    print(
        json.dumps(
            {
                "crypto": dataset["crypto"],
                "y_train": dataset["y_train_count"],
                "y_test": dataset["y_test_count"],
                "window_start": dataset["window_start"],
                "window_end": dataset["window_end"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

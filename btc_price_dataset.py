from __future__ import annotations

import argparse
import bisect
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COINBASE_API_BASE = "https://api.exchange.coinbase.com"
MAX_TRADES_PER_REQUEST = 1000
MAX_CANDLES_PER_REQUEST = 300
DEFAULT_MAX_TRADES = 500_000
DEFAULT_REQUEST_DELAY = 0.35
DEFAULT_MAX_RETRIES = 8
CANDLE_LOOKBACK_DAYS = 7
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
    """Build Y-train/Y-test target rows from Coinbase Exchange prices."""

    def __init__(
        self,
        api_base_url: str = COINBASE_API_BASE,
        timeout: int = 30,
        max_trades: int = DEFAULT_MAX_TRADES,
        request_delay: float = DEFAULT_REQUEST_DELAY,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.max_trades = max(1, max_trades)
        self.request_delay = max(0.0, request_delay)
        self.max_retries = max(1, max_retries)
        self.session = requests.Session()

    def build_dataset(
        self,
        crypto: str,
        days: int,
        test_ratio: float = 0.2,
        granularity: str = "1m",
        align_with: Optional[str] = None,
        price_source: str = "auto",
    ) -> Dict[str, Any]:
        window_start, window_end = lookback_window(days)
        product = product_id(crypto)
        use_candles = self._should_use_candles(price_source, days, align_with is not None)
        price_points = self.fetch_price_points(
            product,
            window_start,
            window_end,
            use_candles=use_candles,
            candle_granularity_seconds=GRANULARITIES[granularity],
        )
        if len(price_points) < 2:
            raise RuntimeError(
                f"At least two {product} price points are required in the requested window."
            )

        if align_with:
            target_timestamps = self._load_alignment_timestamps(align_with)
            price_rows = self._rows_at_timestamps(price_points, target_timestamps, window_start, window_end)
            notes = [
                "Y rows are aligned to timestamps from the Kalshi X dataset.",
                "Each price is the most recent market price at or before the aligned timestamp.",
            ]
        else:
            price_rows = self._rows_on_grid(
                price_points,
                window_start,
                window_end,
                GRANULARITIES[granularity],
            )
            notes = [
                "Y rows are sampled on a regular time grid.",
                "Each price is the most recent market price at or before the row timestamp.",
            ]

        if len(price_rows) < 2:
            raise RuntimeError("At least two target rows are required to create y_train/y_test splits.")

        splits = split_targets(price_rows, test_ratio)
        source_label = (
            f"Coinbase Exchange {product} 1-minute candle close prices"
            if use_candles
            else f"Coinbase Exchange {product} trades"
        )
        target_definition = (
            "Each row contains a timestamp and the USD price of the most recent "
            f"{crypto.upper()} "
            + ("candle close (last trade in the interval)." if use_candles else "trade at or before that timestamp.")
        )

        return {
            "crypto": crypto.upper(),
            "quote_currency": "USD",
            "product_id": product,
            "requested_days": days,
            "window_start": isoformat_z(window_start),
            "window_end": isoformat_z(window_end),
            "test_ratio": test_ratio,
            "granularity": granularity if not align_with else None,
            "price_source": "candles" if use_candles else "trades",
            "aligned_with": align_with,
            "source": source_label,
            "target_definition": target_definition,
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

    def _should_use_candles(self, price_source: str, days: int, aligned: bool) -> bool:
        if price_source == "candles":
            return True
        if price_source == "trades":
            return False
        return aligned or days >= CANDLE_LOOKBACK_DAYS

    def fetch_price_points(
        self,
        product: str,
        window_start: datetime,
        window_end: datetime,
        *,
        use_candles: bool,
        candle_granularity_seconds: int = 60,
    ) -> List[Dict[str, Any]]:
        if use_candles:
            return self.fetch_candles(product, window_start, window_end, candle_granularity_seconds)
        return self.fetch_trades(product, window_start, window_end)

    def _request_json(self, url: str, params: Dict[str, Any], headers: Dict[str, str]) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            if self.request_delay:
                time.sleep(self.request_delay)

            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else min(60.0, 2.0 ** attempt)
                logger.warning(
                    "Rate limited by Coinbase (429); sleeping %.1fs (attempt %s/%s)",
                    sleep_seconds,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(sleep_seconds)
                last_error = requests.HTTPError(
                    f"429 Too Many Requests for url: {response.url}", response=response
                )
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                last_error = exc
                if response.status_code >= 500 and attempt + 1 < self.max_retries:
                    sleep_seconds = min(60.0, 2.0 ** attempt)
                    logger.warning(
                        "Coinbase server error %s; retrying in %.1fs",
                        response.status_code,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise

            return response.json()

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts")

    def fetch_candles(
        self,
        product: str,
        window_start: datetime,
        window_end: datetime,
        granularity_seconds: int,
    ) -> List[Dict[str, Any]]:
        url = f"{self.api_base_url}/products/{product}/candles"
        headers = {"User-Agent": "sigwork-crypto-price-dataset/1.0"}
        points: Dict[int, Dict[str, Any]] = {}
        chunk_seconds = granularity_seconds * MAX_CANDLES_PER_REQUEST
        chunk_start = window_start

        while chunk_start < window_end:
            chunk_end = min(window_end, chunk_start + timedelta(seconds=chunk_seconds))
            params = {
                "start": isoformat_z(chunk_start),
                "end": isoformat_z(chunk_end),
                "granularity": granularity_seconds,
            }
            payload = self._request_json(url, params, headers)
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected Coinbase candle response: {payload!r}")

            logger.info(
                "Fetched %s candles for %s from %s to %s",
                len(payload),
                product,
                params["start"],
                params["end"],
            )
            for row in payload:
                if not isinstance(row, list) or len(row) < 5:
                    continue
                timestamp, _low, _high, _open_price, close_price = row[:5]
                timestamp = int(timestamp)
                points[timestamp] = {
                    "timestamp": isoformat_z(datetime.fromtimestamp(timestamp, tz=timezone.utc)),
                    "unix_time": timestamp,
                    "price": float(close_price),
                }

            chunk_start = chunk_end

        ordered = [points[key] for key in sorted(points)]
        logger.info("Collected %s candle price points for %s", len(ordered), product)
        return ordered

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

            payload = self._request_json(url, params, headers)
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
        price_points: Sequence[Dict[str, Any]],
        target_timestamps: Sequence[datetime],
        window_start: datetime,
        window_end: datetime,
    ) -> List[Dict[str, Any]]:
        point_times = [point["unix_time"] for point in price_points]
        point_prices = [point["price"] for point in price_points]
        rows: List[Dict[str, Any]] = []

        for target_time in target_timestamps:
            if target_time < window_start or target_time > window_end:
                continue
            index = bisect.bisect_right(point_times, int(target_time.timestamp())) - 1
            if index < 0:
                continue
            rows.append(
                {
                    "timestamp": isoformat_z(target_time),
                    "price": point_prices[index],
                }
            )
        return rows

    def _rows_on_grid(
        self,
        price_points: Sequence[Dict[str, Any]],
        window_start: datetime,
        window_end: datetime,
        granularity_seconds: int,
    ) -> List[Dict[str, Any]]:
        point_times = [point["unix_time"] for point in price_points]
        point_prices = [point["price"] for point in price_points]
        rows: List[Dict[str, Any]] = []

        cursor = window_start
        while cursor <= window_end:
            index = bisect.bisect_right(point_times, int(cursor.timestamp())) - 1
            if index >= 0:
                rows.append(
                    {
                        "timestamp": isoformat_z(cursor),
                        "price": point_prices[index],
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
    request_delay: float = DEFAULT_REQUEST_DELAY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    price_source: str = "auto",
) -> Dict[str, Any]:
    builder = CryptoPriceDatasetBuilder(
        max_trades=max_trades,
        request_delay=request_delay,
        max_retries=max_retries,
    )
    dataset = builder.build_dataset(
        crypto=crypto,
        days=days,
        test_ratio=test_ratio,
        granularity=granularity,
        align_with=align_with,
        price_source=price_source,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Y-only crypto price dataset (y_train/y_test) using Coinbase prices. "
            "Use kalshi_crypto_pipeline.py to generate the corresponding X dataset."
        )
    )
    parser.add_argument("--crypto", default="BTC", help="Crypto asset symbol, e.g. BTC or ETH")
    parser.add_argument("--days", type=int, default=60, help="Rolling lookback window in days")
    parser.add_argument(
        "--granularity",
        choices=sorted(GRANULARITIES),
        default="1m",
        help="Candle/grid interval used for price lookup",
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
        "--price-source",
        choices=["auto", "candles", "trades"],
        default="auto",
        help=(
            "Price data source. auto uses candles for --align-with or --days >= 7 "
            "to avoid Coinbase rate limits."
        ),
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=DEFAULT_MAX_TRADES,
        help="Maximum Coinbase trades to fetch when --price-source trades is selected",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help="Seconds to wait between Coinbase API requests",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries after Coinbase 429/5xx responses",
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
        request_delay=args.request_delay,
        max_retries=args.max_retries,
        price_source=args.price_source,
    )
    print(
        json.dumps(
            {
                "crypto": dataset["crypto"],
                "price_source": dataset["price_source"],
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

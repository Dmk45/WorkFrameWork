import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from btc_price_dataset import CryptoPriceDatasetBuilder


class FixtureCryptoPriceDatasetBuilder(CryptoPriceDatasetBuilder):
    def __init__(self, trades):
        super().__init__(request_delay=0.0)
        self.trades = trades

    def fetch_price_points(self, product, window_start, window_end, *, use_candles, candle_granularity_seconds=60):
        return [
            trade
            for trade in self.trades
            if window_start.timestamp() <= trade["unix_time"] <= window_end.timestamp()
        ]


class CryptoPriceDatasetTests(unittest.TestCase):
    def test_build_dataset_exports_unsplit_y_rows(self) -> None:
        base = datetime(2026, 7, 16, 18, 0, tzinfo=timezone.utc)
        trades = [
            {
                "timestamp": (base + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                "unix_time": int((base + timedelta(minutes=1)).timestamp()),
                "price": 64000.0,
                "trade_id": 1,
            },
            {
                "timestamp": (base + timedelta(minutes=3)).isoformat().replace("+00:00", "Z"),
                "unix_time": int((base + timedelta(minutes=3)).timestamp()),
                "price": 64100.0,
                "trade_id": 2,
            },
        ]
        builder = FixtureCryptoPriceDatasetBuilder(trades)
        dataset = builder.build_dataset(crypto="BTC", days=1, granularity="1m")

        self.assertIn("y_rows", dataset)
        self.assertNotIn("y_train", dataset)
        self.assertNotIn("y_test", dataset)
        self.assertTrue(all("timestamp" in row and "price" in row for row in dataset["y_rows"]))

    def test_auto_price_source_uses_candles_for_long_or_aligned_windows(self) -> None:
        builder = CryptoPriceDatasetBuilder(request_delay=0.0)
        self.assertTrue(builder._should_use_candles("auto", days=120, aligned=True))
        self.assertTrue(builder._should_use_candles("auto", days=30, aligned=False))
        self.assertFalse(builder._should_use_candles("auto", days=3, aligned=False))
        self.assertTrue(builder._should_use_candles("candles", days=1, aligned=False))
        self.assertFalse(builder._should_use_candles("trades", days=120, aligned=True))

    def test_align_with_kalshi_training_rows_preserves_order(self) -> None:
        base = datetime(2026, 7, 16, 18, 0, tzinfo=timezone.utc)
        trades = [
            {
                "timestamp": (base + timedelta(seconds=30)).isoformat().replace("+00:00", "Z"),
                "unix_time": int((base + timedelta(seconds=30)).timestamp()),
                "price": 64000.0,
                "trade_id": 1,
            },
            {
                "timestamp": (base + timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
                "unix_time": int((base + timedelta(minutes=2)).timestamp()),
                "price": 64200.0,
                "trade_id": 2,
            },
        ]
        kalshi_payload = {
            "training_rows": [
                {"timestamp": (base + timedelta(minutes=1)).isoformat()},
                {"timestamp": (base + timedelta(minutes=3)).isoformat()},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            kalshi_path = Path(tmpdir) / "kalshi.json"
            kalshi_path.write_text(json.dumps(kalshi_payload), encoding="utf-8")
            builder = FixtureCryptoPriceDatasetBuilder(trades)
            dataset = builder.build_dataset(
                crypto="ETH",
                days=1,
                align_with=str(kalshi_path),
            )

        self.assertEqual(len(dataset["y_rows"]), 2)
        self.assertEqual(dataset["y_rows"][0]["price"], 64000.0)
        self.assertEqual(dataset["y_rows"][1]["price"], 64200.0)


if __name__ == "__main__":
    unittest.main()

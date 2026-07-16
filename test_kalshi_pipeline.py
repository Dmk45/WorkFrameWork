import unittest
from datetime import datetime, timedelta, timezone

from kalshi_crypto_pipeline import DEFAULT_API_BASE_URL, KalshiHistoricalDataBuilder


class FixtureKalshiHistoricalDataBuilder(KalshiHistoricalDataBuilder):
    def __init__(self, payloads):
        super().__init__(api_base_url=DEFAULT_API_BASE_URL)
        self.payloads = payloads

    def _fetch_ticket_payloads(self, crypto, days, window_start, window_end):
        return self.payloads


class FixtureTradeFetchBuilder(KalshiHistoricalDataBuilder):
    def __init__(self, trades):
        super().__init__(api_base_url=DEFAULT_API_BASE_URL)
        self.trades = trades

    def _fetch_paginated_items(self, path, params=None, items_key=None, max_items=None):
        return self.trades


class KalshiPipelineExportTests(unittest.TestCase):
    def test_export_includes_full_trade_history_and_training_rows(self) -> None:
        now = datetime.now(timezone.utc)
        builder = FixtureKalshiHistoricalDataBuilder(
            [
                {
                    "ticket_id": "BTC-LIVE",
                    "event_ticker": "BTC",
                    "open_timestamp": (now - timedelta(days=2)).isoformat(),
                    "close_timestamp": now.isoformat(),
                    "trades": [
                        {
                            "timestamp": (now - timedelta(hours=2)).isoformat(),
                            "quantity": 1,
                            "position": "yes",
                            "action": "buy",
                            "probability": 0.52,
                        },
                        {
                            "timestamp": (now - timedelta(hours=1)).isoformat(),
                            "quantity": 2,
                            "position": "no",
                            "action": "sell",
                            "probability": 0.48,
                        },
                    ],
                }
            ]
        )

        dataset = builder.build_dataset(crypto="BTC", days=60, raw_only=False)

        self.assertIn("ticket_histories", dataset)
        self.assertEqual(len(dataset["ticket_histories"]), 1)
        first_ticket = dataset["ticket_histories"][0]
        self.assertIn("trades", first_ticket)
        self.assertEqual(len(first_ticket["trades"]), 2)
        self.assertEqual(len(dataset["training_rows"]), 2)
        self.assertIn("window_start", dataset)
        self.assertIn("window_end", dataset)

    def test_ticket_history_keeps_only_requested_lookback_window(self) -> None:
        builder = KalshiHistoricalDataBuilder(
            api_base_url=DEFAULT_API_BASE_URL,
        )
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(days=60)
        payload = {
            "ticket_id": "BTC-WINDOW",
            "event_ticker": "BTC",
            "open_timestamp": (window_start - timedelta(days=20)).isoformat(),
            "close_timestamp": window_end.isoformat(),
            "trades": [
                {
                    "timestamp": (window_start - timedelta(seconds=1)).isoformat(),
                    "quantity": 1,
                    "position": "yes",
                    "action": "buy",
                    "probability": 0.45,
                },
                {
                    "timestamp": (window_start + timedelta(days=1)).isoformat(),
                    "quantity": 2,
                    "position": "yes",
                    "action": "buy",
                    "probability": 0.55,
                },
                {
                    "timestamp": (window_end + timedelta(seconds=1)).isoformat(),
                    "quantity": 3,
                    "position": "yes",
                    "action": "buy",
                    "probability": 0.65,
                },
            ],
        }

        ticket = builder._build_ticket_history(payload, "BTC", window_start, window_end)

        self.assertEqual(len(ticket.trades), 1)
        self.assertGreaterEqual(ticket.trades[0].timestamp, window_start)
        self.assertLessEqual(ticket.trades[0].timestamp, window_end)

    def test_trade_fetch_keeps_only_matching_crypto_tickers(self) -> None:
        now = datetime.now(timezone.utc)
        builder = FixtureTradeFetchBuilder(
            [
                {
                    "trade_id": "btc-trade",
                    "ticker": "KXBTC15M-26JUL151345-45",
                    "timestamp": now.isoformat(),
                    "quantity": 1,
                    "position": "yes",
                    "action": "buy",
                    "probability": 0.52,
                },
                {
                    "trade_id": "soccer-trade",
                    "ticker": "KXWCGAME-26JUL15ENGARG-ENG",
                    "timestamp": now.isoformat(),
                    "quantity": 1,
                    "position": "yes",
                    "action": "buy",
                    "probability": 0.52,
                },
            ]
        )

        payloads = builder._fetch_ticket_payloads_from_trades(
            "BTC",
            now - timedelta(days=1),
            now + timedelta(seconds=1),
        )

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["ticket_id"], "KXBTC15M-26JUL151345-45")


if __name__ == "__main__":
    unittest.main()

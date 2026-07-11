import json
import os
import tempfile
import unittest

from kalshi_crypto_pipeline import build_dataset


class KalshiPipelineExportTests(unittest.TestCase):
    def test_export_includes_full_trade_history_and_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "dataset.json")
            dataset = build_dataset(
                crypto="BTC",
                days=60,
                output_path=output_path,
                api_key=None,
                api_base_url="https://example.test",
                use_offline=True,
                raw_only=False,
            )

            self.assertIn("ticket_histories", dataset)
            self.assertGreater(len(dataset["ticket_histories"]), 0)
            first_ticket = dataset["ticket_histories"][0]
            self.assertIn("trades", first_ticket)
            self.assertGreater(len(first_ticket["trades"]), 0)
            self.assertGreater(len(dataset["training_rows"]), 0)

            with open(output_path, "r", encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertIn("ticket_histories", persisted)
            self.assertIn("trades", persisted["ticket_histories"][0])


if __name__ == "__main__":
    unittest.main()

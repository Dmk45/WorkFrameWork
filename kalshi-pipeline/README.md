# Kalshi Pipeline

Standalone tools for collecting Kalshi market history, aligning BTC price targets,
and evaluating the resulting data. The Zig evaluator uses ModelWork2 as an
external framework dependency; it is not bundled in this repository.

## Python

Install the pipeline and test dependencies:

```powershell
python -m pip install -r requirements.txt
```

Generate a Kalshi export and align Coinbase price targets:

```powershell
python kalshi_crypto_pipeline.py --crypto BTC --days 120 --output kalshi_training_120d_dataset.json
python btc_price_dataset.py --crypto BTC --days 120 --align-with kalshi_training_120d_dataset.json --output btc_price_120d_dataset.json
```

Run the Python tests:

```powershell
python -m pytest
```

The PyTorch comparison script is optional and requires PyTorch:

```powershell
python -m pip install -r requirements-evaluation.txt
python crypto_model_pytorch.py
```

Generated datasets are intentionally excluded from Git. Create them locally
before running the model evaluation.

## Zig evaluation

The Zig parser and model evaluator are under `zig/` and import the separate
ModelWork2 framework. Clone that repository next to this checkout, then run:

```powershell
zig build test -Dframework-path=../WorkFrameWork
zig build parse -Dframework-path=../WorkFrameWork
zig build crypto-model -Dframework-path=../WorkFrameWork
```

The parser and model commands expect the generated dataset files in the current
working directory. Use Zig 0.13.0 to match the framework project.

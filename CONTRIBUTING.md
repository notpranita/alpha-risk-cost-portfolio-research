# Contributing / replication checklist

1. Fork or clone; install with `python -m pip install -e '.[dev]'`.
2. Run the complete suite with `python -m pytest -q --cov=acadian_arc`.
3. Run the CLI with `acadian-arc --months 42 --seed 17 --output-dir reports/verification`.
4. Do not commit private/vendor-licensed market data; preserve `examples/synthetic/` as fictional data.
5. Changes to optimization math must add deterministic tests of units, gradients, active constraints, portfolio turnover and net return accounting.
6. Any new real-data experiment must clearly mark sources, dates, permissible redistribution, feature publication lags, survivorship controls, and any hyperparameters tuned on validation rather than test data.
7. Separate synthetic demo metrics from observed real-world investment performance in all documentation.

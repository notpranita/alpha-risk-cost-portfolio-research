"""Run: python -m acadian_arc.cli --output-dir examples/synthetic"""
import argparse
import json
from pathlib import Path
import pandas as pd
from .data import generate_synthetic_panel
from .backtest import run_backtest
from .reporting import save_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent alpha-risk-cost research experiment")
    parser.add_argument("--input-csv", help="User-provided validated monthly panel; no downloading or scraping")
    parser.add_argument("--months", type=int, default=42)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2]/"configs"/"research.json"))
    parser.add_argument("--output-dir", default="reports/synthetic")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    settings = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if args.input_csv:
        panel = pd.read_csv(args.input_csv)
        print("USER-SUPPLIED PANEL: Verify historical universe, prices, lags and data licenses independently.")
    else:
        panel = generate_synthetic_panel(months=args.months, seed=args.seed)
        print("SYNTHETIC DEMONSTRATION — not real-world historical performance")
    result = run_backtest(panel, **settings)
    metadata = {"data": "user_supplied" if args.input_csv else "synthetic",
                "seed":None if args.input_csv else args.seed,
                "months":None if args.input_csv else args.months,
                "parameters":settings}
    output = save_report(result, args.output_dir, config=metadata)
    print(f"OOS periods: {result.summary['forecast_diagnostics']['n_oos_months']}")
    for strategy, met in result.summary["strategies"].items():
        print(f"{strategy:<17} synthetic/user-panel CAGR: {met['cagr']:.2%}; cost sum: {met['total_estimated_cost_fraction_summed']:.3%}")
    print(f"Saved {output.resolve()}")
    return result


if __name__ == "__main__":
    main()

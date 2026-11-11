"""Smoke test command-line wiring, output paths, and JSON metadata."""
import json
from acadian_arc.cli import main


def test_cli_generates_reproducible_artifacts(tmp_path):
    folder = tmp_path / 'example'
    result = main(['--months', '18', '--seed', '9', '--output-dir', str(folder)])
    assert result.summary['data_type'] == 'synthetic'
    assert (folder / 'summary.json').is_file()
    assert (folder / 'synthetic_wealth.png').stat().st_size > 0
    assert (folder / 'EXPERIMENT_REPORT.md').is_file()
    metadata = json.loads((folder / 'summary.json').read_text())
    assert metadata['config']['seed'] == 9
    assert metadata['config']['months'] == 18

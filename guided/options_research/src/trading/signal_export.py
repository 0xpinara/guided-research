"""Export trading signals for external use (e.g., QuantConnect)."""

import pandas as pd

from src.utils.logger import setup_logger
from src.utils.io_helpers import RESULTS_DIR

log = setup_logger(__name__)


def export_signals(
    dates, tickers, predictions, pred_direction,
    model_name: str,
) -> pd.DataFrame:
    """Export model signals in a standard format.

    Returns
    -------
    DataFrame with columns: date, ticker, predicted_return, signal (1/-1/0).
    """
    df = pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        "predicted_return": predictions,
        "signal": pred_direction,
    })

    out_dir = RESULTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"signals_{model_name}.csv"
    df.to_csv(path, index=False)
    log.info("Exported %d signals to %s", len(df), path)

    return df

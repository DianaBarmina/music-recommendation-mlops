from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset  # , DataQualityPreset

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reports_dir: str = "reports/",
    data_date: str = "",
    columns: list[str] | None = None,
) -> str:

    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    date_suffix = data_date.replace("-", "") if data_date else timestamp
    report_path = str(Path(reports_dir) / f"drift_report_{date_suffix}.html")

    if columns is None:
        columns = [
            c
            for c in reference_df.columns
            if reference_df[c].dtype in ["float64", "float32", "int64", "int32"]
            and c in current_df.columns
        ]

    if not columns:
        logger.warning("No columns for report")
        return ""

    try:
        definition = DataDefinition(numerical_columns=columns)
        ref_ds = Dataset.from_pandas(
            reference_df[columns].dropna(), data_definition=definition
        )
        cur_ds = Dataset.from_pandas(
            current_df[columns].dropna(), data_definition=definition
        )
        # report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=ref_ds, current_data=cur_ds)
        result.save_html(report_path)
        logger.info(f"Drift report saved: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return ""

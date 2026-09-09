import numpy as np
import pandas as pd

class DataFetchError(Exception):
    pass

def validate_and_clean_data(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict]:
    if df is None or df.empty:
        raise DataFetchError(f"Data fetch failed for ticker '{ticker}'")
    return df, {'status': 'SUCCESS'}

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import EMBEDDED_DIR

import pandas as pd
import numpy as np

df = pd.read_parquet(str(EMBEDDED_DIR / "embedded_26k_raw.parquet"))

print(df.columns)
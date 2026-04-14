import os
import pandas as pd
import torch

def save(df: pd.DataFrame, path: str = "final/data.parquet") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Saved to {path} (Parquet format)")


def load_df(path: str = "processed/rdf_results_final.parquet") -> pd.DataFrame:
    print("Loading df")
    return pd.read_parquet(path)



def _get_device() -> str:
    """Detect the best available device for local embeddings."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"⚡ Using CUDA GPU: {device_name}")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("🍎 Using Apple MPS GPU")
        return "mps"
    else:
        print("💻 No GPU found — using CPU")
        return "cpu"


M2M_LANG_MAP = {
    "ja": "ja",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hk": "zh",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "ru": "ru",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
    "ar": "ar",
    "hi": "hi",
    "en": "en",
}

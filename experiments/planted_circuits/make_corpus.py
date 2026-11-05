"""Rebuild corpus.txt: wikitext-103 validation paragraphs longer than 200 characters (1.1 MB,
~247k Qwen3.5 tokens). Every script slices it by fixed token offsets, so results depend on it."""
import pandas as pd
from huggingface_hub import hf_hub_download

p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/validation-00000-of-00001.parquet", repo_type="dataset")
txt = [t for t in pd.read_parquet(p).text if len(t) > 200]
open("corpus.txt", "w").write("\n".join(txt))
print(len(txt), "paragraphs")

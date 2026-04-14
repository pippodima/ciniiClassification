import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama
import pandas as pd
from utils import load_df, save
from config import CLUSTER_METADATA, MERGE_METADATA, MERGE_TOPICS_JSON, MERGE_MAIN_TOPICS
import re
from tqdm import tqdm
import json

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", None)

def clean_response(text):
    # Remove <think> ... </think> blocks (multi-line safe)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# -----------------------------
# Function to generate subtopic
# -----------------------------
def generate_subtopic(keywords):
    prompt = (
        "Given this list of keywords: "
        f"{', '.join(keywords)}\n\n"
        "Create a short, meaningful subtopic title (2–5 words) "
        "that best summarizes the topic.\n"
        "Do NOT output <think> tags or chain-of-thought."
    )

    response = ollama.chat(
        model="qwen3:1.7b",  # Change model name to whichever you use in Ollama
        messages=[{"role": "user", "content": prompt}]
    )

    content = response["message"]["content"]
    cleaned = clean_response(content)
    return cleaned
 
def generate_main_topic(all_keywords):
    prompt = (
        "Given this list of keywords:\n"
        f"{', '.join(all_keywords)}\n\n"
        "Create a short, meaningful MAIN TOPIC title (2–4 words) that "
        "summarizes the entire topic. Avoid generic labels. "
    )

    response = ollama.chat(
        model="qwen3:1.7b",  # same as your subtopic script
        messages=[{"role": "user", "content": prompt}]
    )

    content = response["message"]["content"]
    return clean_response(content)




def main():
    df = load_df(str(CLUSTER_METADATA))
    tqdm.pandas(desc="Generating subtopics")
    df["subtopic"] = df["keywords"].progress_apply(generate_subtopic)

    save(df, str(MERGE_METADATA))

    with open(str(MERGE_TOPICS_JSON), "r") as f:
        merged_clusters = json.load(f)
    
    name_to_keywords = dict(zip(df["name"], df["keywords"]))

    rows = []
    for big_cluster_id, subcluster_names in merged_clusters.items():

        # Collect keywords from all subclusters
        collected_keywords = []

        for name in subcluster_names:
            if name in name_to_keywords:
                collected_keywords.extend(name_to_keywords[name])
            else:
                print(f"WARNING: subcluster '{name}' not found in metadata")

        # Remove duplicates while preserving order
        collected_keywords = list(dict.fromkeys(collected_keywords))

        rows.append({
            "merged_cluster_id": int(big_cluster_id),
            "subclusters": subcluster_names,
            "keywords": collected_keywords
        })

    merged_df = pd.DataFrame(rows)

    tqdm.pandas(desc="Generating main topics")
    merged_df["main_topic"] = merged_df["keywords"].progress_apply(generate_main_topic)

    save(merged_df, str(MERGE_MAIN_TOPICS))
    print(merged_df.head(10))



if __name__=="__main__":
    main()
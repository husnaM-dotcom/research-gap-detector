import re
import pandas as pd
import spacy


nlp = spacy.load("en_core_web_sm")

# Alias normalizer: abbreviations & full names merge into one clean label
NORMALIZE_MAP = {
    "llm": "Large Language Model", "llms": "Large Language Model",
    "large language models": "Large Language Model",
    "cnn": "CNN", "convolutional neural network": "CNN",
    "convolutional neural networks": "CNN",
    "gnn": "Graph Neural Network", "graph neural network": "Graph Neural Network",
    "rl": "Reinforcement Learning", "fl": "Federated Learning",
    "ehr": "Electronic Health Records", "ehrs": "Electronic Health Records",
    "electronic health record": "Electronic Health Records",
}

# Head nouns that signal a technical Method/Approach 
METHOD_HEADS = {
    "method", "methods", "model", "models", "network", "networks",
    "algorithm", "algorithms", "framework", "approach", "technique",
    "architecture", "mechanism", "learning", "transformer", "transformers",
    "clustering", "regression", "classification", "analysis", "simulation",
    "screening", "sequencing", "optimization", "pipeline",
}

# Head nouns that signal a Dataset, Domain, or Target Data Type
DATASET_HEADS = {
    "dataset", "datasets", "data", "benchmark", "benchmarks", "corpus",
    "records", "imagery", "images", "signals", "scans", "patients",
    "cohort", "samples", "sequences", "series", "logs", "database",
}

LIMITATION_KEYWORDS = [
    "limitation", "future work", "however", "drawback",
    "challenge", "lack of", "remains open", "bottleneck",
]


def _clean_phrase(phrase: str) -> str:
    """Strips leading articles/pronouns and normalizes common acronyms."""
    cleaned = re.sub(r"^(a|an|the|this|these|our|their|such|novel|proposed)\s+", "", phrase.lower()).strip()
    if cleaned in NORMALIZE_MAP:
        return NORMALIZE_MAP[cleaned]
    return cleaned.title() if len(cleaned) > 4 else cleaned.upper()


def extract_entities_and_limitations(df: pd.DataFrame):
    """Automatically discovers Methods, Datasets, and Limitations using spaCy NLP."""
    paper_methods, paper_datasets, limitations = [], [], []

    for _, row in df.iterrows():
        doc = nlp(row["abstract"])
        found_m, found_d = set(), set()

        # 1. Grammatical Noun-Chunk Discovery 
        for chunk in doc.noun_chunks:
            root_word = chunk.root.lemma_.lower()
            words = [t.text for t in chunk if not t.is_stop and not t.is_punct]
            if not (1 <= len(words) <= 4):
                continue

            phrase = _clean_phrase(" ".join(words))
            if len(phrase) < 2:
                continue

            if root_word in METHOD_HEADS:
                found_m.add(phrase)
            elif root_word in DATASET_HEADS:
                found_d.add(phrase)

        # 2. Catch uppercase dataset/model acronyms 
        for token in doc:
            if re.match(r"^[A-Z]{2,}[-\d]*$", token.text) and token.text not in {"AI", "ML", "API", "GPU", "CPU"}:
                if any(char.isdigit() for char in token.text):
                    found_d.add(token.text)

        paper_methods.append(sorted(found_m))
        paper_datasets.append(sorted(found_d))

        # 3. Extract Limitation / Future-Work sentences using spaCy 
        for sent in doc.sents:
            s_text = sent.text.strip()
            if any(kw in s_text.lower() for kw in LIMITATION_KEYWORDS):
                limitations.append({
                    "title": row["title"],
                    "year": row["year"],
                    "url": row["url"],
                    "sentence": s_text,
                })

    df["methods"] = paper_methods
    df["datasets"] = paper_datasets
    return df, pd.DataFrame(limitations)


if __name__ == "__main__":
    from fetcher import fetch_papers

    sample_df = fetch_papers("Federated Learning Healthcare", max_results=15)
    enriched_df, lim_df = extract_entities_and_limitations(sample_df)

    print("Discovered Methods & Datasets:")
    print(enriched_df[["arxiv_id", "methods", "datasets"]].head(10))
    print(f"\nFound {len(lim_df)} limitation/challenge sentences!")

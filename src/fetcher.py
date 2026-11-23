import arxiv
import pandas as pd


def fetch_papers(query: str, max_results: int = 10) -> pd.DataFrame:
    """Fetches research papers from arXiv for a given topic query."""
    client = arxiv.Client(page_size=100, delay_seconds=1.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    records = []
    for result in client.results(search):
        records.append({
            "arxiv_id": result.get_short_id(),
            "title": result.title.replace("\n", " ").strip(),
            "abstract": result.summary.replace("\n", " ").strip(),
            "year": result.published.year,
            "url": result.entry_id,
        })

    df = (
        pd.DataFrame(records)
        .drop_duplicates(subset=["title"])
        .reset_index(drop=True)
    )
    return df


if __name__ == "__main__":
    sample_df = fetch_papers("Federated Learning Healthcare", max_results=10)
    print(f"Fetched {len(sample_df)} papers!")
    print(sample_df[["arxiv_id", "year", "title"]].head())

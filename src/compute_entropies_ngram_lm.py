"""Use the model trained ngram language models\
    to compute the entropies on Providence data."""
from pathlib import Path
from pandas import DataFrame
from math import exp

def get_entropies(entropies_file: str) -> dict:
    """Reads the entopies of utterances into dictionary."""
    entropies_df = []
    with open(entropies_file) as entropies:
        for line in entropies:
            utterance_id, entropy = line.strip().split("\t")
            entropy = float(entropy)
            entropies_df.append({
                "utterance_id": utterance_id,
                "perplexity": exp(entropy),
                "entropy": entropy
            })
    return DataFrame(entropies_df)

def main():
    entropies = get_entropies("data/Providence/model_inputs/Providence.entropies")
    output_folder = Path("results")
    output_folder.mkdir(exist_ok=True, parents=True)
    entropies.to_csv("results/Librispeech_360h.csv")

if __name__ == "__main__":
    main()
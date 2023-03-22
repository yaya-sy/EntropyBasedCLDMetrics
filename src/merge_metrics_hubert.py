# hubert_nat = pd.read_csv("results/HuBERT-nat_entropy_ngram-2-merge-False_mmap.csv")
# hubert_tts = pd.read_csv("results/HuBERT-tts_entropy_ngram-2-merge-False_mmap.csv")

# hubert_nat.loc[(hubert_nat["segment_speaker"] == "CHI"), "segment_speaker"] = "Target_Child"
# hubert_nat.loc[(hubert_nat["segment_speaker"] == "FEM"), "segment_speaker"] = "Mother"
# hubert_tts.loc[(hubert_tts["segment_speaker"] == "CHI"), "segment_speaker"] = "Target_Child"
# hubert_tts.loc[(hubert_tts["segment_speaker"] == "FEM"), "segment_speaker"] = "Mother"

# hubert_nat_ages = []
# hubert_tts_ages = []

# hubert_nat_families = []
# hubert_tts_families = []

# hubert_nat_groups = hubert_nat.sort_values(by=['file_name'])
# hubert_nat_groups = hubert_nat_groups.groupby("file_name").groups
# hubert_tts_groups = hubert_tts.sort_values(by=['file_name'])
# hubert_tts_groups = hubert_tts_groups.groupby("file_name").groups

# for group in tqdm(hubert_nat_groups):
#     filename = group
#     child, filename = str(Path(filename).stem).split("-")
#     filename += ".cha"
#     filename = base_path / child / filename
#     cha = pylangacq.read_chat(str(filename))
#     age = cha.ages(months=True)[0]
#     if age == 0.0:
#         continue
#     hubert_nat_ages.extend([age] * len(hubert_nat_groups[group]))
#     hubert_nat_families.extend([child] * len(hubert_nat_groups[group]))
# hubert_nat["age"] = hubert_nat_ages
# hubert_nat["family"] = hubert_nat_families

# for group in tqdm(hubert_tts_groups):
#     filename = Path(group)
#     child, filename = filename.parent.stem, filename.stem
#     filename += ".cha"
#     filename = base_path / child / filename
#     cha = pylangacq.read_chat(str(filename))
#     age = cha.ages(months=True)[0]
#     if age == 0.0:
#         continue
#     hubert_tts_ages.extend([age] * len(hubert_tts_groups[group]))
#     hubert_tts_families.extend([child] * len(hubert_tts_groups[group]))
# hubert_tts["age"] = hubert_tts_ages
# hubert_tts["family"] = hubert_tts_families

# hubert_nat = hubert_nat.rename(columns={"segment_speaker": "speaker"})
# hubert_tts = hubert_tts.rename(columns={"segment_speaker": "speaker"})

# hubert_tts = hubert_tts[hubert_tts['speaker'].map(lambda x: x in {"Mother", "Target_Child"})]
# hubert_nat = hubert_nat[hubert_nat['speaker'].map(lambda x: x in {"Mother", "Target_Child"})]

# hubert_nat = hubert_nat.groupby(["family", "speaker", "age"]).mean()
# hubert_nat.drop("Unnamed: 0", axis=1, inplace=True)
# hubert_tts = hubert_tts.groupby(["family", "speaker", "age"]).mean()
# hubert_tts.drop("Unnamed: 0", axis=1, inplace=True)

# hubert_nat.to_csv("results/HuBERT_nat.csv")
# hubert_tts.to_csv("results/HuBERT_tts.csv")

# hubert_nat = pd.read_csv("results/HuBERT_nat.csv")
# hubert_nat = hubert_nat.loc[hubert_nat["speaker"] == "Target_Child"]

# hubert_tts = pd.read_csv("results/HuBERT_tts.csv")
# hubert_tts = hubert_tts.loc[hubert_tts["speaker"] == "Target_Child"]

# hubert_nat = hubert_nat.merge(metrics_csv, on = ['family', 'age'])
# hubert_nat.to_csv("results/HuBERT_nat_metrics.csv")

# hubert_tts = hubert_tts.merge(metrics_csv, on = ['family', 'age'])
# hubert_tts.to_csv("results/HuBERT_tts_metrics.csv")
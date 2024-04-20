import pandas as pd
import json


def split_seas(out_path, users=5, training_num=8, sex="male"):
    """
    Splits voice data in training, validation, and test data
    Args:
        listing_df (pandas.DataFrame): _description_
        users (int): number of users to get samples
        training_num (int): nuber of samples to give for training
    """
    listing_df = pd.read_csv(r"./dataset/validated.tsv", sep="\t")
    if sex == "male":
        listing_df = listing_df[listing_df["gender"] == "male_masculine"]
    else:
        listing_df = listing_df[listing_df["gender"] == "female_feminine"]

    values_counts_df = pd.DataFrame(listing_df["client_id"].value_counts())
    distribution = {}
    for i in values_counts_df.iterrows():
        if i[1][0] == int(training_num + training_num * (2 / 8)) and users > 0:
            distribution[i[0]] = 0
            users -= 1

    for i in distribution.keys():
        lst = list(listing_df.loc[listing_df["client_id"] == i]["path"])
        st = int(training_num + training_num * (1 / 8))
        distribution[i] = {
            "train": lst[:training_num],
            "validation": lst[training_num:st],
            "test": lst[st:],
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(distribution, f)


split_seas("src/female.json", users=5, sex="female")
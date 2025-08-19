import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from config import Config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_and_split_data():
    df = pd.read_json(Config.DATA_FILE)

    print(f"[load_and_split_data] original data shape: {df.shape}")
    print(f"[load_and_split_data] sample data:\n{df.head()}")

    df["question"] = df["question"].str.strip()
    df["action"] = df["action"].str.strip().str.lower()

    print(f"[load_and_split_data] data after cleaning sample:\n{df.head()}")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["action"], random_state=Config.SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["action"], random_state=Config.SEED
    )

    print(f"[load_and_split_data] train size: {len(train_df)}, val size: {len(val_df)}, test size: {len(test_df)}")

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True))
    })

    return dataset

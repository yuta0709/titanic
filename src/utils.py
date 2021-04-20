from typing import List
import pandas as pd

Vector = List[float]


def split_train(df: pd.DataFrame) -> (Vector, Vector):
    y_train = df["Survived"].values
    x_train = df.drop(["PassengerId", "Survived"], axis=1).values
    return x_train, y_train

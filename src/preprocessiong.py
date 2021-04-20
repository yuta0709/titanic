import pandas as pd


def process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("Age", axis=1)
    df = df.drop("Cabin", axis=1)
    df = df.drop("Name", axis=1)
    df = df.drop("Ticket", axis=1)
    df = dummy(df, "Embarked")
    df = dummy(df, "Sex")
    return df


def dummy(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df_dummy = pd.get_dummies(df[label])
    df2 = pd.concat([df.drop(label, axis=1), df_dummy], axis=1)
    return df2

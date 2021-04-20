import pandas as pd
import tensorflow
import os
import numpy as np
import utils
import preprocessiong
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    df = pd.read_csv("../dataset/train.csv")
    test_df = pd.read_csv("../dataset/test.csv")

    df = preprocessiong.process(df)
    test_df = preprocessiong.process(test_df)

    passenger_id = test_df["PassengerId"]

    test = test_df.drop("PassengerId", axis=1).values
    (x_train, y_train) = utils.split_train(df)

    model = get_model(len(x_train[0]))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, verbose=1, batch_size=128)

    ans = model.predict(test, verbose=0)
    ans = np.round(ans).astype(int)
    ans = ans.flatten()
    my_solution = pd.DataFrame(ans, passenger_id, columns=["Survived"])
    my_solution.to_csv("../data/answer.csv", index_label=["PassengerId"])


def get_model(input_size: int) -> tensorflow.keras.models.Model:
    model = Sequential()
    model.add(Dense(28, activation="relu", input_shape=(input_size,)))
    model.add(Dropout(0.1))
    model.add(Dense(56, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(56, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    return model


if __name__ == "__main__":
    main()

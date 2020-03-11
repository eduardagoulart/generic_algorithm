import pandas as pd
import numpy as np
import random
import math

BITS = 6
SIZE_OF_COMMUNITY = 30
MIN_VALUE = -2
MAX_VALUE = 2


def create_random_specimen():
    return [1 if random.random() > 0.5 else 0 for _ in range(BITS)]


def convert_int_to_bin(bin):
    value = 0
    for i in range(BITS):
        value += 2 ** i * bin[i]
    return value


def selection_method(df):
    def specimen_counter(row):
        counter = 0
        for gen in range(len(row)):
            counter += np.cos(2 * math.pi * row[gen])
        print(len(row))
        return (
            -20 * math.exp(-0.2 * math.sqrt(1 / len(row) * sum(np.power(row, 2))))
            - math.exp(1 / len(row) * counter)
            + 20
            + math.exp(1)
        )

    real_values = {"id": [], "fitness": []}
    for key, gen_data in df.iterrows():
        real_values["id"].append(key)
        real_values["fitness"].append(
            specimen_counter([gen_data.real_value_gen_1, gen_data.real_value_gen_0])
        )

    real_values = pd.DataFrame(real_values)
    return df.merge(real_values, on="id", how="left")


if __name__ == "__main__":
    community = {"id": [], "gen_0": [], "gen_1": []}

    for specimen in range(SIZE_OF_COMMUNITY):
        community["id"].append(specimen)
        community["gen_0"].append(create_random_specimen())
        community["gen_1"].append(create_random_specimen())

    df = pd.DataFrame(community)
    df["real_value_gen_0"] = df.gen_0.apply(
        lambda row: MIN_VALUE
        + ((MAX_VALUE - MIN_VALUE) / (2 ** BITS - 1)) * convert_int_to_bin(row)
    )
    print(df)

    df["real_value_gen_1"] = df.gen_1.apply(
        lambda row: MIN_VALUE
        + ((MAX_VALUE - MIN_VALUE) / (2 ** BITS - 1)) * convert_int_to_bin(row)
    )

    selection_method(df)

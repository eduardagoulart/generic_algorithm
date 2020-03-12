import pandas as pd
import numpy as np
import random
import math

BITS = 6
SIZE_OF_COMMUNITY = 30
MIN_VALUE = -2
MAX_VALUE = 2
NUMBER_OF_PARENTS = 2


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
        return (
            -20
            * math.exp(-0.2 * math.sqrt(1 / float(len(row)) * sum(np.power(row, 2))))
            - math.exp(1 / float(len(row)) * counter)
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


def tournament(df):
    sample_parents = df.sample(2)
    winner = sample_parents.nlargest(1, "fitness")
    couples = df[df.id != int(winner.id)]
    return winner, couples


if __name__ == "__main__":
    community = {"id": [], "gen_0": [], "gen_1": []}

    for specimen in range(SIZE_OF_COMMUNITY):
        community["id"].append(specimen)
        community["gen_0"].append(create_random_specimen())
        community["gen_1"].append(create_random_specimen())

    df = pd.DataFrame(community)
    df["real_value_gen_0"] = df.apply(
        lambda row: (
            MIN_VALUE
            + (float(MAX_VALUE - MIN_VALUE) / (2 ** BITS - 1))
            * convert_int_to_bin(row.gen_0)
        ),
        axis=1,
    )

    df["real_value_gen_1"] = df.apply(
        lambda row: (
            MIN_VALUE
            + (float(MAX_VALUE - MIN_VALUE) / (2 ** BITS - 1))
            * convert_int_to_bin(row.gen_1)
        ),
        axis=1,
    )

    df = selection_method(df)
    couples = df.copy()
    first_parent, couples = tournament(couples)
    second_parent, couples = tournament(couples)

    parents = first_parent.append(first_parent)

import pandas as pd
import numpy as np
import random
import math


BITS = 6
SIZE_OF_COMMUNITY = 30
MIN_VALUE = -2
MAX_VALUE = 2
NUMBER_OF_PARENTS = 2
GENERATION = 100
MUTATION_PROB = 0.2
CROSSOVER_PROB = 0.7


def create_random_specimen():
    return [1 if random.random() > 0.5 else 0 for _ in range(BITS)]


def convert_int_to_bin(bin):
    value = 0
    for i in range(BITS):
        value += 2 ** i * bin[i]
    return value


def real_gen_0(row):
    return MIN_VALUE + (
        float(MAX_VALUE - MIN_VALUE) / (2 ** BITS - 1)
    ) * convert_int_to_bin(row.gen_0)


def real_gen_1(row):
    return MIN_VALUE + (
        float(MAX_VALUE - MIN_VALUE) / (2 ** BITS - 1)
    ) * convert_int_to_bin(row.gen_1)


def selection_method(df):
    def _specimen_counter(row):
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
            _specimen_counter([gen_data.real_value_gen_1, gen_data.real_value_gen_0])
        )

    real_values = pd.DataFrame(real_values)
    return df.merge(real_values, on="id", how="left")


def tournament(df):
    sample_parents = df.sample(2)
    winner = sample_parents.nlargest(1, "fitness")
    couples = df[df.id != int(winner.id)]
    return winner, couples


def crossover(df, last_id):
    child_1 = pd.DataFrame()
    child_2 = pd.DataFrame()
    child_1["gen_0"] = [df.iloc[0]["gen_0"][0:4] + df.iloc[1]["gen_0"][-2:]]
    child_1["gen_1"] = [df.iloc[0]["gen_1"][0:4] + df.iloc[1]["gen_1"][-2:]]
    child_2["gen_0"] = [df.iloc[1]["gen_0"][0:4] + df.iloc[0]["gen_0"][-2:]]
    child_2["gen_1"] = [df.iloc[1]["gen_1"][0:4] + df.iloc[0]["gen_1"][-2:]]
    child_1["id"] = last_id + 1
    child_2["id"] = last_id + 2
    child_1["real_value_gen_0"] = child_1.apply(real_gen_0, axis=1)
    child_1["real_value_gen_1"] = child_1.apply(real_gen_1, axis=1)
    child_2["real_value_gen_0"] = child_2.apply(real_gen_0, axis=1)
    child_2["real_value_gen_1"] = child_2.apply(real_gen_1, axis=1)
    children = child_1.append(child_2)
    children = children.set_index("id")
    children = selection_method(children)
    children = children.reset_index()
    return children.nlargest(1, "fitness")


def mutation(df, gen, pos):
    temp = list(df[gen])
    if temp[0][pos] == 1:
        temp[0][pos] = 0
    else:
        temp[0][pos] = 1
    df[gen] = temp
    return df[gen]


def elitism(df, child):
    remove = df.loc[df.fitness == min(df["fitness"])]
    if len(remove) > 1:
        remove = remove.head(1)
    if float(remove["fitness"]) < float(child["fitness"]):
        df = df.loc[df["id"] != int(remove["id"])]
        df = df.append(child)
    return df


def main(df, last_id):
    for i in range(GENERATION):
        couples = df.copy()
        pc = random.random()
        if pc <= CROSSOVER_PROB:
            pass
        first_parent, couples = tournament(couples)
        second_parent, couples = tournament(couples)
        parents = first_parent.append(second_parent)

        children = crossover(parents, last_id)
        children.drop(columns=["fitness"], inplace=True)
        last_id += 2
        for i in range(BITS):
            prob = random.random()
            if prob <= MUTATION_PROB:
                children["gen_0"] = mutation(children, "gen_0", i)
            prob = random.random()
            if prob <= MUTATION_PROB:
                children["gen_1"] = mutation(children, "gen_1", i)

        children["real_value_gen_0"] = children.apply(real_gen_0, axis=1)
        children["real_value_gen_1"] = children.apply(real_gen_1, axis=1)
        children = children.set_index("id")
        children = selection_method(children)
        df = elitism(df, children)
        print(df)
        print(f"--------{i}---------")
    df.drop(columns=["index"], inplace=True)
    print(f"final:\n{df}")
    print("END")


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
    main(df, last_id=29)
    # while boolean:
    #     main(df)

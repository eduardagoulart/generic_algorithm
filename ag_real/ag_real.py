import argparse
import math
import numpy as np
import pandas as pd
import random

DIMENSION = 6


def func_obj(row):
    f_exp = -0.2 * math.sqrt(1 / DIMENSION * sum(np.power(row.realRepresentation, 2)))
    t = 0
    for i in range(DIMENSION):
        t += np.cos(2 * math.pi * row.realRepresentation[i])
    return -20 * math.exp(f_exp) - math.exp(1 / DIMENSION * t) + 20 + math.exp(1)


def create_single_unit():
    """
    This method creates a subject based on the dimension information
    ---
    Return:
        list: with the random value for that gen
    """
    subject = [random.uniform(-2, 2) for _ in range(DIMENSION)]
    return subject


def create_random_comunity(population_size, gen):
    """
    Create random community based on the desired size
    ---
    Return:
        Pandas DataFrame
    """
    subject_info = {"id": [], "generation": [], "realRepresentation": [], "fitness": []}
    for idx in range(population_size):
        subject_info["id"].append(idx)
        subject_info["generation"].append(gen)
        subject_info["realRepresentation"].append(create_single_unit())
        subject_info["fitness"].append(0)
    return pd.DataFrame(subject_info)


def create_roulette(parents_data):
    parents_data["finalFitness"] = 0
    parents_data.loc[parents_data["fitness"] == 0, ["finalFitness"]] = 1
    parents_data.loc[parents_data["fitness"] > 0, ["finalFitness"]] = (
        1 / parents_data["fitness"]
    )
    return parents_data["finalFitness"] / parents_data["finalFitness"].sum()


def get_parents(community, population_size):
    community["finalFitness"] = create_roulette(community)
    choosen_parents = []
    roulette_sum = 0
    for _ in range(population_size):
        random_limit = random.random()
        for index, subject in community.iterrows():
            roulette_sum += subject["finalFitness"]
            if roulette_sum >= random_limit:
                roulette_sum = 0
                choosen_parents.append(subject)
                break
    return choosen_parents


def main():
    parser = argparse.ArgumentParser(
        description="Creates Non Prepared as well as Prepared DFs and uploads CSV caches to s3."
    )
    parser.add_argument(
        "-npop",
        "--pop",
        nargs="+",
        help="The necessary size of the community",
        required=False,
        default=10,
    )
    parser.add_argument(
        "-sg",
        "--sg",
        nargs="+",
        help="start gen (the value for the first generation)",
        required=False,
        default=0,
    )
    parser.add_argument(
        "-ngen",
        "--ngen",
        nargs="+",
        help="number of how many gens we want to look for",
        required=False,
        default=10,
    )

    args = parser.parse_args()
    generation = args.sg
    population_size = args.pop
    community = create_random_comunity(population_size, generation)
    community["fitness"] = community[["realRepresentation", "fitness"]].apply(
        func_obj, axis=1
    )
    # import pudb
    # pudb.set_trace()
    best_fitness = [min(community["fitness"])]
    avg_fitness = [community["fitness"].to_list()]
    gen_fitness = []
    generation += 1
    # import pudb
    # pudb.set_trace()
    for _ in range(1, args.ngen):
        random_parents = get_parents(community, population_size)


if __name__ == "__main__":
    main()

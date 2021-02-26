import argparse
import ast
import math
import numpy as np
import pandas as pd
import random

DIMENSION = 6
ALPHA = 0.75
BETA = 0.25


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
    """
    Args:
        community: DataFrame with the information from the base community
        population_size: Integer with the community size

    Returns:
        List of DataFrames with the better parents possible
        (the same parent can show more than once in the list)
    """
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


def get_first_parents(parents):
    """
    The idea is select the parents to mutation.
    This method will return the first two parents based on a random choice
    and considering parent X and Y are not the same subject
    Args:
        parents: DataFrame

    Returns:
        DataFrame with only two rows
    """
    parents_df = pd.DataFrame.from_dict(map(dict, parents))
    parents_df = parents_df.drop_duplicates(subset=["id"])
    return parents_df.head(2)


def genetic_operators_blx_alpha(real_representation, parent_x, parent_y):
    """
    This method calculates the random operator between a range
    (min(float(parent_x[i]), float(parent_y[i])) - alpha * real_representation[i],
    max(float(parent_x[i]), float(parent_y[i])) - alpha * real_representation[i])
    Note that this only uses the alpha value to calculate
    Args:
        real_representation: List with the DIMENSION size
        parent_x: pandas DataFrame with one row
        parent_y: pandas DataFrame with one row

    Returns:
        Float value with the information to be added in the realRepresentation for
        the children
    """
    min_value = min(
        [
            min(float(parent_x[i]), float(parent_y[i])) - ALPHA * real_representation[i]
            for i in range(DIMENSION)
        ]
    )

    max_value = max(
        [
            max(float(parent_x[i]), float(parent_y[i])) - ALPHA * real_representation[i]
            for i in range(DIMENSION)
        ]
    )
    return random.uniform(min_value, max_value)


def cross_validation_blx_alpha(community, parents, gen):
    parents = get_first_parents(parents)
    parent_x = parents["realRepresentation"][0]
    parent_y = parents["realRepresentation"][1]

    real_representation = list(np.array(parent_x) - np.array(parent_y))
    idx = max(community.id) + 1

    def _generate_kids_info(idx):
        kids_info = {
            "id": [],
            "generation": [],
            "realRepresentation": [],
            "fitness": [],
        }
        kids_info["id"].append(idx)
        kids_info["generation"].append(gen)
        kids_info["fitness"].append(0)
        for i in range(DIMENSION):
            kids_info["realRepresentation"].append(
                genetic_operators_blx_alpha(real_representation, parent_x, parent_y)
            )
        kids_info["realRepresentation"] = str(kids_info["realRepresentation"])
        return kids_info

    kid_x = pd.DataFrame(_generate_kids_info(idx))
    kid_y = pd.DataFrame(_generate_kids_info(max(kid_x.id) + 1))
    return kid_x, kid_y


def cross_validation_blx_alpha_beta(community, parents, gen):
    parents = get_first_parents(parents)
    if float(parents["finalFitness"][0]) <= float(parents["finalFitness"][0]):
        parent_x = parents["realRepresentation"][0]
        parent_y = parents["realRepresentation"][1]
    else:
        parent_y = parents["realRepresentation"][0]
        parent_x = parents["realRepresentation"][1]
    real_representation = list(np.array(parent_x) - np.array(parent_y))
    idx = max(community.id) + 1

    def _generate_kids_info(idx):
        kids_info = {
            "id": [],
            "generation": [],
            "realRepresentation": [],
            "fitness": [],
        }
        kids_info["id"].append(idx)
        kids_info["generation"].append(gen)
        kids_info["fitness"].append(0)
        for i in range(DIMENSION):
            if parent_x[i] <= parent_y[i]:
                kids_info["realRepresentation"].append(
                    random.uniform(
                        parent_x[i] - ALPHA * real_representation[i],
                        parent_y[i] + BETA * real_representation[i],
                    )
                )
            else:
                kids_info["realRepresentation"].append(
                    random.uniform(
                        parent_y[i] - BETA * real_representation[i],
                        parent_x[i] + ALPHA * real_representation[i],
                    )
                )

        kids_info["realRepresentation"] = str(kids_info["realRepresentation"])
        return kids_info

    kid_x = pd.DataFrame(_generate_kids_info(idx))
    kid_y = pd.DataFrame(_generate_kids_info(max(kid_x.id) + 1))
    return kid_x, kid_y


def make_mutation(mutation_taxes, kid):
    """
    The mutation operation will be made by choosing one random index
    in the children realRepresentation that should be changed
    Args:
        mutation_taxes: Float with the mutation probability
        kid: DataFrame with one children information
    Returns:
        DataFrame with one children information
    """
    for i in range(DIMENSION):
        mutation_probability = random.random()
        if mutation_probability <= mutation_taxes:
            mutation_index = random.randint(0, DIMENSION - 1)
            list_real_representation = ast.literal_eval(
                kid["realRepresentation"].to_list()[0]
            )
            list_real_representation[mutation_index] = random.random()
            kid["realRepresentation"] = str(list_real_representation)
            return kid
    return kid


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
    parser.add_argument(
        "-mutation",
        "--mutation",
        nargs="+",
        help="The mutation taxes (between 0 and 1 - percentage)",
        required=False,
        default=1,
    )

    args = parser.parse_args()
    generation = args.sg
    population_size = args.pop
    mutation_taxes = args.mutation
    community = create_random_comunity(population_size, generation)
    community["fitness"] = community[["realRepresentation", "fitness"]].apply(
        func_obj, axis=1
    )

    best_fitness = [min(community["fitness"])]
    avg_fitness = [community["fitness"].to_list()]
    gen_fitness = []
    generation += 1

    for _ in range(1, args.ngen):
        random_parents = get_parents(community, population_size)
        cross_validation_blx_alpha(community, random_parents, population_size)
        kid_x, kid_y = cross_validation_blx_alpha_beta(
            community, random_parents, population_size
        )
        kid_x = make_mutation(mutation_taxes, kid_x)
        kid_y = make_mutation(mutation_taxes, kid_y)
        generation += 1
        print(random_parents)


if __name__ == "__main__":
    main()

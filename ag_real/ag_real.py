import ast
import math
import numpy as np
import pandas as pd
import random

DIMENSION = 6
ALPHA = 0.75
BETA = 0.25


def func_obj(row):
    if type(row.realRepresentation) == str:
        row.realRepresentation = ast.literal_eval(row.realRepresentation)
    f_exp = -0.2 * math.sqrt(
        1 / DIMENSION * sum(np.power(row.realRepresentation, 2))
    )
    t = 0
    for i in range(DIMENSION):
        t += np.cos(2 * math.pi * row.realRepresentation[i])
    return (
        -20 * math.exp(f_exp) - math.exp(1 / DIMENSION * t) + 20 + math.exp(1)
    )


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
    subject_info = {
        "id": [],
        "generation": [],
        "realRepresentation": [],
        "fitness": [],
    }
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
    This method will return the first two parents based
    on a random choice and considering parent X and Y
    are not the same subject
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
    (
        min(
            float(parent_x[i]),
            float(parent_y[i])
        ) - alpha * real_representation[i],
        max(
            float(parent_x[i]),
            float(parent_y[i])
        ) - alpha * real_representation[i]
    )
    Note that this only uses the alpha value to calculate
    Args:
        real_representation: List with the DIMENSION size
        parent_x: pandas DataFrame with one row
        parent_y: pandas DataFrame with one row

    Returns:
        Float value with the information to be added in the
        realRepresentation for the children
    """
    min_value = min(
        [
            min(float(parent_x[i]), float(parent_y[i]))
            - ALPHA * real_representation[i]
            for i in range(DIMENSION)
        ]
    )

    max_value = max(
        [
            max(float(parent_x[i]), float(parent_y[i]))
            - ALPHA * real_representation[i]
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
                genetic_operators_blx_alpha(
                    real_representation, parent_x, parent_y
                )
            )
        kids_info["realRepresentation"] = str(kids_info["realRepresentation"])
        return kids_info

    kid_x = pd.DataFrame(_generate_kids_info(idx))
    kid_y = pd.DataFrame(_generate_kids_info(max(kid_x.id) + 1))
    return kid_x, kid_y


def cross_validation_blx_alpha_beta(community, parents, gen, cross_taxes):
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

    temporary_generation = []
    idx = max(community.id) + 1

    for i in range(0, len(parents) - 1, 2):
        if i == len(parents):
            return temporary_generation
        parent_x = parents[i]
        parent_y = parents[i + 1]
        if parent_x.id == parent_y.id:
            counter = i + 2
            while parent_x.id == parent_y.id:
                if counter >= len(parents):
                    random_value = random.randint(0, len(parents) - 1)
                    parent_y = parents[random_value]
                else:
                    parent_y = parents[counter]
                counter += 1

        if parent_x["finalFitness"] <= parent_y["finalFitness"]:
            parent_x, parent_y = (
                parent_x["realRepresentation"],
                parent_y["realRepresentation"],
            )
        else:
            parent_x, parent_y = (
                parent_y["realRepresentation"],
                parent_x["realRepresentation"],
            )

        if type(parent_x) != list:
            parent_x = ast.literal_eval(parent_x)
        if type(parent_y) != list:
            parent_y = ast.literal_eval(parent_y)

        real_representation = list(np.array(parent_x) - np.array(parent_y))

        kid_x = pd.DataFrame(_generate_kids_info(idx))

        kid_x["fitness"] = kid_x[["realRepresentation", "fitness"]].apply(
            func_obj, axis=1
        )
        idx = max(kid_x.id) + 1
        kid_y = pd.DataFrame(_generate_kids_info(idx))
        kid_y["fitness"] = kid_y[["realRepresentation", "fitness"]].apply(
            func_obj, axis=1
        )
        idx = max(kid_y.id) + 1
        temporary_generation.append(kid_x)
        temporary_generation.append(kid_y)
    if cross_taxes == 1.0:
        return temporary_generation
    else:
        how_many_new_gen = round(len(temporary_generation) * cross_taxes)
        how_many_old_gen = len(temporary_generation) - how_many_new_gen
        new_gen = random.choices(temporary_generation, k=how_many_new_gen)
        old_gen = community.sample(how_many_old_gen)
        old_gen_ids = old_gen["id"].unique()
        for idx in old_gen_ids:
            row = old_gen.loc[old_gen["id"] == idx]
            new_gen.append(row)
        return new_gen


def make_mutation(mutation_taxes, temp_pop):
    """
    The mutation operation will be made by choosing one random index
    in the children realRepresentation that should be changed
    Args:
        mutation_taxes: Float with the mutation probability
        kid: DataFrame with one children information
    Returns:
        DataFrame with one children information
    """
    temporary_population = []
    for kid in temp_pop:
        for i in range(DIMENSION):

            mutation_probability = random.random()
            if mutation_probability <= mutation_taxes:
                mutation_index = random.randint(0, DIMENSION - 1)
                if type(kid["realRepresentation"].to_list()[0]) == str:
                    list_real_representation = ast.literal_eval(
                        kid["realRepresentation"].to_list()[0]
                    )
                else:
                    list_real_representation = kid[
                        "realRepresentation"
                    ].to_list()[0]

                list_real_representation[mutation_index] = random.random()
                kid["realRepresentation"] = str(list_real_representation)
                break
        temporary_population.append(kid)
    return temporary_population


def select_by_elitims(community, temporary_population):
    best_subject = community.loc[
        community["fitness"] == min(community["fitness"])
    ].head()
    random_index = random.randrange(0, len(temporary_population) - 1)
    temporary_population[random_index] = best_subject
    new_generation = pd.DataFrame(columns=community.columns)
    for i in temporary_population:
        new_generation = new_generation.append(i)
    return new_generation


def extract_arguments():
    return pd.read_csv("ag_real/dataset/dataset.csv")


def write_output_csv(best_fitness, avg_fitness, tax, runner):
    df = pd.DataFrame()

    for i in range(10):
        temp_df = pd.DataFrame(
            {
                "Average": avg_fitness,
                "Best Fitness": best_fitness,
                "Round": [runner] * len(best_fitness),
            }
        )
        df = pd.concat([df, temp_df])
    mutation = tax["Mutation"]
    cross = tax["Cross"]
    pop_size = int(tax["Population Size"])
    n_gen = int(tax["Number of Generations"])
    customized_values = f"{mutation}_{cross}_{pop_size}_{n_gen}_{runner}"

    df.to_csv(f"ag_real/dataset/results_{customized_values}.csv")


def main(df, runner):

    best_fitness = []
    avg_fitness = []
    for index, tax in df.iterrows():
        generation = int(tax["Number of Generations"])
        population_size = int(tax["Population Size"])
        mutation_taxes = tax["Mutation"]
        cross_taxes = tax["Cross"]

        community = create_random_comunity(population_size, generation)
        community["fitness"] = community[
            ["realRepresentation", "fitness"]
        ].apply(func_obj, axis=1)

        for _ in range(1, generation):
            random_parents = get_parents(community, population_size)
            temporary_population = cross_validation_blx_alpha_beta(
                community, random_parents, population_size, cross_taxes
            )

            temporary_population = make_mutation(
                mutation_taxes, temporary_population
            )
            community = select_by_elitims(community, temporary_population)
            best_fitness.append(min(community["fitness"]))
            avg_fitness.append(
                sum(community["fitness"]) / len(community["fitness"])
            )
            generation += 1
        write_output_csv(best_fitness, avg_fitness, tax, runner)
    runner += 1


if __name__ == "__main__":
    df = extract_arguments()
    runner = 0
    for i in range(10):
        main(df, runner)
        runner += 1

import logging
import random
from typing import List, Dict, Tuple, Optional
import numpy as np
from numpy.random import default_rng

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration and constants
CONFIG = {
    "population_size": 50,
    "generations": 100,
    "crossover_probability": 0.7,
    "mutation_probability": 0.01,
    "tournament_size": 5,
    "elite_count": 5,
    "gene_limits": (-100, 100),  # Minimum and maximum gene values
    "gene_count": 10,  # Number of genes in each chromosome
    "fitness_threshold": -50.0,  # Minimum fitness value for acceptable solution
}

GENE_LIMIT_MIN, GENE_LIMIT_MAX = CONFIG["gene_limits"]


class Individual:
    def __init__(self, genes: List[float] = None):
        if genes is None:
            self.genes = [
                random.uniform(GENE_LIMIT_MIN, GENE_LIMIT_MAX) for _ in range(CONFIG["gene_count"])
            ]
        else:
            self.genes = genes

    def __str__(self):
        return str(self.genes)

    def compute_fitness(self) -> float:
        # Example fitness function: Sum of absolute values of genes
        fitness = sum(abs(gene) for gene in self.genes)
        return fitness


class Population:
    def __init__(self, size: int):
        self.individuals = [Individual() for _ in range(size)]
        self.fitness_values = [individual.compute_fitness() for individual in self.individuals]

    def get_elite(self) -> List[Individual]:
        elite_index = np.argsort(self.fitness_values)[: CONFIG["elite_count"]]
        return [self.individuals[i] for i in elite_index]

    def get_random_individual(self) -> Individual:
        random_index = random.randint(0, len(self.individuals) - 1)
        return self.individuals[random_index]

    def get_tournament_participants(self) -> List[Individual]:
        participants = []
        for _ in range(CONFIG["tournament_size"]):
            participants.append(self.get_random_individual())
        return participants

    def get_fittest(self) -> Individual:
        fittest_index = np.argmax(self.fitness_values)
        return self.individuals[fittest_index]

    def compute_fitness(self):
        for i, individual in enumerate(self.individuals):
            self.fitness_values[i] = individual.compute_fitness()


def select_parents(population: Population) -> Tuple[Individual, Individual]:
    tournament_participants = population.get_tournament_participants()
    fittest_participant = min(tournament_participants, key=lambda x: x.compute_fitness())
    second_fittest_participant = min(
        [p for p in tournament_participants if p != fittest_participant],
        key=lambda x: x.compute_fitness(),
    )
    return fittest_participant, second_fittest_participant


def perform_crossover(
    parent1: Individual, parent2: Individual, p_cross: float
) -> Tuple[Individual, Individual]:
    child1 = Individual()
    child2 = Individual()
    crossover_points = random.sample(range(1, CONFIG["gene_count"] - 1), 2)
    crossover_point1, crossover_point2 = sorted(crossover_points)

    gene_index = 0
    for i in range(crossover_point1):
        child1.genes[gene_index] = parent1.genes[gene_index]
        child2.genes[gene_index] = parent2.genes[gene_index]
        gene_index += 1

    for i in range(crossover_point1, crossover_point2):
        child1.genes[gene_index] = parent2.genes[gene_index]
        child2.genes[gene_index] = parent1.genes[gene_index]
        gene_index += 1

    for i in range(crossover_point2, CONFIG["gene_count"]):
        child1.genes[gene_index] = parent1.genes[gene_index]
        child2.genes[gene_index] = parent2.genes[gene_index]
        gene_index += 1

    return child1, child2


def mutate_genes(individual: Individual, p_mut: float):
    for i in range(CONFIG["gene_count"]):
        if random.random() < p_mut:
            individual.genes[i] = random.uniform(GENE_LIMIT_MIN, GENE_LIMIT_MAX)


def evolve_population(population: Population, p_cross: float, p_mut: float) -> Population:
    new_population = Population(0)

    # Perform elitism
    elite_individuals = population.get_elite()
    new_population.individuals += elite_individuals

    while len(new_population.individuals) < CONFIG["population_size"]:
        parent1, parent2 = select_parents(population)

        # Perform crossover
        child1, child2 = perform_crossover(parent1, parent2, p_cross)

        # Perform mutation
        mutate_genes(child1, p_mut)
        mutate_genes(child2, p_mut)

        new_population.individuals.append(child1)
        new_population.individuals.append(child2)

    new_population.compute_fitness()
    return new_population


def genetic_algorithm() -> Individual:
    current_generation = 0
    initial_population = Population(CONFIG["population_size"])
    best_individual = initial_population.get_fittest()

    logger.info("Starting genetic algorithm optimization...")
    logger.info("Initial population fitness: %s", initial_population.fitness_values)

    while current_generation < CONFIG["generations"]:
        current_generation += 1
        logger.info("Starting generation %d", current_generation)

        new_population = evolve_population(initial_population, CONFIG["crossover_probability"], CONFIG["mutation_probability"])

        # Update the best individual found so far
        if new_population.get_fittest().compute_fitness() > best_individual.compute_fitness():
            best_individual = new_population.get_fittest()
            logger.info("New best individual found: %s", best_individual)

        # Update the initial population for the next generation
        initial_population = new_population

    logger.info("Genetic algorithm optimization completed.")
    logger.info("Best individual found: %s", best_individual)
    return best_individual


if __name__ == "__main__":
    # Example usage
    best_solution = genetic_algorithm()
    print("Best solution found:", best_solution)
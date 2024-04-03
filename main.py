import random
from enum import Enum
import matplotlib.pyplot as plt

class State(Enum):
    AVAILABLE = 0
    BORROWED = 1
    RESERVED = 2
    BORROWED_RESERVED = 3

class Event(Enum):
    BORROW = 0
    RETURN = 1
    RESERVE = 2
    CANCEL = 3

class BookFSM:
    def __init__(self):
        self.transition_table = {}
        self.current_state = State.AVAILABLE
        # Add transitions to the Book FSM
        self.add_transition(State.AVAILABLE, Event.BORROW, State.BORROWED)
        self.add_transition(State.AVAILABLE, Event.RESERVE, State.RESERVED)
        self.add_transition(State.BORROWED, Event.RETURN, State.AVAILABLE)
        self.add_transition(State.BORROWED, Event.RESERVE, State.BORROWED_RESERVED)
        self.add_transition(State.RESERVED, Event.CANCEL, State.AVAILABLE)
        self.add_transition(State.RESERVED, Event.BORROW, State.BORROWED)
        self.add_transition(State.BORROWED_RESERVED, Event.RETURN, State.RESERVED)
        self.add_transition(State.BORROWED_RESERVED, Event.CANCEL, State.BORROWED)

    def add_transition(self, from_state, event, to_state):
        if from_state not in self.transition_table:
            self.transition_table[from_state] = {}
        self.transition_table[from_state][event] = to_state

    def process_event(self, event):
        current_state = self.current_state
        if current_state in self.transition_table:
            if event in self.transition_table[current_state]:
                self.current_state = self.transition_table[current_state][event]
                # print(f"Transitioned from State {current_state} to State {self.current_state}")
                return True
        # print(f"No transition defined for Event {event} in State {self.current_state}")
        return False

    def get_current_state(self):
        return self.current_state


# Define the fitness function to evaluate test suites
def fitness(chromosome):
    book = BookFSM()
    score = 0
    for event in chromosome:
        if book.process_event(event) is False:
            break
        score += 1
    return score

# Selection operator: Tournament selection
def selection(population, fitness_scores, tournament_size):
    selected_individuals = []
    while len(selected_individuals) < len(population):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        min_fitness_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        selected_individuals.append(population[min_fitness_index])
    return selected_individuals


# Crossover operator: Single-point crossover
def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(0, (len(parent1) - 1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Crossover operator: Uniform crossover
def uniform_crossover(parent1, parent2):
    child1 = []
    child2 = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.choice([True, False]):
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return child1, child2

# Mutation operator: Random mutation
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.choice(list(Event))

def output_population(population, fitness_scores, generation, file):
    cmax = max(fitness_scores)
    csum = sum(fitness_scores)
    avg = csum / len(fitness_scores)
    for ind, chromosome in enumerate(population):
        file.write(f"C{ind}: {out(chromosome)} | Fitness:{fitness_scores[ind]}\n")
    file.write(f'Generation {generation}  Best: {cmax}  Average: {avg}\n\n')
    return cmax, avg

def out(chromosome):
    output = ""
    for event in chromosome:
        output += str(event.value) + " "
    return output

def plot_fitness_progress(cmax_values, avg_values, gen_size, filename=None):
    plt.figure()
    generations = [i for i in range(gen_size)]
    plt.plot(generations, cmax_values, label='max')
    plt.plot(generations, avg_values, label='avg')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progress Across Generations')
    plt.legend()

    if filename:
        plt.savefig(filename)

# Genetic algorithm to generate test suites
def generate_test_suites(chromosome_size, population_size, num_generations, tournament_size, crossover_rate, mutation_rate):

    file = open("results.txt","w")

    cmax_values = []
    avg_values = []
    population = []

    for _ in range(population_size):
        chromosome = [random.choice(list(Event)) for _ in range(chromosome_size)]
        population.append(chromosome)

    # Evolve test suites over multiple generations
    for generation in range(num_generations):
        # Evaluate fitness of each test suite in the population
        fitness_scores = [fitness(chromosome) for chromosome in population]

        # Select individuals for reproduction using tournament selection
        selected_individuals = selection(population, fitness_scores, tournament_size)

        # Perform crossover to create offspring
        offspring = []
        for i in range(0, len(selected_individuals), 2):
            if random.random() < crossover_rate:
                child1, child2 = uniform_crossover(selected_individuals[i], selected_individuals[i + 1])
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(selected_individuals[i])
                offspring.append(selected_individuals[i + 1])

        # Mutate offspring
        for chromosome in offspring:
            mutate(chromosome, mutation_rate)

        # Replace population with offspring
        population = offspring
        cmax, avg = output_population(population, fitness_scores, generation, file)
        cmax_values.append(cmax)
        avg_values.append(avg)

    return cmax_values, avg_values

def experiment_longest_valid_path():
    # Define parameters for genetic algorithm
    chromosome_size = 30
    population_size = 30
    num_generations = 30
    registry = []
    for ts in range(2, 9):
        for cr in range(2, 9):
            for mr in range(2, 9):
                tournament_size = ts
                crossover_rate = cr / 10
                mutation_rate = mr / 10

                cmax_values, avg_values = generate_test_suites(chromosome_size, population_size, num_generations,
                                                               tournament_size, crossover_rate, mutation_rate)
                top_cmax = sum(cmax_values) / num_generations
                top_avg = sum(avg_values) / num_generations
                registry.append((top_cmax, top_avg, ts, cr, mr, cmax_values, avg_values))

    # Sort registry by top_cmax value
    top_cmax_list = sorted(registry, key=lambda x: x[0], reverse=True)
    top_cmax_list = top_cmax_list[:10]
    # Sort registry by top_avg value
    top_avg_list = sorted(registry, key=lambda x: x[1], reverse=True)
    top_avg_list = top_avg_list[:10]

    for tuplu in top_cmax_list:
        top_cmax, top_avg, ts, cr, mr, cmax_values, avg_values = tuplu[0], tuplu[1], tuplu[2], tuplu[3], tuplu[4], \
        tuplu[5], tuplu[6]
        plot_fitness_progress(cmax_values, avg_values, num_generations,
                              f'./plots/exp2-uniform-crossover/top_max/ts{ts}-cr0{cr}-mr0{mr}')

    for tuplu in top_avg_list:
        top_cmax, top_avg, ts, cr, mr, cmax_values, avg_values = tuplu[0], tuplu[1], tuplu[2], tuplu[3], tuplu[4], \
        tuplu[5], tuplu[6]
        plot_fitness_progress(cmax_values, avg_values, num_generations,
                              f'./plots/exp2-uniform-crossover/top_avg/ts{ts}-cr0{cr}-mr0{mr}')

if __name__ == "__main__":
    experiment_longest_valid_path()
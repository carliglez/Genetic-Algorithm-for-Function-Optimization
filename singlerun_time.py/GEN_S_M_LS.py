import numpy as np
import time
import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.my_functions import (
  alluffi_pentiny,
  bohachevsky_1, bohachevsky_2,
  becker_lago,
  branin,
  camel,
  cb3,
  cosine_mixture,
  dejong,
  easom,
  exponential,
  gkls,
  goldstein_price,
  griewank2,
  hansen,
  hartman_3, hartman_6,
  rastrigin,
  rosenbrock,
  shekel_5, shekel_7, shekel10,
  shubert,
  sinusoidal,
  test2n, test30n,
  potential
)

# Genetic Algorithm Function
def genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound, u_bound, fitness_function, K_ls):
    # Initialization
    dim = len(l_bound)
    S = np.random.uniform(l_bound, u_bound, size=(N, dim))
    iter_count = 0
    last_improvement_generation = 0
    best_fitness_values = []

    while True:
        # Evaluation
        fitness_values = np.apply_along_axis(fitness_function, 1, S)

        # Termination check
        f_l = np.min(fitness_values)
        f_h = np.max(fitness_values)
        best_fitness_values.append(f_l)

        if np.abs(f_h - f_l) <= e or iter_count > ITERMAX or check_stopping_rule(best_fitness_values, last_improvement_generation):
            break

        # Selection and Crossover
        selected_parents = tournament_selection(S, fitness_values)
        offsprings = crossover(selected_parents)

        # Mutation
        best_solution = S[np.argmin(fitness_values)]  # Get the best solution
        mutated_offsprings = mutation(offsprings, p_m, b, iter_count, l_bound, u_bound, best_solution)

        # Replacement
        S = replace_population(S, fitness_values, mutated_offsprings)

        # Local Technique every K_ls generations
        if iter_count % K_ls == 0:
            best_index = np.argmin(np.apply_along_axis(fitness_function, 1, S))
            local_technique(S, fitness_function, best_index)

        # Increment iteration count
        iter_count += 1

    # Return the best solution found
    best_solution = S[np.argmin(fitness_values)]
    return best_solution, fitness_function(best_solution)

# Check Stopping Rule
def check_stopping_rule(best_fitness_values, last_improvement_generation, patience=10):
    if len(best_fitness_values) > patience:
        recent_best_values = best_fitness_values[-patience:]

        # Check if there are enough elements to calculate variance
        if len(recent_best_values) > 1:
            current_variance = np.nanvar(recent_best_values)

            # Check for zero variance to avoid division by zero
            if current_variance > 0 and last_improvement_generation > 0:
                last_variance = np.nanvar(best_fitness_values[:last_improvement_generation])

                return current_variance <= last_variance / 2

    return False

# Tournament Selection
def tournament_selection(population, fitness_values, tournament_size=4):
    selected_parents = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = fitness_values[tournament_indices]
        selected_parents.append(population[tournament_indices[np.argmin(tournament_fitness)]])
    return np.array(selected_parents)

# Crossover
def crossover(parents):
    a_values = np.random.uniform(-0.5, 1.5, size=parents.shape)
    offsprings = a_values * parents + (1 - a_values) * np.roll(parents, shift=1, axis=0)
    return offsprings

# Mutation
def mutation(offsprings, p_m, b, iter_count, l_bound, u_bound, best_solution):
    mask = np.random.choice([0, 1], size=offsprings.shape, p=[1 - p_m, p_m])
    t_values = np.random.choice([0, 1], size=offsprings.shape[1])
    
    # Acceleration coefficients
    c1 = 1.5
    c2 = 1.5
    
    # Create a copy of the best solution
    xb = np.copy(best_solution)
    
    mutated_offsprings = np.where(mask == 1, 
                                   c1 * np.random.rand() * (xb - offsprings) + c2 * np.random.rand() * (xb - offsprings),
                                   offsprings)
    
    mutated_offsprings = np.clip(mutated_offsprings, l_bound, u_bound)
    return mutated_offsprings

# Replacement
def replace_population(population, fitness_values, offsprings):
    sorted_indices = np.argsort(fitness_values)
    worst_indices = sorted_indices[-len(offsprings):]
    population[worst_indices] = offsprings
    return population

# Local Technique
def local_technique(population, fitness_function, best_index):
    random_index = np.random.randint(len(population))
    random_point = population[random_index]
    gamma_values = np.random.uniform(-0.5, 1.5, size=population.shape[1])
    trial_point = (1 + gamma_values) * population[best_index] - gamma_values * random_point
    if fitness_function(trial_point) <= fitness_function(population[best_index]):
        population[best_index] = trial_point

# Define parameters
N = 100  # Population size
ITERMAX = 200  # Maximum number of iterations
p_m = 0.05  # Mutation probability
b = 5  # User-defined parameter for mutation
e = 1e-4  # Termination criterion
# modification 3.3
K_ls = 5
# alluffi_pentiny
l_bound, u_bound = np.array([-10, -10]), np.array([10, 10])
# bohachevsky_1
l_bound_b1, u_bound_b1 = np.array([-100, -100]), np.array([100, 100])
# bohachevsky_2
l_bound_b2, u_bound_b2 = np.array([-50, -50]), np.array([50, 50])
# becker_lago
l_bound_bl, u_bound_bl = np.array([-10, -10]), np.array([10, 10])
# branin
l_bound_b, u_bound_b = np.array([-5, 0]), np.array([10, 15])
# camel
l_bound_c, u_bound_c = np.array([-5, -5]), np.array([5, 5])
# cosine mixture
n = 4  # dimension for CM function
l_bound_cm, u_bound_cm = -1, 1  # bounds for CM function
# dejong
n_dejong = 3  # dimension for DeJong function
l_bound_dejong, u_bound_dejong = -5.12, 5.12  # bounds for DeJong function
# easom
n_easom = 2  # dimension for Easom function
l_bound_easom, u_bound_easom = -100, 100  # bounds for Easom function
# exponential
n_exponential2 = 2  # dimension for Exponential function
n_exponential4 = 4  # dimension for Exponential function
n_exponential8 = 8  # dimension for Exponential function
n_exponential16 = 16  # dimension for Exponential function
n_exponential32 = 32  # dimension for Exponential function
n_exponential64 = 64  # dimension for Exponential function
l_bound_exp, u_bound_exp = -1, 1  # bounds for Exponential function
# gkls
n_gkls2 = 2  # dimension for Gkls function
n_gkls3 = 3  # dimension for Gkls function
w_gkls = 50  # number of local minima for Gkls function
l_bound_gkls, u_bound_gkls = -1, 1  # bounds for Gkls function
# goldstein
l_bound_gp, u_bound_gp = -2, 2
# griewank2
l_bound_griewank2, u_bound_griewank2 = -100, 100
# hansen
l_bound_hansen, u_bound_hansen = -10, 10
# hartman 3
l_bound_hartman3, u_bound_hartman3 = 0, 1
# hartman 6
l_bound_hartman6, u_bound_hartman6 = 0, 1
# rastrigin
l_bound_rastrigin, u_bound_rastrigin = -1, 1
# rosenbrock
n_rosenbrock = 2
l_bound_rosenbrock, u_bound_rosenbrock = -30, 30
# shekel 5
n_shekel5 = 4
l_bound_shekel5, u_bound_shekel5 = 0, 10
# shekel 7
n_shekel7 = 4
l_bound_shekel7, u_bound_shekel7 = 0, 10
# shekel 10
n_shekel10 = 4
l_bound_shekel10, u_bound_shekel10 = 0, 10
# shubert
l_bound_shubert, u_bound_shubert = -10, 10
# sinusoidal
n_sinusoidal2 = 2  # Number of dimensions
n_sinusoidal4 = 4  # Number of dimensions
n_sinusoidal8 = 8  # Number of dimensions
n_sinusoidal16 = 16  # Number of dimensions
n_sinusoidal32 = 32  # Number of dimensions
z_sinusoidal = np.pi / 6  # Constant parameter for Sinusoidal
l_bound_sinusoidal, u_bound_sinusoidal = 0, np.pi
# test2n
n_test2n4 = 4  # Number of dimensions
n_test2n5 = 5  # Number of dimensions
n_test2n6 = 6  # Number of dimensions
n_test2n7 = 7  # Number of dimensions
l_bound_test2n, u_bound_test2n = -5, 5
# test30n
n_test30n3 = 3
n_test30n4 = 4
l_bound_test30n, u_bound_test30n = -10, 10
# potential
bounds = [0, 1]  # Assuming a box with sides of length 1
n_atoms_potential3 = 3
l_bound_potential3, u_bound_potential3 = np.array([bounds[0]] * (3 * n_atoms_potential3)), np.array([bounds[1]] * (3 * n_atoms_potential3))
n_atoms_potential5 = 5
l_bound_potential5, u_bound_potential5 = np.array([bounds[0]] * (3 * n_atoms_potential5)), np.array([bounds[1]] * (3 * n_atoms_potential5))

# Execution start_time
start_time = time.time()

# Run the genetic algorithm for Alluffi-Pentiny
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound, u_bound, alluffi_pentiny, K_ls)
# Print results
print("Best Solution (Alluffi-Pentiny):", best_solution)
print("Minimum Value (Alluffi-Pentiny):", min_value)

# Run the genetic algorithm for Bohachevsky 1 function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_b1, u_bound_b1, bohachevsky_1, K_ls)
# Print results
print("Best Solution (Bohachevsky 1):", best_solution)
print("Minimum Value (Bohachevsky 1):", min_value)

# Run the genetic algorithm for Bohachevsky 2 function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_b2, u_bound_b2, bohachevsky_2, K_ls)
# Print results
print("Best Solution (Bohachevsky 2):", best_solution)
print("Minimum Value (Bohachevsky 2):", min_value)

# Run the genetic algorithm for Becker and Lago function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_bl, u_bound_bl, becker_lago, K_ls)
# Print results
print("Best Solution (Becker and Lago):", best_solution)
print("Minimum Value (Becker and Lago):", min_value)

# Run the genetic algorithm for Branin function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_b, u_bound_b, branin, K_ls)
# Print results
print("Best Solution (Branin):", best_solution)
print("Minimum Value (Branin):", min_value)

# Run the genetic algorithm for Camel function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_c, u_bound_c, camel, K_ls)
# Print results
print("Best Solution (Camel):", best_solution)
print("Minimum Value (Camel):", min_value)

# Run the genetic algorithm for Cb3 function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_c, u_bound_c, cb3, K_ls)
# Print results
print("Best Solution (CB3):", best_solution)
print("Minimum Value (CB3):", min_value)

# Run the genetic algorithm for Cosine Mixture (CM) function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_cm*np.ones(n), u_bound_cm*np.ones(n), cosine_mixture, K_ls)
# Print results
print("Best Solution (CM):", best_solution)
print("Minimum Value (CM):", min_value)

# Run the genetic algorithm for DeJong function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_dejong*np.ones(n_dejong), u_bound_dejong*np.ones(n_dejong), dejong, K_ls)
# Print results for DeJong function
print("Best Solution (DeJong):", best_solution)
print("Minimum Value (DeJong):", min_value)

# Run the genetic algorithm for Easom function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_easom*np.ones(n_easom), u_bound_easom*np.ones(n_easom), easom, K_ls)
# Print results for Easom function
print("Best Solution (Easom):", best_solution)
print("Minimum Value (Easom):", min_value)

# Run the genetic algorithm for Exponential function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_exp*np.ones(n_exponential2), u_bound_exp*np.ones(n_exponential2), exponential, K_ls)
# Print results for Exponential function
print("Best Solution (EXP2):", best_solution)
print("Minimum Value (EXP2):", min_value)
# Run the genetic algorithm for Exponential function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_exp*np.ones(n_exponential4), u_bound_exp*np.ones(n_exponential4), exponential, K_ls)
# Print results for Exponential function
print("Best Solution (EXP4):", best_solution)
print("Minimum Value (EXP4):", min_value)
# Run the genetic algorithm for Exponential function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_exp*np.ones(n_exponential8), u_bound_exp*np.ones(n_exponential8), exponential, K_ls)
# Print results for Exponential function
print("Best Solution (EXP8):", best_solution)
print("Minimum Value (EXP8):", min_value)
# Run the genetic algorithm for Exponential function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_exp*np.ones(n_exponential16), u_bound_exp*np.ones(n_exponential16), exponential, K_ls)
# Print results for Exponential function
print("Best Solution (EXP16):", best_solution)
print("Minimum Value (EXP16):", min_value)
# Run the genetic algorithm for Exponential function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_exp*np.ones(n_exponential32), u_bound_exp*np.ones(n_exponential32), exponential, K_ls)
# Print results for Exponential function
print("Best Solution (EXP32):", best_solution)
print("Minimum Value (EXP32):", min_value)
# Run the genetic algorithm for Exponential function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_exp*np.ones(n_exponential64), u_bound_exp*np.ones(n_exponential64), exponential, K_ls)
# Print results for Exponential function
print("Best Solution (EXP64):", best_solution)
print("Minimum Value (EXP64):", min_value)

# Run the genetic algorithm for Gkls function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_gkls*np.ones(n_gkls2), u_bound_gkls*np.ones(n_gkls2), lambda x: gkls(x, n_gkls2, w_gkls), K_ls)
# Print results for Gkls function
print("Best Solution (GKLS250):", best_solution)
print("Minimum Value (GKLS250):", min_value)

# Run the genetic algorithm for Gkls function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_gkls*np.ones(n_gkls3), u_bound_gkls*np.ones(n_gkls3), lambda x: gkls(x, n_gkls3, w_gkls), K_ls)
# Print results for Gkls function
print("Best Solution (GKLS350):", best_solution)
print("Minimum Value (GKLS350):", min_value)

# Run the genetic algorithm for Goldstein and Price function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_gp*np.ones(2), u_bound_gp*np.ones(2), goldstein_price, K_ls)
# Print results for Goldstein and Price function
print("Best Solution (Goldstein and Price):", best_solution)
print("Minimum Value (Goldstein and Price):", min_value)

# Run the genetic algorithm for Griewank2 function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_griewank2*np.ones(2), u_bound_griewank2*np.ones(2), griewank2, K_ls)
# Print results for Griewank2 function
print("Best Solution (Griewank2):", best_solution)
print("Minimum Value (Griewank2):", min_value)

# Run the genetic algorithm for Hansen function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_hansen*np.ones(2), u_bound_hansen*np.ones(2), hansen, K_ls)
# Print results for Hansen function
print("Best Solution (Hansen):", best_solution)
print("Minimum Value (Hansen):", min_value)

# Run the genetic algorithm for the Hartman 3 function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_hartman3*np.ones(3), u_bound_hartman3*np.ones(3), hartman_3, K_ls)
# Print results for Hartman 3 function
print("Best Solution (Hartman 3):", best_solution)
print("Minimum Value (Hartman 3):", min_value)

# Run the genetic algorithm for the Hartman 6 function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_hartman6*np.ones(6), u_bound_hartman6*np.ones(6), hartman_6, K_ls)
# Print results for Hartman 6 function
print("Best Solution (Hartman 6):", best_solution)
print("Minimum Value (Hartman 6):", min_value)

# Run the genetic algorithm for the Rastrigin function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_rastrigin*np.ones(2), u_bound_rastrigin*np.ones(2), rastrigin, K_ls)
# Print results for Rastrigin function
print("Best Solution (Rastrigin):", best_solution)
print("Minimum Value (Rastrigin):", min_value)

# Run the genetic algorithm for the Rosenbrock function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_rosenbrock*np.ones(n_rosenbrock), u_bound*np.ones(n_rosenbrock), rosenbrock, K_ls)
# Print results for Rosenbrock function
print("Best Solution (Rosenbrock):", best_solution)
print("Minimum Value (Rosenbrock):", min_value)

# Run the genetic algorithm for the Shekel 5 function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_shekel5*np.ones(n_shekel5), u_bound_shekel5*np.ones(n_shekel5), shekel_5, K_ls)
# Print results for Shekel 5 function
print("Best Solution (Shekel 5):", best_solution)
print("Minimum Value (Shekel 5):", min_value)

# Run the genetic algorithm for the Shekel 7 function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_shekel7*np.ones(n_shekel7), u_bound_shekel7*np.ones(n_shekel7), shekel_7, K_ls)
# Print results for Shekel 7 function
print("Best Solution (Shekel 7):", best_solution)
print("Minimum Value (Shekel 7):", min_value)

# Run the genetic algorithm for the Shekel 10 function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_shekel10*np.ones(n_shekel10), u_bound_shekel10*np.ones(n_shekel10), shekel10, K_ls)
# Print results for Shekel 10 function
print("Best Solution (Shekel 10):", best_solution)
print("Minimum Value (Shekel 10):", min_value)

# Run the genetic algorithm for the Shubert function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_shubert*np.ones(2), u_bound_shubert*np.ones(2), shubert, K_ls)
# Print results for Shubert function
print("Best Solution (Shubert):", best_solution)
print("Minimum Value (Shubert):", min_value)

# Run the genetic algorithm for the Sinusoidal function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_sinusoidal*np.ones(n_sinusoidal2), u_bound_sinusoidal*np.ones(n_sinusoidal2), lambda x: sinusoidal(x, z_sinusoidal), K_ls
    )
# Print results for Sinusoidal function
print("Best Solution (SINU2):", best_solution)
print("Minimum Value (SINU2):", min_value)

# Run the genetic algorithm for the Sinusoidal function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_sinusoidal*np.ones(n_sinusoidal4), u_bound_sinusoidal*np.ones(n_sinusoidal4), lambda x: sinusoidal(x, z_sinusoidal), K_ls
    )
# Print results for Sinusoidal function
print("Best Solution (SINU4):", best_solution)
print("Minimum Value (SINU4):", min_value)

# Run the genetic algorithm for the Sinusoidal function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_sinusoidal*np.ones(n_sinusoidal8), u_bound_sinusoidal*np.ones(n_sinusoidal8), lambda x: sinusoidal(x, z_sinusoidal), K_ls
    )
# Print results for Sinusoidal function
print("Best Solution (SINU8):", best_solution)
print("Minimum Value (SINU8):", min_value)

# Run the genetic algorithm for the Sinusoidal function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_sinusoidal*np.ones(n_sinusoidal16), u_bound_sinusoidal*np.ones(n_sinusoidal16), lambda x: sinusoidal(x, z_sinusoidal), K_ls
    )
# Print results for Sinusoidal function
print("Best Solution (SINU16):", best_solution)
print("Minimum Value (SINU16):", min_value)

# Run the genetic algorithm for the Sinusoidal function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_sinusoidal*np.ones(n_sinusoidal32), u_bound_sinusoidal*np.ones(n_sinusoidal32), lambda x: sinusoidal(x, z_sinusoidal), K_ls
    )
# Print results for Sinusoidal function
print("Best Solution (SINU32):", best_solution)
print("Minimum Value (SINU32):", min_value)

# Run the genetic algorithm for the Test2N function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_test2n*np.ones(n_test2n4), u_bound_test2n*np.ones(n_test2n4), test2n, K_ls
    )
# Print results for Test2N function
print("Best Solution (Test2N4):", best_solution)
print("Minimum Value (Test2N4):", min_value)

# Run the genetic algorithm for the Test2N function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_test2n*np.ones(n_test2n5), u_bound_test2n*np.ones(n_test2n5), test2n, K_ls
    )
# Print results for Test2N function
print("Best Solution (Test2N5):", best_solution)
print("Minimum Value (Test2N5):", min_value)

# Run the genetic algorithm for the Test2N function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_test2n*np.ones(n_test2n6), u_bound_test2n*np.ones(n_test2n6), test2n, K_ls
    )
# Print results for Test2N function
print("Best Solution (Test2N6):", best_solution)
print("Minimum Value (Test2N6):", min_value)

# Run the genetic algorithm for the Test2N function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_test2n*np.ones(n_test2n7), u_bound_test2n*np.ones(n_test2n7), test2n, K_ls
    )
# Print results for Test2N function
print("Best Solution (Test2N7):", best_solution)
print("Minimum Value (Test2N7):", min_value)

# Run the genetic algorithm for the Test30N function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_test30n*np.ones(n_test30n3), u_bound_test30n*np.ones(n_test30n3), test30n, K_ls
    )
# Print results for Test30N function
print("Best Solution (Test30N3):", best_solution)
print("Minimum Value (Test30N3):", min_value)

# Run the genetic algorithm for the Test30N function
best_solution, min_value = genetic_algorithm(
    N, ITERMAX, p_m, b, e, l_bound_test30n*np.ones(n_test30n4), u_bound_test30n*np.ones(n_test30n4), test30n, K_ls
    )
# Print results for Test30N function
print("Best Solution (Test30N4):", best_solution)
print("Minimum Value (Test30N4):", min_value)

# Run the genetic algorithm for the Potential function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_potential3, u_bound_potential3, potential, K_ls)
# Print results for Potential function
print(f"Best Solution (POTENTIAL3): {best_solution}")
print(f"Minimum Value (POTENTIAL3): {min_value}")

# Run the genetic algorithm for the Potential function
best_solution, min_value = genetic_algorithm(N, ITERMAX, p_m, b, e, l_bound_potential5, u_bound_potential5, potential, K_ls)
# Print results for Potential function
print(f"Best Solution (POTENTIAL5): {best_solution}")
print(f"Minimum Value (POTENTIAL5): {min_value}")

# Execution end_time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time} seconds")
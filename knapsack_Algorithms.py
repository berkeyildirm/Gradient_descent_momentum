' I used numPy and pandas library for working with arrays. It is easier with the multi-dimensional arrays. Im using the Anaconda app to use these librarys. '

''' Simulated Annealing ''' 
# Simulated Annealing

import numpy as np
import random

from simanneal import Annealer

list_to_be_presented_in_knapsack = np.zeros(20)
# list_to_be_presented_in_knapsack[0] = 1
# list_to_be_presented_in_knapsack[1] = 1

list_to_be_presented_in_knapsack

mass = np.array([1, .2, .5, 2, .1, .5, 10, 2, 6, 1, 2, .5, 4, 10, 5, 1, 6, 1, 8, 1.5])
len(mass)

max_size = 25 # Kg
max_item = 10 # Items

total_mass = np.sum(mass)
total_mass
# total_value_of_knapsack * 1000 - total_mass_in_knapsack = maximizes the value and minimize the mass if the values are equal
# total_mass is 62.3 which is less than 1000 

values = np.array([1, 2, 5, 10, .5, .3, 1.6, 4, 2, 1.9, 1.25, 2.5, 8, 12, 3.75, 1.4, 3.1, 1.3, .7, .4])
len(values)

def total_value_size(list_to_be_presented_in_knapsack, values, mass, max_size, max_item):
  v = 0.0  # Total value
  s = 0.0  # Total size 
  n = len(list_to_be_presented_in_knapsack)
  cnt = 0 # Count of total items
  for i in range(n):
    if list_to_be_presented_in_knapsack[i] == 1:
      v += values[i]
      s += mass[i]
      cnt += 1
  if s > max_size or cnt > max_item:  # If its too big to fit in knapsack
    v = 0.0
  return v*1000 - s


class KnapSackProblem(Annealer):
  def __init__(self, list_to_be_presented_in_knapsack, values, mass):
    self.state = list_to_be_presented_in_knapsack
    self.values = values
    self.mass = mass
    self.max_size = 25
    self.max_items = 10
    print('Created Constructor')
  def move(self):
    n = len(self.state)
    i = np.random.randint(len(self.state))
    if self.state[i] == 0:
      self.state[i] = 1
    elif self.state[i] == 1:
      self.state[i] = 0
  def energy(self):
    v = 0.0  # Total value 
    s = 0.0  # Total size 
    n = len(self.state)
    cnt = 0  # Count of total items
    for i in range(n):
      if self.state[i] == 1:
        v += self.values[i]
        s += self.mass[i]
        cnt += 1
    to_be_returned = 0
    if s > self.max_size:  # If its too big to fit in knapsack
      to_be_returned = -s
    elif cnt > self.max_items:
      to_be_returned = -cnt
    else:
      to_be_returned = v*1000
    return -to_be_returned


def find_total_value_and_size(list_to_be_presented_in_knapsack):
  v = 0.0  # Total value 
  s = 0.0  # Total size 
  n = len(list_to_be_presented_in_knapsack)
  cnt = 0  # Count of total items
  for i in range(n):
    if list_to_be_presented_in_knapsack[i] == 1:
      v += values[i]
      s += mass[i]
      cnt += 1
  if s > max_size or cnt > max_item:  # If its too big to fit in knapsack
    v = 0.0
  return (v, s, cnt)

def print_best_guess():
  print('Best route :')
  for i in range(0, len(state)):
    if state[i] == 1:
      print(i, mass[i], values[i])
  print('Energy:', -e)
  result = find_total_value_and_size(state)
  print("total_value_in_knapsack:", result[0])
  print("total_mass_in_knapsack:", result[1])
  print("total_count_in_knapsack:", result[2])

KSP = KnapSackProblem(list_to_be_presented_in_knapsack, values, mass)

for i in [1, 10, 100, 1000, 10000]:  
  # Identifies the number of steps
  KSP.steps = i

  state, e = KSP.anneal()   # Running the Annealer
  print_best_guess()
  print(" ")
# I got 48.01 total value with a total mass of 22.2 and a total_count of 10

print('----------Genetic Algorithms---------')
print(' ')
# Genetic Algorithm

import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt
item_number = np.arange(1,21)
weight = np.array([1, .2, .5, 2, .1, .5, 10, 2, 6, 1, 2, .5, 4, 10, 5, 1, 6, 1, 8, 1.5])
value = np.array([1, 2, 5, 10, .5, .3, 1.6, 4, 2, 1.9, 1.25, 2.5, 8, 12, 3.75, 1.4, 3.1, 1.3, .7, .4])
knapsack_threshold = 25    # Maximum weight 
count_threshold = 10  # Maximum count 

solutions_per_population = 300
size_of_population = (solutions_per_population, item_number.shape[0])
print('Population size = {}'.format(size_of_population))
initial_population = np.random.randint(2, size = size_of_population)
initial_population = initial_population.astype(int)
number_of_generations = 30
print('Initial population: \n{}'.format(initial_population))

def calculate_fitness(weight, value, population, threshold, count_threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        sum1 = np.sum(population[i] * value)
        sum2 = np.sum(population[i] * weight)
        sum3 = np.sum(population[i] * 1)
        if sum2 <= threshold and sum3 <= count_threshold:
            fitness[i] = sum1 * 1000 - sum2
        else :
            fitness[i] = 0 
    return fitness.astype(int)

def selection(fitness, number_of_parents, population):
    fitness = list(fitness)
    parents = np.empty((number_of_parents, population.shape[1]))
    for i in range(number_of_parents):
        maximum_fitness_index = np.where(fitness == np.max(fitness))
        parents[i,:] = population[maximum_fitness_index[0][0], :]
        fitness[maximum_fitness_index[0][0]] = -999999
    return parents

def optimize(weight, value, population, size_of_population, number_of_generations, threshold, count_threshold):
    parameters, fitness_history = [], []
    number_of_parents = int(size_of_population[0]/2)
    for i in range(number_of_generations):
        fitness = calculate_fitness(weight, value, population, threshold, count_threshold)
        fitness_history.append(fitness)
        parents = selection(fitness, number_of_parents, population)
        population[0:parents.shape[0], :] = parents
         

    print('Last generation: \n{}\n'.format(population)) 
    fitness_last_gen = calculate_fitness(weight, value, population, threshold, count_threshold)      
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])
    return parameters, fitness_history

parameters, fitness_history = optimize(weight, value, initial_population, size_of_population, number_of_generations, knapsack_threshold, count_threshold)
print('The optimized parameters are: \n{}'.format(parameters))
print(' ')

def find_total_value_and_size(list_to_be_presented_in_knapsack):
  v = 0.0  # Total value 
  s = 0.0  # Total size 
  n = len(list_to_be_presented_in_knapsack)
  cnt = 0 # Count of total items
  for i in range(n):
    if list_to_be_presented_in_knapsack[i] == 1:
      v += values[i]
      s += mass[i]
      cnt += 1
  if s > max_size or cnt > max_item:  # If its too big to fit in knapsack
    v = 0.0
  return (v, s, cnt)

v, s, cnt = find_total_value_and_size(parameters[0])

print("value:", v)
print("size:", s)
print("count:", cnt)
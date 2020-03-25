import random as rand, pickle, client_moodle as cli
from numpy.random import normal
f1 = 'top.pkl'
f2 = 'res.pkl'
f3 = 'topvec.pkl'
fn1 = open(f1, 'rb')
fn2 = open(f2, 'rb')
fn3 = open(f3, 'rb')
arr = pickle.load(fn1)
pop = arr
best = pickle.load(fn2)
top = pickle.load(fn3)
fn1.close()
fn2.close()
check = open('check.txt', 'w')
numpop = 40
matsize = int(numpop/4)

def get_best(pop1, best1, arr1, top1):
    res = []
    w = []
    for i in range(len(pop1)):
        res = cli.get_errors('jcikTU98ZdeaUH5uBHsOPXzXAzAhBVdwtDDj7SqoF98mqbjZLw', pop1[i])
        if (res[0]+0*res[1]) < (best[0]+0*best1[1]) and res[0] <= 1.5*res[1] and res[1] <= 1.5*res[0]:
            arr1 = pop1
            top1 = pop1[i]
            best1 = res
            print(f"Train Error : {best1[0]} , Validation Error : {best1[1]}")
            print(pop[i], file = check)
        w.append(res[0] + res[1]*0)
    zipped_pairs = zip(w, pop1)
    z = [a for _, a in sorted(zipped_pairs)]
    pop1 = z  
    return pop1, best1, arr1, top1

def cross(pop1):
    for i in range(numpop - matsize):
        ind1 = i%matsize
        #ind2 = (i+rand.randint(1, 3))%matsize
        ind2 = ((i+1)%matsize + int(i/matsize))%matsize
        cutoff = rand.randint(4, 8)
        child = pop1[ind1][:cutoff] + pop1[ind2][cutoff:]
        num = rand.randint(1, 10)
        if num == 2 or num == 3:
            ind = rand.randint(0, 10)
            multiply = rand.uniform(0.99, 1.01)
            tt = child[ind]*multiply
            if tt <= 10 and tt >= -10:
                child[ind] = tt            
        pop1[matsize - 1 + i] = child
    return pop1

for i in range(10):
    print(f"Generation. {i}")
    print(f"Generation. {i}", file = check)
    pop, best, arr, top = get_best(pop, best, arr, top)
    pop = cross(pop)

fn1 = open(f1, 'wb')
pickle.dump(arr, fn1)
fn1.close()
fn2 = open(f2, 'wb')
print(best)
pickle.dump(best, fn2)
fn2.close()
fn3 = open(f3, 'wb')
pickle.dump(top, fn3)
fn3.close()
'''
# TEAM NUMBER 91
import random


overweight_error = 3.705362469604573213e+06
population_number = 40
secret_key = 'jcikTU98ZdeaUH5uBHsOPXzXAzAhBVdwtDDj7SqoF98mqbjZLw'
import numpy as np
import client_moodle


def cal_pop_fitness(vector):
    # Calculating the fitness value of each solution in the current population.
    for itr in range( population_number ):
        temp = list( vector[itr][1:] )  # take each of the genes

        err = client_moodle.get_errors( secret_key, temp )
        vector[itr][0] = err[1]  # taking fitness as validation error for now
        print( "query number:- ", itr, vector[itr][0] )


def select_mating_pool(vector, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    return vector[:num_parents, 1:]


def crossover(parents, offspring_size):
    offspring = np.empty( (offspring_size, 11) )

    crossover_point = random.randrange( 2, 9 )

    for k in range( offspring_size ):
        # Index of the first parent to mate.
        parent1_idx = k % 20
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % 20

        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring


def mutate(val, start, stop):
    temp = val * random.uniform( start, stop )
    if (temp > 10.0):
        return mutate( val, start, 10 - val )
    if temp < -10.0:
        return mutate( val, -10 - val, stop )
    return temp


def mutation(offspring, parents, offspring_no, parents_no):
    for idx in range( offspring_no ):
        for g in range( 0, 11 ):
            temp = mutate( offspring[idx][g], -1.0, 1.0 )
            offspring[idx][g] = temp

    for idx in range( parents_no ):
        for g in range( 0, 11 ):
            temp = mutate( parents[idx][g], -1.0, 1.0 )
            parents[idx][g] = temp
    return np.append( parents, offspring, axis=0 )


def initialize_population():  # load previous population as per status
    returnable = np.loadtxt( "./saved_populations.txt", delimiter=',' )

    np.savetxt( './temp.txt', returnable, delimiter=',' )
    return returnable


if __name__ == "__main__":
    vector = initialize_population()

    print( "Population has been loaded" )

    parents = select_mating_pool( vector, 20 )
    offspring = crossover( parents, 20 )
    temp = mutation( offspring, parents, 20, 20 )
    print( "New population has been created" )

    new = np.append( np.zeros( (40, 1) ), temp, axis=1 )
    print( "starting fitness measure now" )

    cal_pop_fitness( new )
    print( "fitness measured" )

    initial = np.loadtxt("./temp.txt",delimiter=',')

    total_population = np.append( initial, new, axis=0 )
    np.savetxt("./before.txt",total_population,delimiter=',')
    total_population = total_population[total_population[:, 0].argsort()]
    np.savetxt("./after.txt",total_population,delimiter=',')

    final_population = total_population[:40, :]

    np.savetxt( './saved_populations.txt', final_population, delimiter=',' )
'''
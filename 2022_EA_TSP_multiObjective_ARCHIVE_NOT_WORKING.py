# -*- coding: utf-8 -*-
"""
Created in July 2022 for Metaheuristics part of Vertiefung Business Analytics

Modified version of https://github.com/ezstoltz/genetic-algorithm/blob/master/genetic_algorithm_TSP.ipynb
(description https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35)

Modified for multicriteria TSP
"""



import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

#Create necessary classes and functions
#Create class to handle "cities
class City:
    def __init__(self, nr, traffic, x, y):
        self.nr = nr
        #attribute used for stress calculation
        self.traffic = traffic
        #coordinates used for distance calculation
        self.x = x
        self.y = y
    
    #calculate distance to other city  
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    #calculate stress on way to other city
    def stress(self,city):
        stress = self.traffic*city.traffic
        if (self.nr > city.nr):
            if self.nr % city.nr == 0:
                stress = stress /2.5
        elif self.nr > 1 and city.nr % self.nr == 0:
            stress = stress /2.5
        return stress

    #provide information about city
    def __repr__(self):
        return "C"+str(self.nr)+"_"+"(" + str(self.x) + "," + str(self.y) + ")_(T:"+str(self.traffic) +")"

#Create a fitness function
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.stress = 0
        self.fitnessDistanceBased = 0.0
        self.fitnessStressBased = 0.0
    
    #fitness calculation for objective: distance
    #1. distance calculation
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                fromCityNr = fromCity.nr-1
                toCityNr = toCity.nr-1
                pathDistance += distances[fromCityNr][toCityNr]
            self.distance = pathDistance
        return self.distance
    
    #2. fitness = 1/distance
    def routeFitnessDistanceBased(self):
        if self.fitnessDistanceBased == 0:
            self.fitnessDistanceBased = 1 / float(self.routeDistance())
        return self.fitnessDistanceBased
    
    #fitness calculation for objective: stress
    #1. stress calculation
    def routeStress(self):
        if self.stress ==0:
            pathStress = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                fromCityNr = fromCity.nr-1
                toCityNr = toCity.nr-1
                pathStress += stresses[fromCityNr][toCityNr]
            self.stress = pathStress
        return self.stress
    
    #2. fitness = 1/stress
    def routeFitnessStressBased(self):
        if self.fitnessStressBased == 0:
            self.fitnessStressBased = 1 / float(self.routeStress())
        return self.fitnessStressBased


#Create our initial population
#Route generator
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

#Create first "population" (list of routes)
def initialPopulation(popSize, cityList, specialInitialSolutions):
    population = []
    
    population += specialInitialSolutions

    numberInitialSolutions = len(specialInitialSolutions)
    print ("Number of special initial solutions:" + str(numberInitialSolutions))
    #for i in range(0, popSize):
    for i in range(numberInitialSolutions, popSize):
        population.append(createRoute(cityList))
    return population

#Create the genetic algorithm
#Rank individuals
# use 1 = DISTANCE BASED RANKING
# use 2 = STRESS BASED RANKING
# use 3 = PARETO FITNESS BASED RANKING
def rankRoutes(population, objectiveNrUsed):
    fitnessResults = {}
    if (objectiveNrUsed == 1):
        for i in range(0,len(population)):
            fitnessResults[i] = Fitness(population[i]).routeFitnessDistanceBased()
    elif (objectiveNrUsed == 2):
        for i in range(0,len(population)):
            fitnessResults[i] = Fitness(population[i]).routeFitnessStressBased()
    elif (objectiveNrUsed == 3):
        fitnessResults = rankRoutesBasedOnDominance(population)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

#Provide Pareto-Based Fitness Calculation <<<<<<<<<<<<
# Dictionary bei dem für jedes Individuum die zugehörigen Werte gespeichert werden
# IndexNr:[distance,stress,[dominated solutions], [is dominated by solutions], R(i), F(i)]    
def rankRoutesBasedOnDominance(population):
    #store single fitness values per individuum
    fitnessValuesPerIndividuum = {}
    distance = 0
    stress = 0
    for i in range(0,len(population)):
        distance = Fitness(population[i]).routeDistance()
        stress = Fitness(population[i]).routeStress() 
        fitnessValuesPerIndividuum[i] = [distance, stress, [], [], 0, 0]
    #compute number of dominated solutions
    for i in range(0,len(population)):
       for j in range(0,len(population)):
           if (i != j):
              if (fitnessValuesPerIndividuum[i][0] < fitnessValuesPerIndividuum[j][0]
              and fitnessValuesPerIndividuum[i][1] < fitnessValuesPerIndividuum[j][1]):
                   fitnessValuesPerIndividuum[i][2].append(j) #add dominated solution
                   fitnessValuesPerIndividuum[j][3].append(i) #add dominating solution
    for i in range(0,len(population)):
        for domSol in fitnessValuesPerIndividuum[i][3]:
            fitnessValuesPerIndividuum[i][4] += len(fitnessValuesPerIndividuum[domSol][2])
    distanceValuesPerIndividuum = {}
    for i in range(0,len(population)):
        distanceValuesPerIndividuum[i] = []
        for j in range(0,len(population)):
           if (i != j):
               euclDistance = computeEuclideanDistance(fitnessValuesPerIndividuum[i][0],fitnessValuesPerIndividuum[j][0],
                                        fitnessValuesPerIndividuum[i][1], fitnessValuesPerIndividuum[j][1])
               distanceValuesPerIndividuum[i].append(euclDistance)
        distanceValuesPerIndividuum[i].sort()
    #determine k-nearest neighbour    
    k = int(np.floor((np.sqrt(len(population)))))
    if (k == 0):
        print("Something went wrong.")
        k = 1
    #index der Distanzberechnung, kter-Nachbar
    k -=1
    
    fitnessResults = {}
    for i in range(0,len(population)):
        #compute D(i)
        d_i = 1/ (distanceValuesPerIndividuum[i][k] + 2)
        fitnessValuesPerIndividuum[i][5] = fitnessValuesPerIndividuum[i][4] + d_i
        fitnessResults[i] = 1/fitnessValuesPerIndividuum[i][5] #damit größte Fitness = beste
    return fitnessResults 
  
def computeEuclideanDistance(distanceA, distanceB, stressA, stressB):
    return np.sqrt( (distanceA-distanceB)** 2 + (stressA-stressB)** 2)

#Create a selection function that will be used to make the list of parent routes
def selection(popRanked, eliteSize):
    selectionResults = []
    # roulette wheel by calculating a relative fitness weight for each individual
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    #We’ll also want to hold on to our best routes, so we introduce elitism
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    if selectionMethod == 1: # FITNESSBASIERT
        #we compare a randomly drawn number to these weights to select our mating pool
        for i in range(0, len(popRanked) - eliteSize):
            pick = 100*random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i,3]:
                    selectionResults.append(popRanked[i][0])
                    break
    if selectionMethod == 2:
        while len(selectionResults)<len(popRanked):
            competition = random.sample(popRanked, k)
            winner = competition[0]
            for i in range(0,k):
                if competition[i][1] > winner[1]:
                    winner = competition[i]
            selectionResults.append(winner[0])

    return selectionResults
    
def selectionWithArchive(popRanked):
    selectionResults = []

    # roulette wheel by calculating a relative fitness weight for each individual
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    #We’ll also want to hold on to our best routes, so we introduce elitism
    #here wie hold all non-dominated solutions
    #TODO (optional): ein festes Archiv vorsehen wie es im ursprünglichen SPEA2 vorgesehen ist 
    # for i in range(0, len(popRanked)):
    #     if (popRanked[i][1] > 1):
    #         selectionResults.append(popRanked[i][0])
    # currentArchiveSize = len(selectionResults)

    if selectionMethod == 1: # FITNESSBASIERT
        #we compare a randomly drawn number to these weights to select our mating pool
        for i in range(0, len(popRanked) - archiveSize):
            pick = 100*random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i,3]:
                    selectionResults.append(popRanked[i][0])
                    break
    if selectionMethod == 2: # TURNIERBASIERT
        print(selectionResults)
        while len(selectionResults)<(len(popRanked)-archiveSize):
            competition = random.sample(popRanked, k)
            winner = competition[0]
            for i in range(0,k):
                if competition[i][1] > winner[1]:
                    winner = competition[i]
            selectionResults.append(winner[0])

    return selectionResults

#Create mating pool
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        print(index)
        matingpool.append(population[index])
    return matingpool

def matingPoolWithArchive(population, selectionResults, archive):
    matingpool = archive
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

# Create a crossover function for two parents to create one child
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    #In ordered crossover, we randomly select a subset of the first parent string
    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    #and then fill the remainder of the route with the genes from the second parent
    #in the order in which they appear, 
    #without duplicating any genes in the selected subset from the first parent      
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

#Create function to run crossover over full mating pool
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    #we use elitism to retain the best routes from the current population.
    for i in range(0,eliteSize):
        children.append(matingpool[i])

    #we use the breed function to fill out the rest of the next generation.    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def breedPopulationWithArchive(matingpool, archive):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    #we use elitism to retain the best routes from the current population.
    for i in range(0,eliteSize):
        children.append(matingpool[i])

    #we use the breed function to fill out the rest of the next generation.    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#Create function to mutate a single route
#we’ll use swap mutation.
#This means that, with specified low probability, 
#two cities will swap places in our route.
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#Create function to run mutation over entire population
def mutatePopulation(population, mutationRate, eliteSize):
    mutatedPop = []
    
    #mating pool is sorted in order of fitness
    #here elitism instead of fixed archive
    #TODO (optional): ein festes Archiv vorsehen wie es im ursprünglichen SPEA2 vorgesehen ist 
    for ind in range(0, eliteSize):
        mutatedPop.append(population[ind])
    for ind in range(eliteSize, len(population)):
    #for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# def determineNonDominatedArchive(currentGen, popRanked):
#     archive = []
#     for i in range(0, len(popRanked)):
#         if (popRanked[i][1] > 1):
#             archive.append(currentGen[popRanked[i][0]])
#     sameSolutions = []
#     for i in range(0, len(archive)-1):
#         for j in range(i+1, len(archive)):
#             if isSameSolution(archive[i], archive[j]):
#                 sameSolutions.append(j)
#     newArchive = []
#     for i in range(0, len(archive)):
#         if (not sameSolutions.__contains__(i)):
#             newArchive.append(archive[i])
#     return newArchive

def determineNonDominatedArchive(currentGen, popRanked, oldArchive):
    archive = []
    for i in range(0, len(oldArchive)):
         if (popRanked[oldArchive[i]][1] > 1):
             archive.append(currentGen[popRanked[oldArchive[i]][0]])

    nonDominated = []
    for i in range(0, len(popRanked)):
        if (popRanked[i][1] > 1):
            for j in range(0,len(archive)):
                if isSameSolution(currentGen[popRanked[i][0]],archive[j]):
                    break
                else:
                    archive.append(popRanked[i][0])
        else:
            nonDominated.append(currentGen[popRanked[i][0]])
    #-------Prüfung auf Gleichheit bei Bedarf auskommentieren 
    # sameSolutions = []
    # for i in range(0, len(archive)-1):
    #     for j in range(i+1, len(archive)):
    #         if isSameSolution(archive[i], archive[j]):
    #             sameSolutions.append(j)
    # newArchive = []
    # for i in range(0, len(archive)):
    #     if (not sameSolutions.__contains__(i)):
    #         newArchive.append(archive[i])
    if len(archive) < archiveSize:
        fillUp = archiveSize-len(archive)
        archive.append(random.sample(nonDominated,fillUp))
        # while fillUp != 0:
        #     while True:
        #         newMember = random.sample(popRanked,1)
        #         for j in range(0,len(archive)):
        #             if isSameSolution(newMember,archive[j]):
        #                 break
        #             if j == len(archive)-1:
        #                 archive.append(newMember)
        #         if fillUp > archiveSize-len(archive):
        #             fillUp -= 1
        #             break
    if len(archive) > archiveSize:
        del archive[archiveSize:len(archive)]

    return archive

def determineNonDominatedArchiveSize(popRanked):
    archiveSize = 0
    for i in range(0, len(popRanked)):
        if (popRanked[i][1] > 1):
            archiveSize += 1
    return archiveSize

def isSameSolution(individuumA, individuumB):
    length = len(individuumA)
    i = 0
    isSameSolution = True
    while i < length and isSameSolution:
        if (not (individuumA[i].nr == individuumB[i].nr)):
            isSameSolution = False
            break
        i+=1
    return isSameSolution
    

#Put all steps together to create the next generation
 
#First, we rank the routes in the current generation using rankRoutes.
#We then determine our potential parents by running the selection function,
#    which allows us to create the mating pool using the matingPool function.
#Finally, we then create our new generation using the breedPopulation function 
# and then applying mutation using the mutatePopulation function. 

def nextGeneration(currentGen, eliteSize, mutationRate, objectiveNrUsed, archiveUsed,oldArchive): 
   # rankRoutesBasedOnDominance(currentGen)
    popRanked = rankRoutes(currentGen,objectiveNrUsed)

    if (not archiveUsed):
        selectionResults = selection(popRanked, eliteSize)
        matingpool = matingPool(currentGen, selectionResults)
        children = breedPopulation(matingpool, eliteSize)
        nextGeneration = mutatePopulation(children, mutationRate,0)
    else:
        #<<<<< use archiv
        #TODO (optional): ein festes Archiv vorsehen wie es im ursprünglichen SPEA2 vorgesehen ist 
        archive = determineNonDominatedArchive(currentGen, popRanked, oldArchive)
        selectionResults = selectionWithArchive(popRanked)
        matingpool = matingPool(currentGen, selectionResults)
        #archiveSize = determineNonDominatedArchiveSize(popRanked)
        children = breedPopulation(matingpool, eliteSize)
        #eliteSize is used to maintain solutions that should be in an archive
        nextGeneration = mutatePopulation(children, mutationRate, archive)
    return nextGeneration

#Final step: create the genetic algorithm
def plotPopulationAndObjectiveValues(population,title):
    distance = []
    stress = []
    for route in population:
        distance.append(Fitness(route).routeDistance())
        stress.append(Fitness(route).routeStress())
    plt.scatter(distance,stress,marker = "o",color="black")
    plt.ylabel('Stress')
    plt.xlabel('Distance')
    plt.title(title)
    plt.show()
        

def geneticAlgorithm(objectiveNrUsed, specialInitialSolutions, population, popSize, eliteSize, mutationRate, generations):
    #create initial population
    pop = initialPopulation(popSize, population, specialInitialSolutions)
    archiveUsed = False
    
    #provide statistics about best initial solution with regard to chosen objective
    if (objectiveNrUsed == 1 or objectiveNrUsed == 2):
        print("Initial objective: " + str(1 / rankRoutes(pop,objectiveNrUsed)[0][1]))
        bestRouteIndex = rankRoutes(pop,objectiveNrUsed)[0][0]
        bestRoute = pop[bestRouteIndex]
        print("Initial distance : " + str(Fitness(bestRoute).routeDistance()))
        print("Initial stress:    " + str(Fitness(bestRoute).routeStress()))
        plotRoute(bestRoute, "Best initial route")
    elif(objectiveNrUsed == 3):
        print("Initial highest fitness value: " + str(rankRoutes(pop,objectiveNrUsed)[0][1]))
        print("Initial best distance value: " + str(1/ rankRoutes(pop,1)[0][1]))
        print("Initial best stress value: " + str(1/ rankRoutes(pop,2)[0][1]))
        archiveUsed = True
    
    #plot intial population with regard to the two objectives
    plotPopulationAndObjectiveValues(pop, "Initial Population")
    
    #store infos to plot progress when finished
    progressDistance = []
    progressDistance.append(1 / rankRoutes(pop,1)[0][1])
    progressStress = []
    progressStress.append(1 / rankRoutes(pop,2)[0][1])
    
    archive = []
    
    #create new generations of populations
    for i in range(0, generations):
        print(i, end=", ")
        pop = nextGeneration(pop, eliteSize, mutationRate,objectiveNrUsed,archiveUsed,archive)
        #store infos to plot progress when finished
        progressDistance.append(1 / rankRoutes(pop,1)[0][1])
        progressStress.append(1 / rankRoutes(pop,2)[0][1])
    print("Done!")
        
    #plot progress - distance
    plt.plot(progressDistance)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.title('Progress of Distance Minimization')
    plt.show()
    #plot progress - stress
    plt.plot(progressStress)
    plt.ylabel('Stress')
    plt.xlabel('Generation')
    plt.title('Progress of Stress Minimization')
    plt.show()
    
    #provide statistics about best final solution with regard to chosen objective
    if (objectiveNrUsed == 1 or objectiveNrUsed == 2):
        print("Final objective: " + str(1 / rankRoutes(pop,objectiveNrUsed)[0][1])) 
        bestRouteIndex = rankRoutes(pop,objectiveNrUsed)[0][0]
        bestRoute = pop[bestRouteIndex]
        print("Final distance : " + str(Fitness(bestRoute).routeDistance()))
        print("Final stress:    " + str(Fitness(bestRoute).routeStress()))
        
        #Provide special initial solutions    <<<<<<<<<<<
        #print city Indizes for initial solution
        bestRouteIndizes = []
        for city in bestRoute:
            bestRouteIndizes.append(city.nr)
    
        print("---- ")
        print("City Numbers of Best Route")
        print(bestRouteIndizes)
        print("---- ")
        plotRoute(bestRoute, "Best final route")
        
    elif(objectiveNrUsed == 3):
        print("Final highest fitness value: " + str(rankRoutes(pop,objectiveNrUsed)[0][1]))
        print("Final best distance value: " + str(1/ rankRoutes(pop,1)[0][1]))
        print("Final best stress value: " + str(1/ rankRoutes(pop,2)[0][1]))
        bestRouteIndex = rankRoutes(pop,objectiveNrUsed)[0][0]
        bestRoute = pop[bestRouteIndex]
        paretoFront = determineNonDominatedArchive(pop, rankRoutes(pop,objectiveNrUsed))
        #TODO (optional): ein festes Archiv vorsehen wie es im ursprünglichen SPEA2 vorgesehen ist
        # dann alle Lösungen ausgeben die im Archiv sind
        
        
    #plot final population with regard to the two objectives
    plotPopulationAndObjectiveValues(paretoFront, "Final Non-Dominated Solutions")
    

    
    return bestRoute

#Running the genetic algorithm
#Create list of cities
cityList = []

random.seed(1111)
for i in range(1,26):
    cityList.append(City(nr= i, traffic=int(random.random()*40), x=int(random.random() * 200), y=int(random.random() * 200)))

# Make one calculation for all distances/stresses and store them in arrays
distances = [[toCity.distance(fromCity) for toCity in cityList] for fromCity in cityList]
stresses = [[toCity.stress(fromCity) for toCity in cityList] for fromCity in cityList]
    
print(cityList)


def plotRoute(cityList, title):
    x = []
    y = []
    for item in cityList:
        x.append(item.x)
        y.append(item.y)
        plt.annotate(item.nr,(item.x,item.y))
    x.append(cityList[0].x)
    y.append(cityList[0].y)
    plt.plot(x,y,marker = "x")
    plt.ylabel('Y-Coordinate')
    plt.xlabel('X-Coordinate')
    plt.title(title)
    plt.show()    

def getCityBasedOnNr(cityList,nr):
    if (nr <= 0 or nr > len(cityList)):
        print("Something is wrong!")
        return cityList[0]
    else:
        return cityList[nr-1]     
    
#Provide special initial solutions     <<<<<<<<<<<
cityNumbersRoute1 = [10,19,12,14,22,4,1,15,7,2,11,5,16,23,6,17,20,25,9,18,13,3,24,21,8] #mindist
cityNumbersRoute2 = [1, 19, 12, 4, 24, 8, 16, 11, 17, 15, 5, 10, 20, 2, 14, 7, 21, 3, 18, 9, 23, 22, 13, 6, 25] #minstress


route1 = []
for nr in cityNumbersRoute1:
    route1.append(getCityBasedOnNr(cityList,nr))

route2 = []
for nr in cityNumbersRoute2:
    route2.append(getCityBasedOnNr(cityList,nr))
    

initialSolutionsList = []
initialSolutionsList.append(route1)
initialSolutionsList.append(route2)

# 1 = fitnessbasiert
# 2 = turnierbasiert
selectionMethod = 2
# Parameter k for contestants in competition
k = 50

archiveSize = 20

#Run the genetic algorithm
#modify parameters popSize, eliteSize, mutationRate, generations to search for the best solution
#modify objectiveNrUsed to use different objectives:
# 1= Minimize distance, 2 = Minimize stress, 3 = MinimizeBoth
#bestRoute = geneticAlgorithm(objectiveNrUsed=3, specialInitialSolutions = initialSolutionsList, population=cityList,
#                             popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
bestRoute = geneticAlgorithm(objectiveNrUsed=3, specialInitialSolutions = initialSolutionsList, population=cityList,
                             popSize=500, eliteSize=archiveSize, mutationRate=0.001, generations=1000)
print(bestRoute)

plotRoute(bestRoute, "Best final route")
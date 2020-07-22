import aadioptimize
from aadioptimize import funcs
from funcs import rosenbrock

N_p = 25 #Number of population
T = 500 #number of iterations

lb =-100 #searchspace lower bound
ub =100  #searchspace upper bound

prob = rosenbrock #the fitness function
N_vars=2 #Number of search variables from fitness function

#Paramaters for adaptive  Scaling FACTOR F (between 0-2)
F_min=0 #minimum of Scaling Factor
F_max=2   #maximum of Scaling Factor
F_const=0.5 # Scaling factor should be constant if only 1-2 dimensonal function. Recomended 0.5


#Paramaters for adaptive crossover probability
P_c_min=0.5
P_c_max=0.1

#THe function aadioptimiz is the function itself it takes arguments 
#in same order as the problem settings. It returns best fitness function value and the 
#optimum location
best_of_f, globopt = aadioptimize(N_p,T,lb,ub,prob,N_vars,F_min,F_max,F_const,P_c_min,P_c_max)


#Found optimization location
found_loc =globopt
#Known location
known_loc=[1,1]
#distance from known location

print(f'Distance between found and known global optimum loc.: {distance(known_loc,found_loc,N_vars)}')
print(f'Global optimum location at: {found_loc}')
print(f'Best fitness function value found was:{best_of_f}')
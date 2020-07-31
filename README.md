# aadioptimize
An optimization utility for hyperparamaters.


## Installation

```bash
pip install https://github.com/aaditep/aadioptimize.git
```
## Usage

```python
from aadioptimize.optimize import main
from funcs.funcs import rosenbrock  
from aadioptimize.optimize import distance               
import numpy as np

# User inputs are in order as follows: evaluation counter(N), Nr of population(N_p), Nr. of iterations(T), lower bound(lb), upper bounds(ub),
#optimization problem(prob), number of search variables(N_vars), scaling factor minimum bound(F_min), 
#constant scaling factor(F_const), adaptive crossover minimum (P_c_min), maximum(P_c_max).


#Paramaters for Differential Evolution
#These are set up for 507 evaluations
N =0 #number of evaluation counter
N_p = 22#Number of population
T = 23 #number of iteratons
#THE NUMBER OF EVALUATIONS IS (N*N_p+1). For example 22*23+1=507 evaluations

lb = -100 #searchspace lower bound
ub =100  #searchspace upper bound	

prob = rosenbrock #the fbitness function
N_vars=2 #Number of search variables from fitness function	

#Paramaters for adaptive  Scaling FACTOR F (between 0-1)
F_min=0.5 #minimum fot adaptive Scaling Factor
F_const=0.32  # Scaling factor should be constant if only 1-2 dimensonal function.

#Paramaters for adaptive crossover probability. Usually between 0-1. Since this example is for 507 evaluations then
#by trial and error these settings worked pretty well.
P_c_min=1.7
P_c_max=1.9
#THe function aadioptimiz is the function itself it takes arguments 
#in same order as the problem settings. It returns number of evaaluations
#best fitness function value and optimum location.
N,best_of_f, globopt = main(N,N_p,T,lb,ub,prob,N_vars,F_min,F_const,P_c_min,P_c_max)	


# A d
#Found optimization location
found_loc =globopt
#Known location
known_loc=[1,1]
#distance from known location	

print(f'Distance between found and known global optimum loc.: {distance(known_loc,found_loc,N_vars)}')
print(f'Global optimum location at: {found_loc}')
print(f'Best fitness function value found was:{best_of_f}')
print(f'Number of evaluations {N+1}')
```


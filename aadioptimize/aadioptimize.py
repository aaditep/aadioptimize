import numpy.matlib as mat
import numpy as np



def initDE(N_p,lb,ub,prob):
            lb = np.full(N_p,lb)
            
            ub = np.full(N_p,ub)
            
            f = np.zeros((N_p,1)) #empty vector for fitness function
            
            fu = np.zeros((N_p,1))#newly created trial vector

            D = len(lb) # Determining amount of decision variables
    
            U = np.zeros((N_p,D)) #Matrix for storing trial solutions 
        
            #Initial random population !!!!!MIND THE "LEN", I DONT KNOW if that works"
            P = mat.repmat(lb,N_p,1)+mat.repmat((ub-lb),N_p,1)*np.random.rand(len(ub-lb),N_p)
        
            for p in np.arange(N_p):
                f[p]=prob(P[p,])
            
            return lb,ub,f,fu,D,U,P





#This function starts the mutation process and generates a donorvector
def mutation1(i,N_p,t,T,P,N_vars,F_const):
            #Adaptive scaling factor
            if N_vars >= 3:
                F=F_min*2**np.exp(1-(T/(T+1-t)))
            else:
                F = F_const
            #candidates are assigned without the i-th element
            candidates= np.delete(np.arange(N_p), np.where(np.arange(N_p)==i))
            #3 target vectors are picked out randomly for the donorvector generator
            cand_rand=np.random.choice(candidates,3,replace= False)
            X1=P[cand_rand[0],]
            X2=P[cand_rand[1],]
            X3=P[cand_rand[2],]
       
            #Donorvctor generator
            V= X1 + F*(X2-X3)
            return V


#this function evaluates donor vector and uses parts of it which fit better
def crossover(f,P_c_min,P_c_max,i,D,V,P,U):
            #ADAPTIVE Crossover
            if f[i] < np.mean(f):
                P_c = P_c_min + (P_c_max-P_c_min)*((f[i]-np.mean(f))/(np.max(f)-np.mean(f)))
            else:
                P_c = P_c_min
        
            delta = np.random.randint(0,D-1) 
            for j in np.arange(D):
                if np.random.uniform(0,1) <= P_c or delta == j:
                    U[i,j] = V[j]
                else:
                    U[i,j]=P[i,j]
        
            return U

#this function bounds the vector and replaces the old target vector with new if better
def boundgreed(j,U,P,f,fu,ub,lb,prob):
            
            U[j]=np.minimum(U[j], ub)
            U[j]=np.maximum(U[j], lb)
    
            fu[j]=prob(U[j])

            if fu[j] < f[j]:
                P[j]= U[j]
                f[j]=fu[j]
            return fu,f,P

#distance from known location
def distance(known_loc,found_loc,N_vars,):
            known = known_loc
            opt = found_loc
            undersqrt=np.zeros(N_vars)
            for i in (np.arange(N_vars)):
                undersqrt[i]  =(known_loc[i]-found_loc[i])**2
                dist = np.sqrt(sum(undersqrt))
        
            return dist



def aadioptimize(N_p,T,lb,ub,prob,N_vars,F_min,F_max,F_const,P_c_min,P_c_max):
            lb,ub,f,fu,D,U,P = initDE(N_p,lb,ub,prob)
            if N_p < 4:
                raise Exception("Sorry, there must be atleast a population of 4. Reccomended 20")
            
            for t in np.arange(T):
                for i in np.arange(N_p):
        
                    V = mutation1(i,N_p,t,T,P,N_vars,F_const)
        
                    U=crossover(f,P_c_min,P_c_max,i,D,V,P,U)
    
                for j in np.arange(N_p):    
        
                    fu,f,P = boundgreed(j,U,P,f,fu,ub,lb,prob)
        
            best_of_f= min(f)
            globopt = P[f.argmin()]
            return best_of_f, globopt[:N_vars]
import unittest
from aadioptimize import optimize
from funcs import funcs
import numpy as np

class TestOptimize(unittest.TestCase):
	
	def test_distance(self):
		# if the distance calculator works at varieties of dimensions
		dist = optimize.distance([1,1],[1,1],2)
		self.assertEqual(dist, 0)
		dist = optimize.distance([1,1,0],[1,1,1],3)
		self.assertEqual(dist,1)
		dist = optimize.distance([-1,-1,-1],[0,-1,-1],3)
		self.assertEqual(dist,1)
		dist = optimize.distance([-1,-1,-1,-1,-1],[8,-1,-1,-1,-1],5)
		self.assertEqual(dist,9)



	def test_funcs(self):
		#If the functions itself are correct
		self.assertEqual(funcs.rosenbrock([1,1]),0)

		self.assertEqual(round(funcs.kenny([-0.54719,-1.54719]),3),-1.913)






	def test_initDE(self):
		#if the function retruns values with correct shapes and values
		np.random.seed(10)
		N_p = 5
		lb = -5
		ub= 5
		prob = funcs.rosenbrock
		lb,ub,f,fu,D,U,P = optimize.initDE(N_p,lb,ub,prob)
		



		#lowerbound and upperbound
		self.assertEqual(len(lb),5)
		self.assertEqual(len(ub),5)
		

		#fitness function
		self.assertIsNot(f,np.zeros)
		self.assertEqual(len(f),5)
		

		#trial fitness function
		self.assertEqual(len(fu),5)
		

		#Decision variables
		self.assertEqual(D,5)
		

		#U must be empty
		self.assertFalse(U.all())
		
		#P must be all True
		self.assertTrue(P.all())
		

		#Testing with random seed if the arrays are correct
		arr1 = P
		arr2 = np.array([[ 2.71320643, -4.79248051,  1.33648235,  2.48803883, -0.01492988],
       [-2.75203354, -3.01937135,  2.60530712, -3.30889163, -4.11660186],
       [ 1.85359818,  4.53393346, -4.96051734,  0.12192263,  3.12620962],
       [ 1.12526067,  2.21755317, -2.08123932,  4.17774123,  2.14575783],
       [ 0.42544368, -3.57829952, -1.2665924 ,  1.74133615, -0.58166826]])
		self.assertIsNone(np.testing.assert_almost_equal(arr1, arr2))
		

		#Testing fitness function values
		arr3 = f
		arr4 = np.array(
 		[[14774.83290809],
		[11235.36973715],
       	[  121.31257991],
       	[   90.52077474],
       	[ 1413.5651541 ]])
		self.assertIsNone(np.testing.assert_almost_equal(arr3, arr4))


	def test_mutation(self):
		np.random.seed(10)
		i = 5
		N_p =5
		t =5
		T=50
		N_vars = 4
		P =np.array([[ 2.71320643, -4.79248051,  1.33648235,  2.48803883, -0.01492988],
       [-2.75203354, -3.01937135,  2.60530712, -3.30889163, -4.11660186],
       [ 1.85359818,  4.53393346, -4.96051734,  0.12192263,  3.12620962],
       [ 1.12526067,  2.21755317, -2.08123932,  4.17774123,  2.14575783],
       [ 0.42544368, -3.57829952, -1.2665924 ,  1.74133615, -0.58166826]])
		F_min =0.5
		F_const = 0.7
		V= optimize.mutation(i,N_p,t,T,P,N_vars,F_min,F_const)
		self.assertEqual(len(V),5)



	def test_crossover(self):
		np.random.seed(10)
		f =np.array(
 		[[14774.83290809],
		[11235.36973715],
       	[  121.31257991],
       	[   90.52077474],
       	[ 1413.5651541 ]])
		P_c_min=0.5
		P_c_max=0.9
		i=4
		D=5
		V=[2,4,3,1.5,2.5]
		P =np.array([[ 2.71320643, -4.79248051,  1.33648235,  2.48803883, -0.01492988],
       [-2.75203354, -3.01937135,  2.60530712, -3.30889163, -4.11660186],
       [ 1.85359818,  4.53393346, -4.96051734,  0.12192263,  3.12620962],
       [ 1.12526067,  2.21755317, -2.08123932,  4.17774123,  2.14575783],
       [ 0.42544368, -3.57829952, -1.2665924 ,  1.74133615, -0.58166826]])
		
		U = np.zeros((5,5))
		U = optimize.crossover(f,P_c_min,P_c_max,i,D,V,P,U)
		self.assertFalse(U.all())
		

	
	def test_boundgreed(self):
		N_p =5
		N=5
		j=4
		
		U =np.array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 2.        ,  4.        , -1.2665924 ,  1.74133615, -0.58166826]])
		
		P =np.array([[ 2.71320643, -4.79248051,  1.33648235,  2.48803883, -0.01492988],
       [-2.75203354, -3.01937135,  2.60530712, -3.30889163, -4.11660186],
       [ 1.85359818,  4.53393346, -4.96051734,  0.12192263,  3.12620962],
       [ 1.12526067,  2.21755317, -2.08123932,  4.17774123,  2.14575783],
       [ 0.42544368, -3.57829952, -1.2665924 ,  1.74133615, -0.58166826]])
		
		f =np.array(
 		[[14774.83290809],
		[11235.36973715],
       	[  121.31257991],
       	[   90.52077474],
       	[ 1413.5651541 ]])
		
		fu =np.zeros((N_p,1))
		
		prob = funcs.rosenbrock
		
		lb = np.full(N_p,5)
    
		ub = np.full(N_p,5)

		N,f,P = optimize.boundgreed(N,j,U,P,f,fu,ub,lb,prob)

		self.assertEqual(len(f),5)
		self.assertEqual(len(P),5)

		self.assertTrue(P.any())


	def test_main(self):
		N=4
		N_p= 5
		T =50
		lb =5
		ub=5
		prob = funcs.rosenbrock
		N_vars =2
		F_min = 0.5
		F_const = 0.6
		P_c_min = 0.5
		P_c_max = 0.8
		N,best_f,globopt= optimize.main(N,N_p,T,lb,ub,prob,N_vars,F_min,F_const,P_c_min,P_c_max)

		self.assertEqual(len(best_f),1)
		self.assertEqual(len(globopt),2)
		


if __name__ == '__main__':
	unittest.main()
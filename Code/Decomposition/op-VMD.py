import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD
from scipy.signal import hilbert
from scipy.stats import entropy
from scipy.signal import find_peaks
from mealpy import BinaryVar, WOA, Problem,FloatVar, StringVar,IntegerVar
import numpy as np
from itertools import permutations
dataset = pd.read_excel(r'D:\Users\三木\Desktop\qingdao.xlsx', usecols=[ 1, 2, 3, 4, 5])
data = dataset['total'].values


class SvmOptimizedProblem(Problem):
    def __init__(self, pop=None, minmax="min", data=None):
        self.data = data
        super().__init__(pop, minmax)
    def obj_func(self,x):
        np.random.seed(0)
        x_decoded = self.decode_solution(x)
        K, alpha = x_decoded["k"], x_decoded["alpha"]
        # print(K,alpha)
        tau = 0
        DC = 0
        init = 1
        tol = 1e-7
        imf, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
        ##模糊熵
        se_imf = []
        for i in range(K):
            # en=envelope_entropy(imf.T[:, i], 500)
            peaks, _ = find_peaks(imf.T[:,i])
            distances = np.diff(peaks)  # 计算峰值之间的距离
            fuzzy_entropy = np.log(np.mean(distances)+0.1)  # 计算模糊熵
            se_imf.append(fuzzy_entropy)
            # se_imf.append(en)
        fit = sum(se_imf)/K
        return fit
pop = [
    IntegerVar(lb=3, ub=10,name="k"),
    IntegerVar(lb=1, ub=20,name="alpha")
]
problem = SvmOptimizedProblem(pop=pop, minmax="min", data=data)
model = WOA.OriginalWOA(epoch=10, pop_size=40)
model.solve(problem)
print(f"Best agent: {model.g_best}")
print(f"Best solution: {model.g_best.solution}")
print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")
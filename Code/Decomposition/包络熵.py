'''适应度函数,最小化各VMD分量的局部包络熵'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD
from scipy.signal import hilbert
from scipy.stats import entropy
import mealpy
def Baoluoshang(signal):
    # 计算信号的解析包络
    analytical_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytical_signal)

    # 将包络值归一化到0-1之间
    normalized_envelope = (amplitude_envelope - np.min(amplitude_envelope)) / (np.max(amplitude_envelope) - np.min(amplitude_envelope))

    # 计算包络熵
    return entropy(normalized_envelope)


def fitness(pop, data):
    np.random.seed(0)

    K = int(pop[0])
    alpha = int(pop[1])
    # print(K,alpha)
    tau = 0
    DC = 0
    init = 1
    tol = 1e-7
    imf, res, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    comp = np.vstack([imf, res.reshape(1, -1)])
    SE = 0
    se_imf = []
    for i in range(comp.shape[0]):
        temp = Baoluoshang(comp[i, :])
        SE += temp
        se_imf.append(temp)
    # fit = SE
    # fit = SE/K
    fit = min(se_imf)
    # np.random.seed(int(time.time()))
    return fit





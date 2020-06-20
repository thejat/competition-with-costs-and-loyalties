# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2019  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np






def get_y0(u, phi, nums_a):
    ## linear system of equations for each player
    
    num_s, num_p, num_a_max = u.shape[0:3]
    strategyAxes = tuple(np.arange(1, 1+num_p))
    
    ## strategies: all players randomize uniformly
    beta = np.nan * np.ones(nums_a.sum(), dtype=np.float64)
    for s in range(num_s):
        for p in range(num_p):
            for a in range(nums_a[s,p]):
                beta[nums_a.ravel()[:s*num_p+p].sum() + a] = np.log(1/nums_a[s,p])
    
    ## state values: solve linear system of equations for each player
    V = np.nan * np.ones(num_s*num_p, dtype=np.float64)
    for p in range(num_p):
        A = np.identity(num_s) - np.nanmean(phi[:,p], axis=strategyAxes)
        b = np.nanmean(u[:,p], axis=strategyAxes)
        mu_p = np.linalg.solve(A, b)
        for s in range(num_s):
            V[s*num_p+p] = mu_p[s]
    
    y0 = np.concatenate([beta, V, [0.0]])
    
    if np.isnan(y0).any():
        print('Error: Linear system of equations could not be solved.')
        return False, y0
    
    else:
        return True, y0






## ============================================================================
## End of script
## ============================================================================
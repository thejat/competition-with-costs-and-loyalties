# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2019  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np






def symmetryPairs(H_test, T_y2beta, deltas, num_s, num_p, nums_a):
    
    ## use probabilistic detection with homotopy evaluated at random test vector
    ## requires actions to be in the same order!
    
    ## check (s1,p1) for symmetry with any other (s2,p2>p1) and (s2>s1,p2=p1)
    ## symmetric if 
        ## same discount factors AND
        ## symmetric payoffs AND
        ## symmetric transition probabilities AND
        ## p1 and p2 symmetric across all states
    
    H_strat = np.einsum('spaN,N->spa', T_y2beta, H_test[:nums_a.sum()])
    H_val = H_test[nums_a.sum():].reshape((num_s,num_p))
    
    symPairs = []
    
    for p1 in range(num_p):
        for p2 in range(p1+1, num_p):
            
            ## pre-check for equal discount factors:
            if deltas[p1] == deltas[p2]:
                
                symPairs_temp = []
                
                for s1 in range(num_s):
                    for s2 in range(num_s):
                        
                        H_strat1 = H_strat[s1,p1]
                        H_strat2 = H_strat[s2,p2]
                        
                        H_val1 = H_val[s1,p1]
                        H_val2 = H_val[s2,p2]
                        
                        ## check for payoff symmetry and transition symmetry 
                        ## (and for equal discount factors)
                        if np.allclose(np.sort(H_strat1.ravel()), np.sort(H_strat2.ravel()), equal_nan=True) and np.allclose(H_val1, H_val2):
                            symPairs_temp.append( ((s1,p1),(s2,p2)) )
            
            ## check whether p1 and p2 are symmetric across all states
            if len(symPairs_temp) == num_s:
                symPairs.extend(symPairs_temp)
    
    
    return symPairs






def reduction_expansion_helpers(sym_pairs, num_s, num_p, nums_a):
    
    len_H = nums_a.sum() + num_s*num_p
    
    def sigma_index(s,p,a): 
        return nums_a.flatten()[:s*num_p + p].sum() + a
    
    def V_index(s,p):
        return nums_a.sum() + s*num_p + p
    
    state_players_reduced = [(s,p) for s in range(num_s) for p in range(num_p)]
    for ((s1,p1),(s2,p2)) in sym_pairs:
        state_players_reduced.remove((s2,p2))
    len_y_reduced = 1 + np.sum([nums_a[s,p] + 1 for (s,p) in state_players_reduced])
    
    
    ## array of indices to reduce H
    idx_reduce = np.sort(np.array(
            [sigma_index(s,p,a) for (s,p) in state_players_reduced for a in range(nums_a[s,p])] + \
            [V_index(s,p) for (s,p) in state_players_reduced],
            dtype=np.int32))
    
    
    ## array of indices to expand y
    idx_expand = np.zeros(len_H+1, dtype=np.int32)
    for (s,p) in state_players_reduced:
        for a in range(nums_a[s,p]):
            idx_expand[sigma_index(s,p,a)] = np.where(idx_reduce == sigma_index(s,p,a))[0][0]
        idx_expand[V_index(s,p)] = np.where(idx_reduce == V_index(s,p))[0][0]
    for ((s1,p1),(s2,p2)) in sym_pairs:
        for a in range(nums_a[s1,p1]):
            idx_expand[sigma_index(s2,p2,a)] = np.where(idx_reduce == sigma_index(s1,p1,a))[0][0]
        idx_expand[V_index(s2,p2)] = np.where(idx_reduce == V_index(s1,p1))[0][0]
    idx_expand[-1] = len_y_reduced - 1
    
    
    ## matrix for right-multiplication with J
        ## 1) reduce columns of J 
        ## 2) sum up reduced columns in J to 
    J_post_reducer = np.eye(len_H+1, dtype=np.int32)
    mask = np.ones(len_H+1, dtype=bool)
    for ((s1,p1),(s2,p2)) in sym_pairs:
        for a in range(nums_a[s2,p2]):
            mask[sigma_index(s2,p2,a)] = False
            J_post_reducer[sigma_index(s2,p2,a), sigma_index(s1,p1,a)] = 1
        mask[V_index(s2,p2)] = False
        J_post_reducer[V_index(s1,p1), V_index(s2,p2)] = 1 
    J_post_reducer = J_post_reducer[:, mask]
    
    
    symmetry_helpers = {
            'idx_reduce': idx_reduce,
            'idx_expand': idx_expand,
            'J_post_reducer': J_post_reducer
            }
            
    return symmetry_helpers






def H_reduced(y_reduced, H_full, symmetry_helpers):
    y_expanded = y_reduced[symmetry_helpers['idx_expand']]
    H_expanded = H_full(y_expanded)
    H_reduced = H_expanded[symmetry_helpers['idx_reduce']]
    return H_reduced


def J_reduced(y_reduced, J_full, symmetry_helpers):
    y_expanded = y_reduced[symmetry_helpers['idx_expand']]
    J_expanded = J_full(y_expanded)
    J_reduced = np.dot(J_expanded[symmetry_helpers['idx_reduce']], symmetry_helpers['J_post_reducer'])
    return J_reduced






## ============================================================================
## end of file
## ============================================================================
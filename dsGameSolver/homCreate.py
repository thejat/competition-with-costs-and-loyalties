# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2019  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np
import string






def T_y2beta(num_s, num_p, nums_a):
    
    ## some helpers
    num_a_max = nums_a.max()
    num_a_tot = nums_a.sum()
    
    ## array to extract beta[s,p,a] from y[:num_a_tot]
    T_y2beta = np.zeros(shape=(num_s,num_p,num_a_max,num_a_tot), dtype=np.float64)
    flat_index = 0
    for s in range(num_s):
        for p in range(num_p):
            for a in range(nums_a[s,p]):
                T_y2beta[s,p,a,flat_index] = 1
                flat_index += 1
            for a in range(nums_a[s,p],num_a_max):
                T_y2beta[s,p,a] = np.nan
    
    return T_y2beta




def homotopy_helpers(num_s, num_p, nums_a, phi_withoutNaN):
    
    ## some helpers
    num_a_max = nums_a.max()
    
    
    ## indices to mask H and J according to nums_a
    
    H_mask = []
    flat_index = 0
    for s in range(num_s):
        for p in range(num_p):
            for a in range(nums_a[s,p]):
                H_mask.append(flat_index)
                flat_index += 1
            flat_index += num_a_max - nums_a[s,p]
    for s in range(num_s):
        for p in range(num_p):
            H_mask.append(flat_index)
            flat_index += 1
    H_mask = np.array(H_mask, dtype=np.int32)
    
    J_mask = tuple(np.meshgrid(H_mask, np.append(H_mask, [num_s*num_p*num_a_max+num_s*num_p]), indexing='ij', sparse=True))
    
    
    ## arrays to assemble H
    
    T_H_0 = np.zeros(shape=(num_s,num_p,num_a_max), dtype=np.float64)
    for s in range(num_s):
        for p in range(num_p):
            T_H_0[s,p,0] = 1
    
    T_H_1 = np.zeros(shape=(num_s,num_p,num_a_max,num_s,num_p,num_a_max), dtype=np.float64)
    for s in range(num_s):
        for p in range(num_p):
            T_H_1[s,p,0,s,p,:] = -1
    
    T_H_2 = np.zeros(shape=(num_s,num_p,num_a_max,num_s,num_p,num_a_max), dtype=np.float64)
    for s in range(num_s):
        for p in range(num_p):
            for a in range(1,nums_a[s,p]):
                T_H_2[s,p,a,s,p,a] = -1
                T_H_2[s,p,a,s,p,0] = 1
    
    T_H_3 = np.zeros(shape=(num_s,num_p,num_s,num_p), dtype=np.float64)
    for s in range(num_s):
        for p in range(num_p):
            T_H_3[s,p,s,p] = -1
    
    T_H = {0: T_H_0,
           1: T_H_1,
           2: T_H_2,
           3: T_H_3}
    
    
    ## arrays to assemble J
    
    T_J_temp = np.zeros(shape=(num_s,num_p,num_a_max,num_s,num_p,num_a_max), dtype=np.float64)
    for s in range(num_s):
        for p in range(num_p):
            for a in range(nums_a[s,p]):
                T_J_temp[s,p,a,s,p,a] = 1
    
    T_J_0 = np.einsum('spatqb,tqbSPA->spaSPA', T_H_2, T_J_temp)
    
    T_J_1 = np.einsum('spatqb,tqbSPA->spaSPA', T_H_1, T_J_temp)
    
    T_J_temp = np.zeros(shape=(num_s,num_p,num_s,num_p), dtype=np.float64)
    for s in range(num_s):
        for p in range(num_p):
            T_J_temp[s,p,s,p] = 1
    
    T_J_3 = np.einsum('sp...t,tpSP->sp...SP', phi_withoutNaN, T_J_temp)
    
    T_J_5 = np.einsum('sptq,tqSP->spSP', T_H_3, T_J_temp)
    
    T_J_2 = np.zeros(shape=(num_s,num_p,*[num_a_max]*(num_p-1),num_s,num_p,num_a_max), dtype=np.float64)
    for s in range(num_s):
        for p in range(num_p):
            a_profiles_without_p = list(np.ndindex(tuple(nums_a[s,:p])+tuple(nums_a[s,(p+1):])))
            for A in a_profiles_without_p:
                for p_ in range(num_p):
                    if p_ != p:
                        a_ = A[p_] if p_ < p else A[p_-1]
                        T_J_2[(s,p)+A+(s,p_,a_)] = 1
    
    T_J_4 = np.zeros(shape=(num_s,num_p,*[num_a_max]*num_p,num_s,num_p,num_a_max), dtype=np.float64)
    for s in range(num_s):
        for p in range(num_p):
            a_profiles = list(np.ndindex(tuple(nums_a[s,:])))
            for A in a_profiles:
                for p_ in range(num_p):
                    T_J_4[(s,p)+A+(s,p_,A[p_])] = 1
    
    T_J = {0: T_J_0,
           1: T_J_1,
           2: T_J_2,
           3: T_J_3,
           4: T_J_4,
           5: T_J_5}
    
    
    ## equations for einsum
    einsum_eqs = {
            'sigma_prod': 's' + ',s'.join(string.ascii_uppercase[0:num_p]) + '->s' + string.ascii_uppercase[0:num_p],
            'sigma_prod_with_p': ['s' + ',s'.join(string.ascii_uppercase[0:num_p]) + '->s' + string.ascii_uppercase[0:num_p] for p in range(num_p)],
            'Eu_tilde_a_H': ['s' + string.ascii_uppercase[0:num_p] + ',s' + ',s'.join([string.ascii_uppercase[p_] for p_ in range(num_p) if p_ != p]) + '->s' + string.ascii_uppercase[p] for p in range(num_p)],
            'Eu_tilde_a_J': ['s' + string.ascii_uppercase[0:num_p] + ',s' + string.ascii_uppercase[0:num_p] + '->s' + string.ascii_uppercase[p] for p in range(num_p)],
            'dEu_tilde_a_dbeta': ['s' + string.ascii_uppercase[0:num_p] + ',s' + ''.join([string.ascii_uppercase[p_] for p_ in range(num_p) if p_ != p]) + 'tqb->s' + string.ascii_uppercase[p] + 'tqb' for p in range(num_p)],
            'dEu_tilde_a_dV': ['s' + string.ascii_uppercase[0:num_p] + 'tp,s' + string.ascii_uppercase[0:num_p] + '->s' + string.ascii_uppercase[p] + 'tp' for p in range(num_p)],
            'dEu_tilde_dbeta': 'sp' + string.ascii_uppercase[0:num_p] + ',sp' + string.ascii_uppercase[0:num_p] + 'tqb->sptqb'}
    
    
    homotopy_helpers = {
            'H_mask': H_mask,
            'J_mask': J_mask,
            'T_H': T_H,
            'T_J': T_J,
            'einsum_eqs': einsum_eqs
            }
    
    return homotopy_helpers






def H_qre(y, u_withoutNaN, phi_withoutNaN, num_s, num_p, num_a_tot, 
          T_y2beta, homotopy_helpers):
    
    H_mask = homotopy_helpers['H_mask']
    T_H = homotopy_helpers['T_H']
    einsum_eqs = homotopy_helpers['einsum_eqs']
    
    ## extract log-strategies beta, state values V and homotopy parameter gamma from y
    beta_withNaN = np.einsum('spaN,N->spa', T_y2beta, y[:num_a_tot])
    V = y[num_a_tot:-1].reshape((num_s,num_p))
    gamma = y[-1]
    
    
    ## generate building blocks of H
    
    sigma = np.exp(beta_withNaN)
    sigma[np.isnan(sigma)] = 0
    beta = beta_withNaN
    beta[np.isnan(beta)] = 0
    
    sigma_p_list = [sigma[:,p,:] for p in range(num_p)]
    u_tilde = u_withoutNaN + np.einsum('sp...S,Sp->sp...', phi_withoutNaN, V)
    
    if num_p > 1:
        Eu_tilde_a = []
        for p in range(num_p):
            Eu_tilde_a.append( np.einsum(einsum_eqs['Eu_tilde_a_H'][p], u_tilde[:,p], *(sigma_p_list[:p]+sigma_p_list[(p+1):])) )
        Eu_tilde_a = np.stack(Eu_tilde_a, axis=1)
    else:
        Eu_tilde_a = u_tilde
    
    Eu_tilde = np.einsum('spa,spa->sp', sigma, Eu_tilde_a)
    
    
    ## assemble H
    
    H_strat = T_H[0]                                                \
        + np.einsum('spaSPA,SPA->spa', T_H[1], sigma)               \
        + np.einsum('spaSPA,SPA->spa', T_H[2], beta)                \
        + gamma * np.einsum('spaSPA,SPA->spa', -T_H[2], Eu_tilde_a)
    
    H_val = np.einsum('spSP,SP->sp', T_H[3], V)       \
        + np.einsum('spSP,SP->sp', -T_H[3], Eu_tilde)
    
    H = np.append(H_strat.ravel(), H_val.ravel())[H_mask]
    
    return H






def J_qre(y, u_withoutNaN, phi_withoutNaN, num_s, num_p, num_a_max, num_a_tot, 
          T_y2beta, homotopy_helpers):
    
    J_mask = homotopy_helpers['J_mask']
    T_H = homotopy_helpers['T_H']
    T_J = homotopy_helpers['T_J']
    einsum_eqs = homotopy_helpers['einsum_eqs']
    
    ## extract log-strategies beta, state values V and homotopy parameter gamma from y
    beta_withNaN = np.einsum('spaN,N->spa', T_y2beta, y[:num_a_tot])
    V = y[num_a_tot:-1].reshape((num_s,num_p))
    gamma = y[-1]
    
    
    ## generate building blocks of J
    
    sigma = np.exp(beta_withNaN)
    sigma[np.isnan(sigma)] = 0
    beta = beta_withNaN
    beta[np.isnan(beta)] = 0
    sigma_p_list = [sigma[:,p,:] for p in range(num_p)]
    
    u_tilde = u_withoutNaN + np.einsum('sp...S,Sp->sp...', phi_withoutNaN, V)
    
    sigma_prod = np.einsum(einsum_eqs['sigma_prod'], *sigma_p_list)
    
    sigma_prod_with_p = []
    for p in range(num_p):
        sigma_p_list_with_p = sigma_p_list[:p] + [np.ones(shape=sigma[:,p,:].shape, dtype=np.float64)] + sigma_p_list[(p+1):]
        sigma_prod_with_p.append( np.einsum(einsum_eqs['sigma_prod_with_p'][p], *sigma_p_list_with_p) )
    sigma_prod_with_p = np.stack(sigma_prod_with_p, axis=1)
    
    
    if num_p > 1:
        Eu_tilde_a = []
        dEu_tilde_a_dbeta = []
        dEu_tilde_a_dV = []
        
        for p in range(num_p):
            Eu_tilde_a.append( np.einsum(einsum_eqs['Eu_tilde_a_J'][p], u_tilde[:,p], sigma_prod_with_p[:,p]) )
            dEu_tilde_a_dV.append( np.einsum(einsum_eqs['dEu_tilde_a_dV'][p], T_J[3][:,p], sigma_prod_with_p[:,p]) )
            
            T_temp = np.einsum('s...,s...->s...', u_tilde[:,p], sigma_prod_with_p[:,p])
            dEu_tilde_a_dbeta.append( np.einsum(einsum_eqs['dEu_tilde_a_dbeta'][p], T_temp, T_J[2][:,p]) )
        
        Eu_tilde_a = np.stack(Eu_tilde_a, axis=1)
        dEu_tilde_a_dbeta = np.stack(dEu_tilde_a_dbeta, axis=1)
        dEu_tilde_a_dV = np.stack(dEu_tilde_a_dV, axis=1)
    
    else:
        Eu_tilde_a = u_tilde
        dEu_tilde_a_dbeta = np.zeros(shape=(num_s,num_p,num_a_max,num_s,num_p,num_a_max), dtype=np.float64) 
        dEu_tilde_a_dV = T_J[3]
    
    
    T_temp = np.einsum('sp...,s...->sp...', u_tilde, sigma_prod)
    dEu_tilde_dbeta = np.einsum(einsum_eqs['dEu_tilde_dbeta'], T_temp, T_J[4])
    
    dEu_tilde_dV = np.einsum('spa,spaSP->spSP', sigma, dEu_tilde_a_dV)
    
    
    ## assemble J
    
    dH_strat_dbeta = T_J[0] + np.einsum('spaSPA,SPA->spaSPA', T_J[1], sigma)     \
        + gamma * np.einsum('spatqb,tqbSPA->spaSPA', -T_H[2], dEu_tilde_a_dbeta)
    dH_strat_dV = gamma * np.einsum('spatqb,tqbSP->spaSP', -T_H[2], dEu_tilde_a_dV)
    dH_strat_dlambda = np.einsum('spatqb,tqb->spa', -T_H[2], Eu_tilde_a)
    dH_val_dbeta = np.einsum('sptq,tqSPA->spSPA', -T_H[3], dEu_tilde_dbeta)
    dH_val_dV = T_J[5] + np.einsum('sptq,tqSP->spSP', -T_H[3], dEu_tilde_dV)
    dH_val_dlambda = np.zeros(shape=(num_s,num_p), dtype=np.float64)
    
    J = np.concatenate([
            np.concatenate([
                    dH_strat_dbeta.reshape((num_s*num_p*num_a_max,num_s*num_p*num_a_max)), 
                    dH_strat_dV.reshape((num_s*num_p*num_a_max,num_s*num_p)), 
                    dH_strat_dlambda.reshape((num_s*num_p*num_a_max,1))
                    ], axis=1),
            np.concatenate([
                    dH_val_dbeta.reshape((num_s*num_p,num_s*num_p*num_a_max)), 
                    dH_val_dV.reshape((num_s*num_p,num_s*num_p)), 
                    dH_val_dlambda.reshape((num_s*num_p,1))
                    ], axis=1)
            ], axis=0)[J_mask]
    
    return J






## ============================================================================
## end of file
## ============================================================================
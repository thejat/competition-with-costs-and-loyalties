# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2019  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np






def inputCorrect(payoffMatrices, transitionMatrices=None, discountFactors=0, symmetryPairs=[]):
    
    
    ## check payoffMatrices
    
    if type(payoffMatrices) != type([]):
        raise TypeError('payoffMatrices should be a list.')
    
    num_states = len(payoffMatrices)
    
    for s in range(num_states):
        if type(payoffMatrices[s]) != type([]) and type(payoffMatrices[s]) != type(np.array([])):
            raise TypeError('payoffMatrices[{0}] should be a nested list or a numpy array.'.format(s))
    
    num_players = payoffMatrices[0].shape[0]
    
    for s in range(num_states):
        if payoffMatrices[s].shape[0] != num_players:
            raise ValueError('payoffMatrices[{0}] should have the first dimension equal to the number of players = {1}.'.format(s, num_players))
    
    for s in range(num_states):
        for p in range(num_players):
            if payoffMatrices[s].shape[1+p] == 0:
                raise ValueError('payoffMatrices[{0}] should have at least one available action for player {1}.'.format(s, p))
    
    nums_actions = np.zeros((num_states, num_players), dtype=np.int32)
    for s in range(num_states):
        for p in range(num_players):
            nums_actions[s,p] = payoffMatrices[s].shape[1+p]
    num_actions_max = nums_actions.max()
    
    payoff_min = np.min([np.min(payoffMatrix) for payoffMatrix in payoffMatrices])
    payoff_max = np.max([np.max(payoffMatrix) for payoffMatrix in payoffMatrices])
    if payoff_min == payoff_max:
        raise ValueError('payoffMatrices should not be flat for all states and players.')
    
    payoffArray = np.nan * np.ones((num_states, num_players, *[num_actions_max]*num_players), dtype=np.float64)
    for s in range(num_states):
        for p in range(num_players):
            for A in np.ndindex(*nums_actions[s]):
                payoffArray[(s,p)+A] = payoffMatrices[s][(p,)+A]
    
    
    
    
    ## check transitionMatrices
    
    if transitionMatrices is None:
        if num_states > 1:
            raise TypeError('payoffMatrices for {0} states given. Please provide transitionMatrices.'.format(num_states))
        else:
            transitionArray = np.ones((1, *[num_actions_max]*num_players, 1))
    
    else:
        
        if type(transitionMatrices) != type([]):
            raise TypeError('transitionMatrices should be a list.')
        
        if len(transitionMatrices) != num_states:
            raise ValueError('transitionMatrices has length {0}, but number of states given by payoffMatrices is {1}.'.format(len(transitionMatrices, num_states)))
        
        for s in range(num_states):
            if type(transitionMatrices[s]) != type([]) and type(transitionMatrices[s]) != type(np.array([])):
                raise TypeError('transitionMatrices[{0}] should be a nested list or a numpy array.'.format(s))
        
        for s in range(num_states):
            if transitionMatrices[s].shape != tuple(nums_actions[s,:]) + (num_states,):
                raise ValueError('transitionMatrices[{0}] has dimensions {1} and should have {2}.'.format(s, transitionMatrices[s].shape, tuple(nums_actions[s,:])+(num_states,)))
        
        for s in range(num_states):
            for index, value in np.ndenumerate(np.sum(transitionMatrices[s], axis=-1)):
                if not np.allclose(value, 1):
                    raise ValueError('transitionMatrices[{0}][{1}] does not sum up to one, but to {2}.'.format(s, index, value))
        
        transitionArray = np.nan * np.ones((num_states, *[num_actions_max]*num_players, num_states), dtype=np.float64)
        for s0 in range(num_states):
            for A in np.ndindex(*nums_actions[s0]):
                for s1 in range(num_states):
                    transitionArray[(s0,)+A+(s1,)] = transitionMatrices[s0][A+(s1,)]
    
    
    
    
    ## check discountFactors
    
    if type(discountFactors) != int and type(discountFactors) != float and type(discountFactors) != type([]) and type(discountFactors) != np.ndarray:
        raise TypeError('discountFactors should be a single number (int or float) or a list or numpy.array of numbers.')
    
    if type(discountFactors) == type([]) or type(discountFactors) == np.ndarray:
        if len(discountFactors) != num_players:
            raise ValueError('discountFactors has length {0}, but the number of players is {1}. Please provide a single number (common discount factor) or a list of numbers with length equal to the number of players (individual discount factors).'.format(len(discountFactors), num_players))
        for i, discountFactor in enumerate(discountFactors):
            if discountFactor < 0 or discountFactor >= 1:
                raise ValueError('discountFactors[{0}] = {1} should be in [0,1).'.format(i, np.round(discountFactor,4)))
    
    else:
        if discountFactors < 0 or discountFactors >= 1:
            raise ValueError('discountFactors = {0} should be in [0,1).'.format(np.round(discountFactors,4)))
        discountFactors = [discountFactors] * num_players
    
    transitionArray_discounted = np.nan * np.ones((num_states, num_players, *[num_actions_max]*num_players, num_states), dtype=np.float64)
    for p in range(num_players):
        transitionArray_discounted[:,p] = discountFactors[p] * transitionArray
    
    
    
    
    ## check symmetryPairs
    
    if type(symmetryPairs) != type([]):
        raise TypeError('symmetryPairs should be a list.')
    
    for k, symmetryPair in enumerate(symmetryPairs):
        if type(symmetryPair) != tuple:
            raise TypeError('symmetryPairs[{0}] should be a tuple.'.format(k))
    
    for k, symmetryPair in enumerate(symmetryPairs):
        if np.array(symmetryPair).shape != (2,2):
            raise ValueError('symmetryPairs[{0}] should be a nested tuple of shape (2,2).'.format(k))
    
    
    for k, symmetryPair in enumerate(symmetryPairs):
        
        if type(symmetryPair[0]) != tuple or type(symmetryPair[1]) != tuple:
            raise TypeError('symmetryPairs[{0}] should be a nested tuple ((state1,player1),(state2,player2)).'.format(k))
        
        ((s1,p1),(s2,p2)) = symmetryPair
        
        if type(s1) != int or type(p1) != int or type(s2) != int or type(p2) != int:
            raise TypeError('symmetryPairs[{0}] should be a nested tuple ((state1,player1),(state2,player2)) of integer indices.'.format(k))
                
    
    for k, ((s1,p1),(s2,p2)) in enumerate(symmetryPairs):
        
        if nums_actions[s1,p1] != nums_actions[s2,p2]:
            raise ValueError('symmetryPair[{0}] = (({1},{2}),({3},{4})) is invalid. Player {2} in state {1} has {5} actions while player {4} in state {3} has {6}.'.format(k, s1, p1, s2, p2, nums_actions[s1,p1], nums_actions[s2,p2]))
        
        if not np.allclose(discountFactors[p1], discountFactors[p2]):
            raise ValueError('symmetryPair[{0}] = (({1},{2}),({3},{4})) is invalid. Player {2} has discount factor {5:0.4f} while player {4} has {6:0.4f}.'.format(k, s1, p1, s2, p2, discountFactors[p1], discountFactors[p2]))
    
    
    for k, ((s1,p1),(s2,p2)) in enumerate(symmetryPairs):
        
        u1 = payoffArray[s1,p1]
        u2 = payoffArray[s2,p2]
        if not np.allclose(np.sort(u1.ravel()), np.sort(u2.ravel()), equal_nan=True):
            raise ValueError('symmetryPair[{0}] = (({1},{2}),({3},{4})) is invalid. The two agents are not payoff-symmetric.'.format(k, s1, p1, s2, p2))
        
        phi1 = transitionArray_discounted[s1,p1]
        phi2 = transitionArray_discounted[s2,p2]
        if not np.allclose(np.sort(phi1.ravel()), np.sort(phi2.ravel()), equal_nan=True):
            raise ValueError('symmetryPair[{0}] = (({1},{2}),({3},{4})) is invalid. The two agents are not transition-symmetric.'.format(k, s1, p1, s2, p2))
    
    
    return True






## ============================================================================
## end of file
## ============================================================================
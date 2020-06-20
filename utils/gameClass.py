# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2019  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np
import matplotlib.pyplot as plt
import copy

import dsGameSolver.gameClass_checkInput as gameClass_checkInput
import dsGameSolver.homCreate as homCreate
import dsGameSolver.homSym as homSym
import dsGameSolver.homInit as homInit
import dsGameSolver.homCont as homCont






class dsGame:
    
    
    
    
    def __init__(self, payoffMatrices, transitionMatrices=None, discountFactors=0, 
                 detectSymmetry=True, symmetryPairs=[], 
                 checkInput=True, confirmationPrintout=True):
        
        ## no transitionMatrix supplied: repeated game, one state
        ## no discountFactors supplied: one-shot game, arbitrary number of states
        
        
        if checkInput:
            self.inputCorrect = gameClass_checkInput.inputCorrect(payoffMatrices=payoffMatrices, transitionMatrices=transitionMatrices, discountFactors=discountFactors, symmetryPairs=symmetryPairs)
        else:
            self.inputCorrect = None
        
        
        
        ## get dimensions of game
        
        self.payoffMatrices = []
        for s in range(len(payoffMatrices)):
            self.payoffMatrices.append( np.array(payoffMatrices[s]) )
        
        self.num_states = len(self.payoffMatrices)
        self.num_players = self.payoffMatrices[0].shape[0]
        
        self.nums_actions = np.zeros((self.num_states, self.num_players), dtype=np.int32)
        for s in range(self.num_states):
            for p in range(self.num_players):
                self.nums_actions[s,p] = self.payoffMatrices[s].shape[1+p]
        
        self.num_actions_max = self.nums_actions.max()
        self.num_actions_total = self.nums_actions.sum()
        self.num_actionProfiles = np.product(self.nums_actions, axis=1).sum()
        
        
        
        ## generate payoffArray and normalize to [0,1] for path tracking
        
        self.payoff_min = min([payoffMatrix.min() for payoffMatrix in self.payoffMatrices])
        self.payoff_max = max([payoffMatrix.max() for payoffMatrix in self.payoffMatrices])
        
        self.payoffArray = np.nan * np.ones((self.num_states, self.num_players, *[self.num_actions_max]*self.num_players), dtype=np.float64)
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.payoffArray[(s,p)+A] = (self.payoffMatrices[s][(p,)+A] - self.payoff_min) / (self.payoff_max - self.payoff_min)
        
        self.payoffArray_withoutNaN = copy.deepcopy(self.payoffArray)
        self.payoffArray_withoutNaN[np.isnan(self.payoffArray_withoutNaN)] = 0
        
        
        
        ## generate transitionArray
        
        if transitionMatrices is None:
            self.transitionMatrices = [np.ones((*self.nums_actions[s], self.num_states)) for s in range(self.num_states)]
            for s in range(self.num_states):
                for index, value in np.ndenumerate(np.sum(self.transitionMatrices[s], axis=-1)):
                    self.transitionMatrices[s][index] *= 1/value
        else:
            self.transitionMatrices = []
            for s in range(self.num_states):
                self.transitionMatrices.append( np.array(transitionMatrices[s]) )
        
        self.transitionArray = np.nan * np.ones((self.num_states, *[self.num_actions_max]*self.num_players, self.num_states), dtype=np.float64)
        for s0 in range(self.num_states):
            for A in np.ndindex(*self.nums_actions[s0]):
                for s1 in range(self.num_states):
                    self.transitionArray[(s0,)+A+(s1,)] = self.transitionMatrices[s0][A+(s1,)]
        self.transitionArray_withoutNaN = copy.deepcopy(self.transitionArray)
        self.transitionArray_withoutNaN[np.isnan(self.transitionArray_withoutNaN)] = 0
        
        
        
        ## generate transitionArray_discounted
            ## transitionArray: [s,A,s']
            ## transitionArray_discounted: [s,p,A,s']   (player index because of potentially different discount factors)
        
        if type(discountFactors) != type([]) and type(discountFactors) != np.ndarray:
            self.discountFactors = discountFactors * np.ones(self.num_players)
        else:
            self.discountFactors = np.array(discountFactors)
        
        self.transitionArray_discounted = np.nan * np.ones((self.num_states, self.num_players, *[self.num_actions_max]*self.num_players, self.num_states), dtype=np.float64)
        for p in range(self.num_players):
            self.transitionArray_discounted[:,p] = self.discountFactors[p] * self.transitionArray
        
        self.transitionArray_discounted_withoutNaN = copy.deepcopy(self.transitionArray_discounted)
        self.transitionArray_discounted_withoutNaN[np.isnan(self.transitionArray_discounted_withoutNaN)] = 0
        
        
        
        ## homCreate setup
        self.T_y2beta = homCreate.T_y2beta(num_s=self.num_states, num_p=self.num_players, nums_a=self.nums_actions)
        self.homotopy_helpers = homCreate.homotopy_helpers(num_s=self.num_states, num_p=self.num_players, nums_a=self.nums_actions, phi_withoutNaN=self.transitionArray_discounted_withoutNaN)
        
        
        
        ## get symmetry pairs and deduce independent agents
        
        self.symmetryPairs_user = copy.deepcopy(symmetryPairs)
        
        ## auto-detect symmetry pairs
        if detectSymmetry:
            
            beta_test = np.random.uniform(size=self.num_actions_max)
            V_test = np.random.uniform(size=1)
            t_test = np.random.uniform(size=1)
            
            y_test = []
            for s in range(self.num_states):
                for p in range(self.num_players):
                    for a in range(self.nums_actions[s,p]):
                        y_test.append( beta_test[a] )
            y_test.extend( [V_test[0]]*self.num_states*self.num_players + [t_test[0]] )
            y_test = np.array(y_test)
            
            H_test = self.H_full(y_test)
            
            self.symmetryPairs_detected = homSym.symmetryPairs(H_test=H_test, T_y2beta=self.T_y2beta, deltas=self.discountFactors, num_s=self.num_states, num_p=self.num_players, nums_a=self.nums_actions)
        
        else:
            self.symmetryPairs_detected = []
        
        
        ## order user-provided symmetryPairs by player and state, keep first
        self.symmetryPairs_ordered = []
        for ((s1,p1),(s2,p2)) in self.symmetryPairs_user:
            
            ## order by player and state: keep first
            if p1 < p2:
                self.symmetryPairs_ordered.append( ((s1,p1),(s2,p2)) )
            elif p2 < p1:
                self.symmetryPairs_ordered.append( ((s2,p2),(s1,p1)) )
            else:
                if s1 <= s2:
                    self.symmetryPairs_ordered.append( ((s1,p1),(s2,p2)) )
                else:
                    self.symmetryPairs_ordered.append( ((s2,p2),(s1,p1)) )
        
        
        if len(self.symmetryPairs_user) > 0 and detectSymmetry and set(self.symmetryPairs_detected) != set(self.symmetryPairs_ordered):
            print('Warning: symmetryPairs provided do not match symmetryPairs_detected.')
            print('         symmetryPairs_ordered  = {0}'.format(self.symmetryPairs_ordered))
            print('         symmetryPairs_detected = {0}'.format(self.symmetryPairs_detected))
            print('         symmetryPairs_ordered are used. To use symmetryPairs_detected instead, set detectSymmetry = True and do not provide symmetryPairs.')
        
        if detectSymmetry and len(self.symmetryPairs_user) == 0:
            self.symmetryPairs_ordered = copy.deepcopy(self.symmetryPairs_detected)
        
        
        self.independentAgents = [(s,p) for s in range(self.num_states) for p in range(self.num_players)]
        for ((s1,p1),(s2,p2)) in self.symmetryPairs_ordered:
            self.independentAgents.remove( (s2,p2) )
        self.num_independentAgents = len(self.independentAgents)
        self.num_independentActions_total = sum([self.nums_actions[s,p] for (s,p) in self.independentAgents])
        
        nums_a = self.nums_actions.copy()
        for s in range(self.num_states):
            for i in range(self.num_players):
                if (s,i) not in self.independentAgents:
                    nums_a[s,i] = 1
        self.num_independentActionProfiles = np.product(nums_a, axis=1).sum()
        
        
        
        ## homSym setup
        self.symmetricGame = True if len(self.symmetryPairs_ordered) > 0 else False
        self.symmetry_helpers = homSym.reduction_expansion_helpers(sym_pairs=self.symmetryPairs_ordered, num_s=self.num_states, num_p=self.num_players, nums_a=self.nums_actions)
        
        
        
        ## miscellaneous initialization
        
        self.solvedPositions = []
        self.pathData = None
        
        if confirmationPrintout:
            print('Dynamic stochastic game with {0} states, {1} players and {2} actions, in total {3} action profiles.'.format(self.num_states, self.num_players, self.num_actions_total, self.num_actionProfiles))
            if self.symmetricGame:
                print('Symmetries reduce game to {0} state-player pairs and {1} actions, in total {2} action profiles.'.format(self.num_independentAgents, self.num_independentActions_total, self.num_independentActionProfiles))
        
        
        return
    
    
    
    
    
    
    ## define homotopy function and Jacobian
    
    def H_full(self, y): 
        return homCreate.H_qre(y=y, u_withoutNaN=self.payoffArray_withoutNaN, phi_withoutNaN=self.transitionArray_discounted_withoutNaN, num_s=self.num_states, num_p=self.num_players, num_a_tot=self.num_actions_total, T_y2beta=self.T_y2beta, homotopy_helpers=self.homotopy_helpers)
    
    def H_reduced(self, y): 
        return homSym.H_reduced(y_reduced=y, H_full=self.H_full, symmetry_helpers=self.symmetry_helpers)
    
    
    def J_full(self, y): 
        return homCreate.J_qre(y=y, u_withoutNaN=self.payoffArray_withoutNaN, phi_withoutNaN=self.transitionArray_discounted_withoutNaN, num_s=self.num_states, num_p=self.num_players, num_a_max=self.num_actions_max, num_a_tot=self.num_actions_total, T_y2beta=self.T_y2beta, homotopy_helpers=self.homotopy_helpers)
    
    def J_reduced(self, y): 
        return homSym.J_reduced(y_reduced=y, J_full=self.J_full, symmetry_helpers=self.symmetry_helpers)
    
    
    
    
    
    
    def init(self, showProgress=False):
        
        success, y0 = homInit.get_y0(u=self.payoffArray, phi=self.transitionArray_discounted, nums_a=self.nums_actions)
        
        if not success:
            print('Error: Initial value for homotopy continuation could not be found.')
            return
        
        
        y0_reduced = np.append(y0[self.symmetry_helpers['idx_reduce']], 0)
            
        solvedPosition = {
                'success': True,
                'num_steps': 0,
                't': 0,
                's': 0,
                'ds': None,
                'sign': None,
                'y': y0,
                'y_reduced': y0_reduced
                }
        
        solvedPosition['strategies'] = np.einsum('spaN,N->spa', self.T_y2beta, np.exp(solvedPosition['y'][:self.num_actions_total]))
        solvedPosition['stateValues'] = solvedPosition['y'][self.num_actions_total:-1].reshape((self.num_states,self.num_players))
        
        self.solvedPositions = [solvedPosition] 
        
        if showProgress:
            print('Initial value for homotopy continuation successfully found.')
        
        return
    
    
    
    
    
    
    def get_trackingParameters(self, trackingMethod='normal', **kwargs):
        
        if trackingMethod == 'normal':
            y_tol = 1e-7
            t_tol = 1e-7
            H_tol = 1e-7
            ds0 = 0.01
            ds_infl = 1.2
            ds_defl = 0.5
            ds_min = 1e-9
            ds_max = 1000
            corr_steps_max = 20
            corr_dist_max = 0.3
            corr_contr_max = 0.3
            detJratio_max = 1.3
            bifurc_angle_min = 175
        
        elif trackingMethod == 'robust':
            y_tol = 1e-7
            t_tol = 1e-7
            H_tol = 1e-8
            ds0 = 0.01
            ds_infl = 1.1
            ds_defl = 0.5
            ds_min = 1e-9
            ds_max = 1000
            corr_steps_max = 30
            corr_dist_max = 0.1
            corr_contr_max = 0.1
            detJratio_max = 1.1
            bifurc_angle_min = 175
        
        else:
            raise ValueError('Unknown trackingMethod {0}. Use "normal" or "robust".'.format(trackingMethod))
        
        
        ## use **kwargs to overwrite default values
        parameters = {
                'y_tol': kwargs.get('y_tol', y_tol),
                't_tol': kwargs.get('t_tol', t_tol),
                'H_tol': kwargs.get('H_tol', H_tol),
                'ds0': kwargs.get('ds0', ds0),
                'ds_infl': kwargs.get('ds_infl', ds_infl),
                'ds_defl': kwargs.get('ds_defl', ds_defl),
                'ds_min': kwargs.get('ds_min', ds_min),
                'ds_max': kwargs.get('ds_max', ds_max),
                'corr_steps_max': kwargs.get('corr_steps_max', corr_steps_max),
                'corr_dist_max': kwargs.get('corr_dist_max', corr_dist_max),
                'corr_contr_max': kwargs.get('corr_contr_max', corr_contr_max),
                'detJratio_max': kwargs.get('detJratio_max', detJratio_max),
                'bifurc_angle_min': kwargs.get('bifurc_angle_min', bifurc_angle_min)
                }
        
        
        return parameters
    
    
    
    
    
    
    def solve(self, t_list=[], showProgress=False, pathData=True, 
              trackingMethod='normal', num_steps_max=np.inf, **kwargs):
        
        
        ## set tracing parameters
        parameters = self.get_trackingParameters(trackingMethod=trackingMethod, **kwargs)
        
        
        if len(self.solvedPositions) == 0:
            self.init(showProgress=showProgress)
        
        if len(self.solvedPositions) == 0:
            return
        
        
        if pathData:
            self.pathData = homCont.hist(dim_y=len(self.solvedPositions[-1]['y_reduced']))
        else:
            self.pathData = None
        
        
        if len(t_list) == 0:
            t_list = [np.inf]
        
        def H(y): 
            if self.symmetricGame: return self.H_reduced(y=y)
            else: return self.H_full(y=y)
        def J(y): 
            if self.symmetricGame: return self.J_reduced(y=y)
            else: return self.J_full(y=y)
        
        
        for t_target in t_list:
            
            sign = self.solvedPositions[-1]['sign']
            if t_target < self.solvedPositions[-1]['t'] and sign is not None:
                sign = - sign
            
            solvedPosition = homCont.solve(H=H, J=J, y0=self.solvedPositions[-1]['y_reduced'], 
                                           s=self.solvedPositions[-1]['s'], ds=None, sign=sign,
                                           sigma_count=self.num_independentActions_total, 
                                           hist=self.pathData, t_target=t_target, 
                                           num_steps_max=num_steps_max,
                                           progress=showProgress, parameters=parameters)
            
            solvedPosition['y'] = solvedPosition['y_reduced'][self.symmetry_helpers['idx_expand']]
            
            solvedPosition['strategies'] = np.einsum('spaN,N->spa', self.T_y2beta, np.exp(solvedPosition['y'][:self.num_actions_total]))
            
            ## scale back state values from payoff array normalized to [0,1]
            solvedPosition['stateValues'] = np.tile((1/(1-self.discountFactors)), (self.num_states,1)) * self.payoff_min + (self.payoff_max - self.payoff_min) * solvedPosition['y'][self.num_actions_total:-1].reshape((self.num_states,self.num_players))
            
            self.solvedPositions.append(solvedPosition)
        
        
        if pathData:
            self.pathData.finalize()
        
        
        return
    
    
    
    
    
    
    def plot(self):
        if self.pathData is None:
            print('Method "solve(pathData=True)" must be run to enable plotting.')
            fig = plt.figure()
        else:
            self.pathData.plotprep()
            fig = self.pathData.plot(sigma_count=self.num_independentActions_total)
        return fig






## ============================================================================
## end of file
## ============================================================================
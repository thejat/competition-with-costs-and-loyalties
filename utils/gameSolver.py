# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2019  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np
import utils.gameClass as gameClass






def dsSolve(payoffMatrices, transitionMatrices=None, discountFactors=0, 
            showProgress=False, plotPath=False, t_target=np.inf,
            detectSymmetry=True, symmetryPairs=[], **kwargs):
    
    
    ## create game class
    game = gameClass.dsGame(payoffMatrices=payoffMatrices, transitionMatrices=transitionMatrices, 
                            discountFactors=discountFactors, detectSymmetry=detectSymmetry, 
                            symmetryPairs=symmetryPairs, confirmationPrintout=showProgress)
    
    
    ## track homotopy path
    pathData=False
    if plotPath:
        pathData=True
        
    for k, trackingMethod in enumerate(['normal', 'robust']):
        
        game.init(showProgress=showProgress)
        if len(game.solvedPositions) == 0:
            ## Initial value not found, error message printed.
            return {
                    'strategies': np.nan * np.ones((game.num_states, game.num_players, game.num_actions_max)),
                    'stateValues': np.nan * np.ones((game.num_states, game.num_players)),
                    'success': False,
                    'num_steps': 0,
                    'homotopyParameter': 0,
                    'pathLength': 0
                    }
        
        game.solve(t_list=[t_target], showProgress=showProgress, pathData=pathData, 
                   trackingMethod=trackingMethod, **kwargs)
        
        if game.solvedPositions[-1]['success']:
            break
        
        else:
            if showProgress:
                print('Path tracing not successful with trackingMethod="{0}".'.format(trackingMethod))
                if trackingMethod == 'normal':
                    print('Trying again with trackingMethod="robust".')
                else:
                    print('Path Tracking unsuccessful. Try again with tighter tracking parameters specified in **kwargs.')
    
    
    ## plot path
    if plotPath:
        game.plot()
    
    
    ## output dictionary
    equilibrium = {
            'strategies': game.solvedPositions[-1]['strategies'],
            'stateValues': game.solvedPositions[-1]['stateValues'],
            'success': game.solvedPositions[-1]['success'],
            'num_steps': game.solvedPositions[-1]['num_steps'],
            'homotopyParameter': game.solvedPositions[-1]['t'],
            'pathLength': game.solvedPositions[-1]['s']
            }
    
    
    return equilibrium






## ============================================================================
## end of file
## ============================================================================
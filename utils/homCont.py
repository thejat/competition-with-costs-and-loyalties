# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2019  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import sys

import utils.homCont_subfunctions as func






class hist:
    
    def __init__(self, dim_y, cutoff=500000):
        self.cutoff = cutoff
        self.t = np.ones(self.cutoff) * np.nan
        self.s = np.ones(self.cutoff) * np.nan
        self.y = np.ones((self.cutoff, dim_y)) * np.nan
        self.finalized = False
    
    def update(self, t, s, y):
        nan_indices = np.where(np.isnan(self.t))[0]
        if len(nan_indices) > 0:
            j = nan_indices[0]
            self.t[j] = t
            self.s[j] = s
            self.y[j,:] = y
    
    def update_final(self, t, s):
        nan_indices = np.where(np.isnan(self.t))[0]
        if len(nan_indices) > 0:
            j = nan_indices[0] - 1
            self.t[j] = t
            self.s[j] = s
    
    def finalize(self):
        self.t = self.t[~np.isnan(self.t)]
        self.s = self.s[~np.isnan(self.s)]
        self.y = self.y[~np.isnan(self.y).any(axis=1), :]
        if not self.finalized:
            self.y = np.exp(self.y)
        self.finalized = True
    
    def plotprep(self, cutoff=100000, samplefreq=10):
        if not self.finalized:
            self.finalize()
        while len(self.t) > cutoff:
            ## downsample
            self.t = self.t[::samplefreq]
            self.s = self.s[::samplefreq]
            self.y = self.y[::samplefreq, :]
    
    def plot(self, sigma_count=-1):
        fig = plt.figure(figsize=(12, 4))
        ## s -> t
        ax1 = fig.add_subplot(121)
        ax1.set_title('Homotopy Path')
        ax1.set_xlabel(r'path length $s$')
        ax1.set_ylabel(r'homotopy parameter $\lambda$')
        ax1.set_ylim(0, np.max([1, np.amax(self.t)]))
        ax1.plot(self.s, self.t)
        ax1.grid()
        ## s -> y
        ax2 = fig.add_subplot(122)
        ax2.set_title('Strategy Convergence')
        ax2.set_xlabel(r'path length $s$')
        ax2.set_ylabel(r'strategies $\sigma$')
        ax2.set_ylim(0,1)
        ax2.plot(self.s, self.y[:, :sigma_count])
        ax2.grid()
        plt.show()
        return fig






def solve(H, J, y0, num_steps=None, s=None, ds=None, sign=None, sigma_count=-1, 
          hist=None, t_target=np.inf, num_steps_max=np.inf, progress=False, 
          parameters = {
                  'y_tol': 1e-7,
                  't_tol': 1e-7,
                  'H_tol': 1e-7,
                  'ds0': 0.01,
                  'ds_infl': 1.3,
                  'ds_defl': 0.5,
                  'ds_min': 1e-9,
                  'ds_max': 1000,
                  'corr_steps_max': 20,
                  'corr_dist_max': 0.3,
                  'corr_contr_max': 0.3,
                  'detJratio_max': 1.2,
                  'bifurc_angle_min': 175
                  }
          ):
    
    ## start stopwatch
    tic = time.time()
    if progress:
        print('=' * 50)
        print('Start homotopy continuation')
    
    ## unpack tracking parameters
    y_tol = parameters['y_tol']
    t_tol = parameters['t_tol']
    H_tol = parameters['H_tol']
    ds0 = parameters['ds0']
    ds_infl = parameters['ds_infl']
    ds_defl = parameters['ds_defl']
    ds_min = parameters['ds_min']
    ds_max = parameters['ds_max']
    corr_steps_max = parameters['corr_steps_max']
    corr_dist_max = parameters['corr_dist_max']
    corr_contr_max = parameters['corr_contr_max']
    detJratio_max = parameters['detJratio_max']
    bifurc_angle_min = parameters['bifurc_angle_min']
    
    ## get orientation of homotopy path
    y_old = y0.copy()
    J_y = J(y_old)
    Q, R = func.QR(J_y)
    if sign is None:
        sign = 1
        dtds = func.tangent(Q, R, sign)[-1]
        if dtds < 0: 
            sign = - sign
    tangent_old = func.tangent(Q, R, sign)
    detJ_y = np.linalg.det(np.vstack([J_y, tangent_old]))
    
    ## starting point for homotopy continuation
    t_init = y_old[-1]
    t_min = min([t_init, t_target])
    t_max = max([t_init, t_target])
    t = t_init
    if num_steps is None:
        num_steps = 0
    if s is None:
        s = 0
    if ds is None:
        ds = ds0
    if hist is not None:
        hist.update(t=t, s=s, y=y0)
    
    
    ## path tracking loop
    continue_tracking = True
    while continue_tracking and num_steps < num_steps_max:
        
        num_steps += 1
    
        ## compute tangent at y_old
        Q, R = func.QR(J_y)
        tangent = func.tangent(Q, R, sign)
        
        ## test for bifurcation point
        angle = np.arccos( min([1, max([-1, np.dot(tangent, tangent_old)])]) ) * 180 / np.pi
        sign, tangent, angle, detJ_y = func.sign_testBifurcation(
                sign, tangent, angle, J_y, detJ_y, bifurc_angle_min=bifurc_angle_min, progress=progress)
        
        
        ## predictor-corrector step loop
        success_corr = False
        while not success_corr:
            
            ## predictor
            y_pred = y_old + ds * tangent
            
            ## compute J_pinv at predictor point
            J_y_pred = J(y_pred)
            Q, R = func.QR(J_y_pred)
            J_pinv = func.J_pinv(Q, R)
            
            ## corrector loop
            y_corr, J_y_corr, success_corr, corr_steps, corr_contr_init, corr_dist_tot, err_msg = func.y_corrector(
                        y=y_pred, H=H, J=J, J_y_pred=J_y_pred, J_pinv=J_pinv, tangent=tangent, ds=ds, sign=sign, 
                        H_tol=H_tol, corr_steps_max=corr_steps_max, corr_dist_max=corr_dist_max, 
                        corr_contr_max=corr_contr_max, detJratio_max=detJratio_max)
            if not success_corr:
                ds = func.deflate(ds=ds, ds_defl=ds_defl)
                if ds < ds_min: break
        
        
        ## update parameters
        tangent_old = tangent.copy()
        J_y = J_y_corr.copy()
        t, s, ds, y_old, continue_tracking, success = func.update_parameters(
                s=s, ds=ds, y_corr=y_corr, y_old=y_old,
                corr_steps=corr_steps, corr_contr_init=corr_contr_init, corr_dist_tot=corr_dist_tot,
                ds_infl=ds_infl, ds_defl=ds_defl, ds_min=ds_min, ds_max=ds_max, 
                t_min=t_min, t_max=t_max, t_init=t_init, t_target=t_target, t_tol=t_tol, 
                y_tol=y_tol, sigma_count=sigma_count, progress=progress, err_msg=err_msg)
        
        
        ## print progress report
        if hist is not None:
            hist.update(t=t, s=s, y=y_corr)
        if progress and success_corr and success:
            sys.stdout.write('\rStep {0}:   t = {1:0.2f},   s = {2:0.2f},   ds = {3:0.2f}   '.format( num_steps, t, s, ds ))
            sys.stdout.flush()
    
    ## end of path tracking loop
    if progress and not success and num_steps > num_steps_max:
        print('\nMaximum number of steps reached.')
    else:
        if hist is not None:
            hist.update_final(t=t, s=s)
    
    
    ## output
    if progress:
        H_test = np.max(np.abs(H(y_corr)))
        y_test = np.max(np.abs(np.exp(y_corr[:sigma_count]) - np.exp(y_old[:sigma_count]))) / ds
        ## report new step and stop stopwatch
        print('\nFinal Result:   max|y-y_|/ds = {0:0.1E},   max|H| = {1:0.1E}'.format( y_test, H_test ))
        print('Time elapsed = {0}'.format( datetime.timedelta(seconds=round(time.time()-tic)) ))
        print('End homotopy continuation')
        print('=' * 50)
    
    output = {
            'success': success,
            'num_steps': num_steps,
            't': t,
            's': s,
            'ds': ds,
            'sign': sign,
            'y_reduced': y_corr
            }
    
    return output






## ============================================================================
## end of file
## ============================================================================
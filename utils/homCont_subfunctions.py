# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2019  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np






def QR(J_y):
    return np.linalg.qr(J_y.transpose(), mode='complete')

def tangent(Q, R, sign):
    return sign * Q[:,-1] * np.sign(np.prod(np.diag(R)))

def J_pinv(Q, R):
    return np.dot(Q, np.vstack((np.linalg.inv(np.delete(R,-1,axis=0).transpose()),np.zeros(R.shape[1]))))




def deflate(ds, ds_defl=0.5):
    return ds * ds_defl

def inflate(ds, corr_steps, corr_dist_tot, corr_contr_init, ds_infl=1.2, ds_max=1000):
    ## optional: step size inflation based on accuracy of predictor step
    if corr_steps < 10:
        return min([ds_infl*ds, ds_max])
    else:
        return ds




def sign_testBifurcation(sign, tangent, angle, J_y, detJ_y_old, bifurc_angle_min=175, progress=False):
    
    detJ_y = np.linalg.det(np.vstack([J_y, tangent]))
    
#    ## bifurcation detection based on change in sign of determinant of augmented Jacobian
#    if detJ_y * detJ_y_old < 0:
    
    ## bifurcation detection based on angle between consecutive tangents
    if angle > bifurc_angle_min:
        sign *= -1
        tangent *= -1
        if progress:
            print('\nBifurcation point encountered at angle {0:0.2f}°. Direction swapped.'.format( angle ))
        angle = 180 - angle
    
    return sign, tangent, angle, detJ_y




def y_corrector(y, H, J, J_y_pred, J_pinv, tangent, ds, sign, H_tol=1e-7, 
                corr_steps_max=20, corr_dist_max=0.3, corr_contr_max=0.3, detJratio_max=1.3):
    
    detJ_y_pred = np.linalg.det(np.vstack([J_y_pred, tangent]))
    
    corr_dist_old = np.inf
    corr_dist_tot = 0
    corr_contr_init = 0
    corr_step = 0
    
    while np.max(np.abs(H(y))) > H_tol:
        
        corr_step += 1
        
        ## corrector step
        vec = np.dot(J_pinv, H(y))
        y = y - vec
        corr_dist_step = np.linalg.norm(vec)
        corr_dist_tot += corr_dist_step
        corr_dist = corr_dist_step / max([ds,1])
        corr_contr = corr_dist / corr_dist_old
        corr_dist_old = corr_dist
        if corr_step == 2:
            corr_contr_init = corr_contr
        
        ## if corrector step violates restriction on distance or contraction or number of steps,
        ## decrease step size and return to predictor step
        if corr_dist > corr_dist_max or corr_contr > corr_contr_max or corr_step > corr_steps_max:
            err_msg = ''
            if corr_dist > corr_dist_max:
                err_msg = err_msg + '\ncorr_dist = {0:0.4f} > corr_dist_max = {1:0.4f};   '.format(corr_dist, corr_dist_max)
            if corr_contr > corr_contr_max:
                err_msg = err_msg + '\ncorr_contr = {0:0.4f} > corr_contr_max = {1:0.4f};   '.format(corr_contr, corr_contr_max)
            if corr_step > corr_steps_max:
                err_msg = err_msg + '\ncorr_step = {0} > corr_steps_max = {1};   '.format(corr_step, corr_steps_max)
            err_msg = err_msg + 'cond(J) = {0:0.0f}'.format( np.linalg.cond(J_y_pred) )
            return y, J_y_pred, False, corr_step, corr_dist_tot, corr_contr_init, err_msg
    
    ## if determinant of augmented Jacobian changes too much during correction,
    ## then also decrease step size and return to predictor step
    J_y = J(y)
    detJ_y = np.linalg.det(np.vstack([J_y, tangent]))
    detJratio = abs(detJ_y) / abs(detJ_y_pred)
    if detJratio > detJratio_max or detJratio < 1/detJratio_max:
        err_msg = '\ndetJratio = {0:0.4f} not in [{1:0.2f}, {2:0.2f}]'.format(detJratio, 1/detJratio_max, detJratio_max)
        return y, J_y, False, corr_step, corr_dist_tot, corr_contr_init, err_msg
    
    return y, J_y, True, corr_step, corr_dist_tot, corr_contr_init, ''




def update_parameters(s, ds, y_corr, y_old,
                      corr_steps, corr_dist_tot, corr_contr_init,
                      ds_infl=1.5, ds_defl=0.5, ds_min=1e-9, ds_max=1000, 
                      t_min=0, t_max=np.inf, t_init=0, t_target=np.inf, t_tol=1e-7, 
                      y_tol=1e-7, sigma_count=-1, progress=False, err_msg=''):
    
    ## update t, s, ds
    t = y_corr[-1]
    step_dist = np.linalg.norm(y_corr-y_old)
    s += step_dist
    continue_tracking = True
    success = True
    
    
    ## good case: path tracking fine
    if t >= max([t_min-ds_max, 0]) and t <= t_max + ds_max and ds >= ds_min:
        
        
        ## convergence criterion: y -> y_final
        if np.isinf(t_target):
            
            ## check convergence
            if ds == ds_max:
                y_test = np.max(np.abs(np.exp(y_corr[:sigma_count]) - np.exp(y_old[:sigma_count]))) / ds
                if y_test < y_tol:
                    continue_tracking = False
            else:
                ds = inflate(ds=ds, corr_steps=corr_steps, corr_dist_tot=corr_dist_tot, 
                             corr_contr_init=corr_contr_init, ds_infl=ds_infl, ds_max=ds_max)
        
        
        ## convergence criterion: t -> t1
        else:
            
            ## case 1: t still in bound -> update and continue loop
            if t_target > t_init: test_bool1 = t < t_target - t_tol   ## increasing t
            else: test_bool1 = t > t_target + t_tol                   ## decreasing t
            
            ## case 2: t too far -> use previous step with decreased step size
            if t_target > t_init: test_bool2 = t > t_target + t_tol   ## increasing t
            else: test_bool2 = t < t_target - t_tol                   ## decreasing t
            
            if test_bool1:
                ds = inflate(ds=ds, corr_steps=corr_steps, corr_dist_tot=corr_dist_tot, 
                             corr_contr_init=corr_contr_init, ds_infl=ds_infl, ds_max=ds_max)
                y_old = y_corr.copy()
            
            elif test_bool2:
                t = y_old[-1]
                s -= step_dist
                ds = np.abs(t - t_target)
                y_corr = y_old.copy()
            
            ## case 3: t_goal - t_tol <= t <= t_goal + t_tol -> done!
            else: 
                continue_tracking = False
    
    
    ## bad case: path tracking stuck
    else:
        continue_tracking = False
        success = False
        if progress: 
            print('\nHomotopy continuation got stuck.' + err_msg)
    
    
    return t, s, ds, y_corr.copy(), continue_tracking, success






## ============================================================================
## end of file
## ============================================================================
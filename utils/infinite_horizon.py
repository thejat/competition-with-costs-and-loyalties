from utils.imports import *
from utils.single_stage import get_xi_dist
'''
In the below variables, first index is always firm, and second index is the strong sub-market
For instance, in paa_arr, first 'a' represents firm A's pricing, second 'a' represents A's strong sub-market 'alpha'
'''

def xi_equations(candidates,ca,cb,ta,tb,F,f,deltaf):
    gammaa = (ca-cb)/ta
    gammab = (cb-ca)/tb
    delbydel = (1-deltaf)/deltaf
    
    xia, xib = candidates
    if xia<0 or xia>1 or xib <0 or xib > 1: #hardcoded for cdf between 0 and 1: TODO
        return (math.inf,math.inf)

    residual1 = (xia-gammaa)*(delbydel+F(xib)+1) \
                + ((2*F(xia)-1)/f(xia))*(delbydel + F(xib) +F(xia)) \
                + (F(xia)/f(xia)) \
                - ((1-F(xib)*tb)/(ta*f(xib)) -F(xib)*(gammaa + xib*tb/ta ))
    residual2 = (xib-gammab)*(delbydel+F(xia)+1) \
                + ((2*F(xib)-1)/f(xib))*(delbydel + F(xib) +F(xia)) \
                + (F(xib)/f(xib)) \
                - ((1-F(xia)*ta)/(tb*f(xia)) -F(xia)*(gammab + xia*ta/tb ))
    
    return (residual1,residual2)

def vaopt_diff(paa,pab,xia,xib,ca,F,deltaf):
    return ((1-F(xia))*(paa-ca) - F(xib)*(pab-ca))/(1-deltaf+deltaf*(F(xia)+F(xib)))

def vbopt_diff(pbb,pba,xia,xib,cb,F,deltaf):
    return ((1-F(xib))*(pbb-cb) - F(xia)*(pba-cb))/(1-deltaf+deltaf*(F(xia)+F(xib)))

def pax_equations(candidates,xia,xib,ca,cb,ta,tb,F,f,deltaf):
    paa,pab = candidates
    if xia<0 or xia>1 or xib <0 or xib > 1:
        return (math.inf,math.inf)
    vaopt = vaopt_diff(paa,pab,xia,xib,ca,F,deltaf)
    residual1 = paa -(ca + (1-F(xia))*ta/f(xia) -deltaf*vaopt)
    residual2 = pab -(ca + F(xib)*tb/f(xib) -deltaf*vaopt)
    return (residual1,residual2)

def pbx_equations(candidates,xia,xib,ca,cb,ta,tb,F,f,deltaf):
    pbb,pba = candidates
    if xia<0 or xia>1 or xib <0 or xib > 1:
        return (math.inf,math.inf)
    vbopt = vbopt_diff(pbb,pba,xia,xib,cb,F,deltaf)
    residual1 = pbb - (cb + (1-F(xib))*tb/f(xib) -deltaf*vbopt)
    residual2 = pba - (cb + F(xia)*ta/f(xia) -deltaf*vbopt)
    return (residual1,residual2)

def constraints_a(paa,pba,ca,cb,ta):
    if paa >= ca and pba >= cb and (0 <= paa-pba) and  (paa-pba <= ta):
        return True
    return False

def constraints_b(pbb,pab,ca,cb,tb):
    if pbb >= cb and pab >= ca and (0 <= pbb-pab) and  (pbb-pab <= tb):
        return True
    return False

def get_prices_xis_constraints_and_vs(ca_arr,cb,ta,tb,F,f,deltaf):
    paa_arr = np.zeros(ca_arr.size) #first index is firm, second index is customer type
    pba_arr = np.zeros(ca_arr.size)
    pbb_arr = np.zeros(ca_arr.size)
    pab_arr = np.zeros(ca_arr.size)
    xia_arr = np.zeros(ca_arr.size)
    xib_arr = np.zeros(ca_arr.size)
    constraint_aa_ba_arr = np.zeros(ca_arr.size)
    constraint_bb_ab_arr = np.zeros(ca_arr.size)
    vaopt_diff_arr = np.zeros(ca_arr.size)
    vbopt_diff_arr = np.zeros(ca_arr.size)

    for i,ca in enumerate(ca_arr):
        xia_arr[i],xib_arr[i] = fsolve(xi_equations, (0.5, 0.5),(ca,cb,ta,tb,F,f,deltaf))
        paa_arr[i],pab_arr[i] = fsolve(pax_equations, (0.5, 0.5),(xia_arr[i],xib_arr[i],ca,cb,ta,tb,F,f,deltaf))
        pbb_arr[i],pba_arr[i] = fsolve(pbx_equations, (0.5, 0.5),(xia_arr[i],xib_arr[i],ca,cb,ta,tb,F,f,deltaf))
        
        vaopt_diff_arr[i] = vaopt_diff(paa_arr[i],pab_arr[i],xia_arr[i],xib_arr[i],ca,F,deltaf)
        vbopt_diff_arr[i] = vbopt_diff(pbb_arr[i],pba_arr[i],xia_arr[i],xib_arr[i],cb,F,deltaf)
        
        if constraints_a(paa_arr[i],pba_arr[i],ca,cb,ta):
            constraint_aa_ba_arr[i] = 1
        if constraints_b(pbb_arr[i],pab_arr[i],ca,cb,tb):
            constraint_bb_ab_arr[i] = 1

    return paa_arr,pba_arr,pbb_arr,pab_arr,xia_arr,xib_arr,constraint_aa_ba_arr,constraint_bb_ab_arr,vaopt_diff_arr,vbopt_diff_arr

def get_vopts(paa,pab,pbb,pba,xia,xib,da,db,ca,cb,F):

    #vaao, vabo
    mat1 = np.array([[1-da*(1-F(xia)),-da*F(xia)],[-da*F(xib),1-da*(1-F(xib))]])
    rhs1 = np.array([(1-F(xia))*(paa-ca),F(xib)*(pab-ca)])
    sol1 = np.linalg.inv(mat1).dot(rhs1)
    vaao,vabo = sol1[0],sol1[1]

    mat2 = np.array([[1-db*(1-F(xib)),-db*F(xib)],[-db*F(xia),1-db*(1-F(xia))]])
    rhs2 = np.array([(1-F(xib))*(pbb-cb),F(xib)*(pba-cb)])
    sol2 = np.linalg.inv(mat2).dot(rhs2)
    vbbo,vbao = sol2[0],sol2[1]

    return vaao,vabo,vbbo,vbao

def get_theoretical_px_obj(ca,cb,ta,tb,F,f,deltaf):

    # xia,xib = fsolve(xi_equations, (0.5, 0.5),(ca,cb,ta,tb,F,f,deltaf))

    paa_arr,pba_arr,pbb_arr,pab_arr,xia_arr,xib_arr,constraint_aa_ba_arr,constraint_bb_ab_arr,\
        vaopt_diff_arr,vbopt_diff_arr = get_prices_xis_constraints_and_vs(np.array([ca]),cb,ta,tb,F,f,deltaf)

    vaao,vabo,vbbo,vbao = get_vopts(paa_arr[0],pab_arr[0],pbb_arr[0],pba_arr[0],xia_arr[0],xib_arr[0],deltaf,deltaf,ca,cb,F)

    return vaao,vabo,vbbo,vbao,paa_arr[0],pab_arr[0],pbb_arr[0],pba_arr[0],constraint_aa_ba_arr[0],constraint_bb_ab_arr[0]


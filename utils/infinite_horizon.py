from utils.imports import *
from utils.single_stage import get_xi_dist,get_payoff_aa,get_payoff_ba,get_payoff_bb,get_payoff_ab,\
                    prob_cust_a_purchase_from_a,prob_cust_b_purchase_from_b
from utils.gameSolver import dsSolve

'''
In the below variables, first index is always firm, and second index is the strong sub-market
For instance, in paa_arr, first 'a' represents firm A's pricing, second 'a' represents A's strong sub-market 'alpha'
'''

def xi_equations(candidates,ca,cb,la,lb,F,f,df):
    gammaa = (ca-cb)/la
    gammab = (cb-ca)/lb
    delbydel = (1-df)/df
    
    xia, xib = candidates
    if xia<0 or xia>1 or xib <0 or xib > 1: #hardcoded for cdf between 0 and 1: TODO
        return (math.inf,math.inf)

    residual1 = (xia-gammaa)*(delbydel+F(xib)+1) \
                + ((2*F(xia)-1)/f(xia))*(delbydel + F(xib) +F(xia)) \
                + (F(xia)/f(xia)) \
                - ((1-F(xib)*lb)/(la*f(xib)) -F(xib)*(gammaa + xib*lb/la ))
    residual2 = (xib-gammab)*(delbydel+F(xia)+1) \
                + ((2*F(xib)-1)/f(xib))*(delbydel + F(xib) +F(xia)) \
                + (F(xib)/f(xib)) \
                - ((1-F(xia)*la)/(lb*f(xia)) -F(xia)*(gammab + xia*la/lb ))
    
    return (residual1,residual2)

def vaopt_diff(paa,pab,xia,xib,ca,F,df):
    return ((1-F(xia))*(paa-ca) - F(xib)*(pab-ca))/(1-df+df*(F(xia)+F(xib)))

def vbopt_diff(pbb,pba,xia,xib,cb,F,df):
    return ((1-F(xib))*(pbb-cb) - F(xia)*(pba-cb))/(1-df+df*(F(xia)+F(xib)))

def pax_equations(candidates,xia,xib,ca,cb,la,lb,F,f,df):
    paa,pab = candidates
    if xia<0 or xia>1 or xib <0 or xib > 1:
        return (math.inf,math.inf)
    vaopt = vaopt_diff(paa,pab,xia,xib,ca,F,df)
    residual1 = paa -(ca + (1-F(xia))*la/f(xia) -df*vaopt)
    residual2 = pab -(ca + F(xib)*lb/f(xib) -df*vaopt)
    return (residual1,residual2)

def pbx_equations(candidates,xia,xib,ca,cb,la,lb,F,f,df):
    pbb,pba = candidates
    if xia<0 or xia>1 or xib <0 or xib > 1:
        return (math.inf,math.inf)
    vbopt = vbopt_diff(pbb,pba,xia,xib,cb,F,df)
    residual1 = pbb - (cb + (1-F(xib))*lb/f(xib) -df*vbopt)
    residual2 = pba - (cb + F(xia)*la/f(xia) -df*vbopt)
    return (residual1,residual2)

def constraints_state_a(paa,pba,ca,cb,la):
    if paa >= ca and pba >= cb and (0 <= paa-pba) and  (paa-pba <= la):
        return True
    return False

def constraints_state_b(pbb,pab,ca,cb,lb):
    if pbb >= cb and pab >= ca and (0 <= pbb-pab) and  (pbb-pab <= lb):
        return True
    return False

def get_prices_xis_constraints_and_vs(ca_arr,cb,la,lb,F,f,df):
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
        xia_arr[i],xib_arr[i] = fsolve(xi_equations, (0.5, 0.5),(ca,cb,la,lb,F,f,df))
        paa_arr[i],pab_arr[i] = fsolve(pax_equations, (0.5, 0.5),(xia_arr[i],xib_arr[i],ca,cb,la,lb,F,f,df))
        pbb_arr[i],pba_arr[i] = fsolve(pbx_equations, (0.5, 0.5),(xia_arr[i],xib_arr[i],ca,cb,la,lb,F,f,df))
        
        vaopt_diff_arr[i] = vaopt_diff(paa_arr[i],pab_arr[i],xia_arr[i],xib_arr[i],ca,F,df)
        vbopt_diff_arr[i] = vbopt_diff(pbb_arr[i],pba_arr[i],xia_arr[i],xib_arr[i],cb,F,df)
        
        if constraints_state_a(paa_arr[i],pba_arr[i],ca,cb,la):
            constraint_aa_ba_arr[i] = 1
        if constraints_state_b(pbb_arr[i],pab_arr[i],ca,cb,lb):
            constraint_bb_ab_arr[i] = 1

    return paa_arr,pba_arr,pbb_arr,pab_arr,xia_arr,xib_arr,constraint_aa_ba_arr,constraint_bb_ab_arr,vaopt_diff_arr,vbopt_diff_arr

def get_vopts(paa,pab,pbb,pba,xia,xib,da,db,ca,cb,F):

    #vaao, vabo
    mat1 = np.array([[1-da*(1-F(xia)),-da*F(xia)],[-da*F(xib),1-da*(1-F(xib))]])
    rhs1 = np.array([(1-F(xia))*(paa-ca),F(xib)*(pab-ca)])
    sol1 = np.linalg.inv(mat1).dot(rhs1)
    vaao,vabo = sol1[0],sol1[1]

    mat2 = np.array([[1-db*(1-F(xib)),-db*F(xib)],[-db*F(xia),1-db*(1-F(xia))]])
    rhs2 = np.array([(1-F(xib))*(pbb-cb),F(xia)*(pba-cb)])
    sol2 = np.linalg.inv(mat2).dot(rhs2)
    vbbo,vbao = sol2[0],sol2[1]

    return vaao,vabo,vbbo,vbao

def get_theoretical_px_obj(ca,cb,la,lb,dist,df):

    _,F,f    = get_xi_dist(dist)

    # xia,xib = fsolve(xi_equations, (0.5, 0.5),(ca,cb,la,lb,F,f,df))

    paa_arr,pba_arr,pbb_arr,pab_arr,xia_arr,xib_arr,constraint_aa_ba_arr,constraint_bb_ab_arr,\
        vaopt_diff_arr,vbopt_diff_arr = get_prices_xis_constraints_and_vs(np.array([ca]),cb,la,lb,F,f,df)

    vaao,vabo,vbbo,vbao = get_vopts(paa_arr[0],pab_arr[0],pbb_arr[0],pba_arr[0],xia_arr[0],xib_arr[0],df,df,ca,cb,F)

    return vaao,vabo,vbbo,vbao,paa_arr[0],pab_arr[0],pbb_arr[0],pba_arr[0],constraint_aa_ba_arr[0],constraint_bb_ab_arr[0]


'''
There are two states: customer is either in set alpha, or in set beta.
If the customer is in state alpha, the two firms show one price each. 
Similarly they show one price each when the customer is in state beta.



'''


def get_common_price_spaces(ca,cb,maxpx,npts):
    pa_common_arr = np.linspace(ca,maxpx,npts) #A's price for its strong sub-market
    pb_common_arr = np.linspace(cb,maxpx,npts) #B's price for its weak sub-market

    return pa_common_arr,pb_common_arr

def get_payoff_matrices_state_a(ca,cb,maxpx,npts,dist,la):

    _,F,f = get_xi_dist(dist)
    pa_state_a_arr,pb_state_a_arr = get_common_price_spaces(ca,cb,maxpx,npts)
    '''
     pa_state_a_arr : array of A's prices for its strong sub-market
     pb_state_a_arr: array of B's prices for its weak sub-market
    '''

    obja_state_a = np.zeros((pa_state_a_arr.size,pb_state_a_arr.size)) #A's obj for its strong sub-market
    objb_state_a = np.zeros((pa_state_a_arr.size,pb_state_a_arr.size)) #B's obj for its weak sub-market
    constraint_state_a = np.zeros((pa_state_a_arr.size,pb_state_a_arr.size)) #px constraints related to A's strong sub-market


    for i,paa in enumerate(pa_state_a_arr): #firm A is the row player
        for j,pba in enumerate(pb_state_a_arr):
            if constraints_state_a(paa,pba,ca,cb,la):
                constraint_state_a[i,j] = 1 
                obja_state_a[i,j] = get_payoff_aa(paa,ca,F,pba,la)
                objb_state_a[i,j] = get_payoff_ba(pba,cb,F,paa,la)

    return pa_state_a_arr,pb_state_a_arr,obja_state_a,objb_state_a,constraint_state_a

def get_payoff_matrices_state_b(ca,cb,maxpx,npts,dist,lb):

    _,F,f = get_xi_dist(dist)
    pa_state_b_arr,pb_state_b_arr = get_common_price_spaces(ca,cb,maxpx,npts)
    '''
     pa_state_b_arr : array of A's prices for its weak sub-market
     pb_state_b_arr: array of B's prices for its strong sub-market
    '''

    obja_state_b = np.zeros((pa_state_b_arr.size,pb_state_b_arr.size)) #A's obj for its weak sub-market
    objb_state_b = np.zeros((pa_state_b_arr.size,pb_state_b_arr.size)) #B's obj for its strong sub-market
    constraint_state_b = np.zeros((pa_state_b_arr.size,pb_state_b_arr.size)) #px constraints related to B's strong sub-market


    for i,pab in enumerate(pa_state_b_arr): #firm A is the row player
        for j,pbb in enumerate(pb_state_b_arr):
            if constraints_state_b(pbb,pab,ca,cb,lb):
                constraint_state_b[i,j] = 1 
                obja_state_b[i,j] = get_payoff_ab(pab,ca,F,pbb,lb)
                objb_state_b[i,j] = get_payoff_bb(pbb,cb,F,pab,lb)

    return pa_state_b_arr,pb_state_b_arr,obja_state_b,objb_state_b,constraint_state_b


def get_transition_prob_matrices(ca,cb,maxpx,npts,dist,la,lb):

    _,F,f = get_xi_dist(dist)
    pa_arr,pb_arr = get_common_price_spaces(ca,cb,maxpx,npts)

    transition_prob_matrices = []
    #state:customer is in set alpha
    transition_prob_matrix = np.zeros((len(pa_arr),len(pb_arr),2))
    for i,pa in enumerate(pa_arr):
        for j,pb in enumerate(pb_arr):
            transition_prob_matrix[i,j,0] = prob_cust_a_purchase_from_a(pa,pb,la,F)  #from alpha to set alpha
            transition_prob_matrix[i,j,1] = 1-transition_prob_matrix[i,j,0] #transitioning to set beta

    transition_prob_matrices.append(transition_prob_matrix)
    #state:customer is in set beta
    transition_prob_matrix = np.zeros((len(pa_arr),len(pb_arr),2))
    for i,pa in enumerate(pa_arr):
        for j,pb in enumerate(pb_arr):
            transition_prob_matrix[i,j,1] = prob_cust_b_purchase_from_b(pb,pa,lb,F)#transitioning to set beta
            transition_prob_matrix[i,j,0] = 1-transition_prob_matrix[i,j,1]  #from beta to set alpha

    transition_prob_matrices.append(transition_prob_matrix)

    return transition_prob_matrices



def get_computed_equilibrium(payoff_matrices,transition_prob_matrices,discount_factors,pa_arr,pb_arr,show_progress=True,plot_path=True):
    equilibrium = dsSolve(payoff_matrices,transition_prob_matrices,discount_factors,show_progress,plot_path)
    eq_indices1 = np.argmax(equilibrium['strategies'][0],axis=1)
    temp1 = [pa_arr[eq_indices1[0]],pb_arr[eq_indices1[1]],equilibrium['stateValues'][0][0],equilibrium['stateValues'][0][1]]
    temp1a = ['paa','pba','vaa','vba']
    result1 = {temp1a[i]:np.round(x,3) for i,x in enumerate(temp1)}
    eq_indices2 = np.argmax(equilibrium['strategies'][1],axis=1)
    temp2 = [pa_arr[eq_indices2[0]],pb_arr[eq_indices2[1]],equilibrium['stateValues'][1][0],equilibrium['stateValues'][1][1]]
    temp2a = ['pab','pbb','vab','vbb']
    result2 = {temp2a[i]:np.round(x,3) for i,x in enumerate(temp2)}
    return result1,result2

def get_computed_px_obj(ca,cb,la,lb,dist,deltaf,maxpx=10,npts=20,show_progress=False,plot_path=False):
    #data for solver: payoff matrices
    pa_arr,pb_arr,obja_state_a,objb_state_a,\
    constraint_state_a = get_payoff_matrices_state_a(ca,cb,maxpx,npts,dist,la)
    _,_,obja_state_b,objb_state_b,\
    constraint_state_b = get_payoff_matrices_state_b(ca,cb,maxpx,npts,dist,lb)
    payoff_matrices = [ ## s = \alpha
                        np.array([obja_state_a,objb_state_a]),
                        ## s = \beta
                        np.array([obja_state_b,objb_state_b])]

    transition_prob_matrices = get_transition_prob_matrices(ca,cb,maxpx,npts,dist,la,lb)
    result1_c,result2_c = get_computed_equilibrium(payoff_matrices,transition_prob_matrices,deltaf,
                             pa_arr,pb_arr,show_progress,plot_path)

    return result1_c,result2_c
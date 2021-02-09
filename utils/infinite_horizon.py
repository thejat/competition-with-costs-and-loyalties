from utils.imports import *
from utils.gameSolver import dsSolve
from utils.single_stage import get_xi_dist,\
    ll_constraint,\
    firm_constraint_cost,\
    firm_constraint_across,\
    ll_get_both_payoff_matrices,\
    ll_prob_cust_a_purchase_from_a,\
    ll_prob_cust_b_purchase_from_b

'''
In the below variables, first index is always firm, and second index is the strong sub-market
For instance, in paa_arr, first 'a' represents firm A's pricing, second 'a' represents A's strong sub-market 'alpha'

There are two states: customer is either in set alpha, or in set beta.
If the customer is in state alpha, the two firms show one price each. 
Similarly they show one price each when the customer is in state beta.

'''


def compute_infinite_horizon_equilibrium(payoff_matrices, pa_arr, pb_arr, transition_prob_matrices, discount_factors, show_progress=True, plot_path=True):
    """
    Compute the equlibrium using the Homotopy method and the quantal/logit approximation idea.
    """
    equilibrium = dsSolve(payoff_matrices, transition_prob_matrices,
                          discount_factors, show_progress, plot_path)

    # logging
    eq_indices_firm_A = np.argmax(equilibrium['strategies'][0], axis=1)
    sol_values_firm_A = [pa_arr[eq_indices_firm_A[0]], pb_arr[eq_indices_firm_A[1]],
                         equilibrium['stateValues'][0][0], equilibrium['stateValues'][0][1]]
    sol_names_firm_A = ['paa', 'pba', 'vaa', 'vba']
    result_firm_A = {sol_names_firm_A[i]: np.round(
        x, 3) for i, x in enumerate(sol_values_firm_A)}

    eq_indices_firm_B = np.argmax(equilibrium['strategies'][1], axis=1)
    sol_values_firm_B = [pa_arr[eq_indices_firm_B[0]], pb_arr[eq_indices_firm_B[1]],
                         equilibrium['stateValues'][1][0], equilibrium['stateValues'][1][1]]
    sol_names_firm_B = ['pab', 'pbb', 'vab', 'vbb']
    result_firm_B = {sol_names_firm_B[i]: np.round(
        x, 3) for i, x in enumerate(sol_values_firm_B)}

    return result_firm_A, result_firm_B


def ll_get_metrics_theory(ca, cb, F, f, deltaf, dist, la, lb, sa, sb):

    # agnostic to the loyalty model (next three functions)

    def vaopt_diff_ll(paa, pab, xia, xib, ca, F, deltaf):
        return ((1-F(xia))*(paa-ca) - F(xib)*(pab-ca))/(1-deltaf+deltaf*(F(xia)+F(xib)))

    def vbopt_diff_ll(pbb, pba, xia, xib, cb, F, deltaf):
        return ((1-F(xib))*(pbb-cb) - F(xia)*(pba-cb))/(1-deltaf+deltaf*(F(xia)+F(xib)))

    def get_vopts_ll(paa, pab, pbb, pba, xia, xib, da, db, ca, cb, F):
        """
        Eq above 15
        """
        #vaao, vabo
        mat1 = np.array([[1-da*(1-F(xia)), -da*F(xia)],
                         [-da*F(xib), 1-da*(1-F(xib))]])
        rhs1 = np.array([(1-F(xia))*(paa-ca), F(xib)*(pab-ca)])
        sol1 = np.linalg.inv(mat1).dot(rhs1)
        vaao, vabo = sol1[0], sol1[1]

        mat2 = np.array([[1-db*(1-F(xib)), -db*F(xib)],
                         [-db*F(xia), 1-db*(1-F(xia))]])
        rhs2 = np.array([(1-F(xib))*(pbb-cb), F(xia)*(pba-cb)])
        sol2 = np.linalg.inv(mat2).dot(rhs2)
        vbbo, vbao = sol2[0], sol2[1]

        return vaao, vabo, vbbo, vbao

    # specific to multiplicative loyalty model (next three functions)

    def xi_equations_ml(candidates, ca, cb, la, lb, F, f, deltaf):
        gammaa = (ca-cb)/la
        gammab = (cb-ca)/lb
        delbydel = (1-deltaf)/deltaf

        xia, xib = candidates
        if xia < 0 or xia > 1 or xib < 0 or xib > 1:  # hardcoded for cdf between 0 and 1: TODO
            return (math.inf, math.inf)

        residual1 = (xia-gammaa)*(delbydel+F(xib)+1) \
            + ((2*F(xia)-1)/f(xia))*(delbydel + F(xib) + F(xia)) \
            + (F(xia)/f(xia)) \
            - (((1-F(xib))*lb)/(la*f(xib)) - F(xib)*(gammaa + xib*lb/la))
        residual2 = (xib-gammab)*(delbydel+F(xia)+1) \
            + ((2*F(xib)-1)/f(xib))*(delbydel + F(xib) + F(xia)) \
            + (F(xib)/f(xib)) \
            - (((1-F(xia))*la)/(lb*f(xia)) - F(xia)*(gammab + xia*la/lb))

        return (residual1, residual2)

    def pax_equations_ml(candidates, xia, xib, ca, cb, la, lb, F, f, deltaf):
        """
        Eq 9, 10
        """
        paa, pab = candidates
        if xia < 0 or xia > 1 or xib < 0 or xib > 1:
            return (math.inf, math.inf)
        vaopt = vaopt_diff_ll(paa, pab, xia, xib, ca, F, deltaf)
        residual1 = paa - (ca + (1-F(xia))*la/f(xia) - deltaf*vaopt)
        residual2 = pab - (ca + F(xib)*lb/f(xib) - deltaf*vaopt)
        return (residual1, residual2)

    def pbx_equations_ml(candidates, xia, xib, ca, cb, la, lb, F, f, deltaf):
        """
        Eq 11, 12
        """
        pbb, pba = candidates
        if xia < 0 or xia > 1 or xib < 0 or xib > 1:
            return (math.inf, math.inf)
        vbopt = vbopt_diff_ll(pbb, pba, xia, xib, cb, F, deltaf)
        residual1 = pbb - (cb + (1-F(xib))*lb/f(xib) - deltaf*vbopt)
        residual2 = pba - (cb + F(xia)*la/f(xia) - deltaf*vbopt)
        return (residual1, residual2)

    # specific to additive loyalty model (next three functions)

    def xi_equations_al(candidates, ca, cb, sa, sb, F, f, deltaf):
        gammaa = (ca - cb - sa)
        gammab = (cb - ca - sb)
        delbydel = (1-deltaf)/deltaf

        xia, xib = candidates
        if xia < 0 or xia > 1 or xib < 0 or xib > 1:  # hardcoded for cdf between 0 and 1: TODO
            return (math.inf, math.inf)

        residual1 = (xia-gammaa)*(delbydel+F(xib)+1) \
            + ((2*F(xia)-1)/f(xia))*(delbydel + F(xib) + F(xia)) \
            + (F(xia)/f(xia)) \
            - ((1-F(xib))/f(xib) - F(xib)*(xib - gammab))
        residual2 = (xib-gammab)*(delbydel+F(xia)+1) \
            + ((2*F(xib)-1)/f(xib))*(delbydel + F(xib) + F(xia)) \
            + (F(xib)/f(xib)) \
            - ((1-F(xia))/f(xia) - F(xia)*(xia - gammaa))

        return (residual1, residual2)

    def pax_equations_al(candidates, xia, xib, ca, cb, F, f, deltaf):
        """
        Eq 9, 10
        """
        paa, pab = candidates
        if xia < 0 or xia > 1 or xib < 0 or xib > 1:
            return (math.inf, math.inf)
        vaopt = vaopt_diff_ll(paa, pab, xia, xib, ca, F, deltaf)
        residual1 = paa - (ca + (1-F(xia))/f(xia) - deltaf*vaopt)
        residual2 = pab - (ca + F(xib)/f(xib) - deltaf*vaopt)
        return (residual1, residual2)

    def pbx_equations_al(candidates, xia, xib, ca, cb, F, f, deltaf):
        """
        Eq 11, 12
        """
        pbb, pba = candidates
        if xia < 0 or xia > 1 or xib < 0 or xib > 1:
            return (math.inf, math.inf)
        vbopt = vbopt_diff_ll(pbb, pba, xia, xib, cb, F, deltaf)
        residual1 = pbb - (cb + (1-F(xib))/f(xib) - deltaf*vbopt)
        residual2 = pba - (cb + F(xia)/f(xia) - deltaf*vbopt)
        return (residual1, residual2)

    if sa == 0 and sb == 0:  # ml

        xia, xib = fsolve(xi_equations_ml,  (.5, .5),
                          (ca, cb, la, lb, F, f, deltaf))  # hardcoded initial point for fsolve
        paa, pab = fsolve(pax_equations_ml, (.5, .5), (xia, xib, ca,
                                                       cb, la, lb, F, f, deltaf))  # hardcoded initial point for fsolve
        pbb, pba = fsolve(pbx_equations_ml, (.5, .5), (xia, xib, ca,
                                                       cb, la, lb, F, f, deltaf))  # hardcoded initial point for fsolve
        vaao, vabo, vbbo, vbao = get_vopts_ll(
            paa, pab, pbb, pba, xia, xib, deltaf, deltaf, ca, cb, F)

    elif la == 1 and lb == 1:  # al

        xia, xib = fsolve(xi_equations_al,  (.5, .5),
                          (ca, cb, sa, sb, F, f, deltaf))  # hardcoded initial point for fsolve
        paa, pab = fsolve(pax_equations_al, (.5, .5), (xia, xib, ca,
                                                       cb, F, f, deltaf))  # hardcoded initial point for fsolve
        pbb, pba = fsolve(pbx_equations_al, (.5, .5), (xia, xib, ca,
                                                       cb, F, f, deltaf))  # hardcoded initial point for fsolve
        vaao, vabo, vbbo, vbao = get_vopts_ll(
            paa, pab, pbb, pba, xia, xib, deltaf, deltaf, ca, cb, F)

    else:
        return NotImplementedError

    return paa, pab, pbb, pba, xia, xib, vaao, vabo, vbbo, vbao


def ll_get_metrics_computed(dist, ca, cb, F, f, deltaf, la, lb, sa, sb, maxpx=10, npts=20, show_progress=False, plot_path=False):  # TODO: change to ll

    def ll_get_transition_prob_matrices(ca, cb, maxpx, npts, F, f, la, lb, sa, sb):

        pa_arr = np.linspace(ca, maxpx, npts)
        pb_arr = np.linspace(cb, maxpx, npts)

        transition_prob_matrices = []
        # state:customer is in set alpha
        transition_prob_matrix = np.zeros((len(pa_arr), len(pb_arr), 2))
        for i, pa in enumerate(pa_arr):
            for j, pb in enumerate(pb_arr):
                transition_prob_matrix[i, j, 0] = ll_prob_cust_a_purchase_from_a(
                    pa, pb, F, la, sa)
                # transitioning to set beta
                transition_prob_matrix[i, j, 1] = 1 - \
                    transition_prob_matrix[i, j, 0]

        transition_prob_matrices.append(transition_prob_matrix)
        # state:customer is in set beta
        transition_prob_matrix = np.zeros((len(pa_arr), len(pb_arr), 2))
        for i, pa in enumerate(pa_arr):
            for j, pb in enumerate(pb_arr):
                transition_prob_matrix[i, j, 1] = ll_prob_cust_b_purchase_from_b(
                    pb, pa, F, lb, sb)
                # from beta to set alpha
                transition_prob_matrix[i, j, 0] = 1 - \
                    transition_prob_matrix[i, j, 1]

        transition_prob_matrices.append(transition_prob_matrix)

        return transition_prob_matrices

    # data for solver: transition probability matrices
    transition_prob_matrices = ll_get_transition_prob_matrices(
        ca, cb, maxpx, npts, F, f, la, lb, sa, sb)

    # data for solver: immediate payoff matrices
    pa_arr, pb_arr, payoff_matrices, _ = ll_get_both_payoff_matrices(
        dist, maxpx, npts, ca, cb, la, lb, sa, sb)

    result1_c, result2_c = compute_infinite_horizon_equilibrium(
        payoff_matrices, pa_arr, pb_arr, transition_prob_matrices, deltaf, show_progress, plot_path)

    xia = (result1_c['paa'] - result1_c['pba'])/la
    xib = (result2_c['pbb'] - result2_c['pab'])/lb

    return result1_c['paa'], result2_c['pab'], result2_c['pbb'], result1_c['pba'], xia, xib, result1_c['vaa'], result2_c['vab'], result2_c['vbb'], result1_c['vba']


def ll_get_metric_arrs_vs_camcb(dist, deltaf, ca_arr, cb, la=1, lb=1, sa=0, sb=0, flag_theory=True, maxpx=10, npts=20, show_progress=False, plot_path=False):
    """Compute the market outcomes as a function of cost asymmetry
    """

    def ll_get_market_shares(paa, pba, pbb, pab, F, la, lb, sa, sb):

        praa = ll_prob_cust_a_purchase_from_a(paa, pba, F, la, sa)
        prbb = ll_prob_cust_b_purchase_from_b(pbb, pab, F, lb, sb)

        A = np.array([[praa-1, 1-prbb], [1, 1]])
        b = np.array([0, 1])
        thetavec = np.linalg.solve(A, b)
        new_market_share_a = thetavec[0]
        new_market_share_b = thetavec[1]
        return (new_market_share_a, new_market_share_b)

    def ll_get_total_profits(vaa, vab, vbb, vba, theta):
        total_profit_a = theta*vaa + (1-theta)*vab
        total_profit_b = (1-theta)*vbb + theta*vba
        return (total_profit_a, total_profit_b)

    if dist != 'uniform':
        return NotImplementedError()

    print('ll_get_metric_arrs_vs_camcb start: ', datetime.datetime.now())

    F, f = get_xi_dist(dist)

    # first index is firm, second index is customer type
    paa_arr = np.zeros(ca_arr.size)
    pba_arr = np.zeros(ca_arr.size)
    pbb_arr = np.zeros(ca_arr.size)
    pab_arr = np.zeros(ca_arr.size)
    xia_arr = np.zeros(ca_arr.size)
    xib_arr = np.zeros(ca_arr.size)
    constraint_aa_ba_arr = np.zeros(ca_arr.size)
    constraint_bb_ab_arr = np.zeros(ca_arr.size)
    constraint_cross_a_arr = np.zeros(ca_arr.size)
    constraint_cross_b_arr = np.zeros(ca_arr.size)
    vaao_arr = np.zeros(ca_arr.size)
    vabo_arr = np.zeros(ca_arr.size)
    vbbo_arr = np.zeros(ca_arr.size)
    vbao_arr = np.zeros(ca_arr.size)
    marketshare_a_arr = np.zeros(ca_arr.size)
    marketshare_b_arr = np.zeros(ca_arr.size)
    total_profit_a_arr = np.zeros(ca_arr.size)
    total_profit_b_arr = np.zeros(ca_arr.size)
    prob_purchase_a_from_a_arr = np.zeros(ca_arr.size)
    prob_purchase_b_from_b_arr = np.zeros(ca_arr.size)

    for i, ca in enumerate(ca_arr):
        print('i,ca,time: ', i, np.round(ca, 3), datetime.datetime.now())

        if flag_theory:
            paa_arr[i], pab_arr[i], pbb_arr[i], pba_arr[i], xia_arr[i], xib_arr[i], vaao_arr[i], vabo_arr[i], vbbo_arr[i], vbao_arr[i]  \
                = ll_get_metrics_theory(ca, cb, F, f, deltaf, dist, la, lb, sa, sb)
        else:
            paa_arr[i], pab_arr[i], pbb_arr[i], pba_arr[i], xia_arr[i], xib_arr[i], vaao_arr[i], vabo_arr[i], vbbo_arr[i], vbao_arr[i]  \
                = ll_get_metrics_computed(dist, ca, cb, F, f, deltaf, la, lb, sa, sb, maxpx, npts, show_progress, plot_path)

        # logging
        if ll_constraint(paa_arr[i], pba_arr[i], la, sa, dist) and firm_constraint_cost(paa_arr[i], ca) and firm_constraint_cost(pba_arr[i], cb):
            constraint_aa_ba_arr[i] = 1
        if ll_constraint(pbb_arr[i], pab_arr[i], lb, sb, dist) and firm_constraint_cost(pbb_arr[i], cb) and firm_constraint_cost(pab_arr[i], ca):
            constraint_bb_ab_arr[i] = 1
        # this constraint is not being imposed while defining feasible actions
        if firm_constraint_across(paa_arr[i], pab_arr[i]):
            constraint_cross_a_arr[i] = 1
        if firm_constraint_across(pbb_arr[i], pba_arr[i]):
            constraint_cross_b_arr[i] = 1

        marketshare_a_arr[i], marketshare_b_arr[i] = ll_get_market_shares(
            paa_arr[i], pba_arr[i], pbb_arr[i], pab_arr[i], F, la, lb, sa, sb)
        total_profit_a_arr[i], total_profit_b_arr[i] = ll_get_total_profits(
            vaao_arr[i], vabo_arr[i], vbbo_arr[i], vbao_arr[i], marketshare_a_arr[i])
        prob_purchase_a_from_a_arr[i] = ll_prob_cust_a_purchase_from_a(
            paa_arr[i], pba_arr[i], F, la, sa)
        prob_purchase_b_from_b_arr[i] = ll_prob_cust_b_purchase_from_b(
            pbb_arr[i], pab_arr[i], F, lb, sb)

    print('ll_get_metric_arrs_vs_camcb end: ', datetime.datetime.now())

    return pd.DataFrame({'paa': paa_arr, 'pba': pba_arr, 'pbb': pbb_arr, 'pab': pab_arr,
                         'vaa': vaao_arr, 'vba': vbao_arr, 'vbb': vbbo_arr, 'vab': vabo_arr,
                         'marketshare_a': marketshare_a_arr, 'marketshare_b': marketshare_b_arr,
                         'total_profit_a': total_profit_a_arr, 'total_profit_b': total_profit_b_arr,
                         'prob_purchase_a_from_a': prob_purchase_a_from_a_arr,
                         'prob_purchase_b_from_b': prob_purchase_b_from_b_arr,
                         'constraint_aa_ba_arr': constraint_aa_ba_arr, 'constraint_bb_ab_arr': constraint_bb_ab_arr,
                         'constraint_cross_a_arr': constraint_cross_a_arr, 'constraint_cross_b_arr': constraint_cross_b_arr,
                         'xia': xia_arr, 'xib': xib_arr
                         })

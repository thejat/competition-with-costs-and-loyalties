from utils.imports import *
from utils.gameSolver import dsSolve

'''
In the variables below, first index is always firm, and second index is the strong sub-market
For instance, in paa_arr, first 'a' represents firm A's pricing, second 'a' represents A's strong sub-market 'alpha'
'''

### General loyalty model

def firm_constraint_across(p_strong_market,p_weak_market):
	if (p_strong_market >= p_weak_market): return True
	return False

def firm_constraint_cost(px,cost):
	if px >= cost: return True
	return False

def get_xi_dist(dist='normal'):

	if dist=='uniform':
		return (uniform.cdf,uniform.pdf)
	else:
		return NotImplementedError # (norm.cdf,norm.pdf)

def get_common_price_spaces(ca,cb,maxpx,npts):
	pa_common_arr = np.linspace(ca,maxpx,npts) #A's price for its strong sub-market
	pb_common_arr = np.linspace(cb,maxpx,npts) #B's price for its weak sub-market
	return pa_common_arr,pb_common_arr

def compute_single_stage_equilibrium(objaa,objba,objbb,objab,paa_arr,pba_arr,pbb_arr,pab_arr):

	def get_computed_equilibrium(payoffs,p1_arr,p2_arr):
		equilibrium = dsSolve(payoffs)
		eq_indices = np.argmax(equilibrium['strategies'][0],axis=1)
		temp = [p1_arr[eq_indices[0]],p2_arr[eq_indices[1]],equilibrium['stateValues'][0][0],equilibrium['stateValues'][0][1]]
		return (np.round(x,3) for x in temp)

	paa_c,pba_c,objaa_c,objba_c = get_computed_equilibrium([np.array([objaa,objba])],paa_arr,pba_arr)
	pab_c,pbb_c,objab_c,objbb_c = get_computed_equilibrium([np.array([objab,objbb])],pab_arr,pbb_arr)

	return {'paa':paa_c,'pba':pba_c,'pbb':pbb_c,'pab':pab_c,'objaa':objaa_c,'objba':objba_c,'objbb':objbb_c,'objab':objab_c}


### Multiplicative loyalty model

def ml_prob_cust_a_purchase_from_a(paa,pba,la,F): return 1-F((paa-pba)/la)
def ml_prob_cust_b_purchase_from_b(pbb,pab,lb,F): return 1-F((pbb-pab)/lb)
def ml_get_payoff_aa(paa,ca,F,pba,la): return (paa-ca)*ml_prob_cust_a_purchase_from_a(paa,pba,la,F) #caution: theta is ignored here
def ml_get_payoff_ba(pba,cb,F,paa,la): return (pba-cb)*(1-ml_prob_cust_a_purchase_from_a(paa,pba,la,F))
def ml_get_payoff_bb(pbb,cb,F,pab,lb): return (pbb-cb)*ml_prob_cust_b_purchase_from_b(pbb,pab,lb,F)
def ml_get_payoff_ab(pab,ca,F,pbb,lb): return (pab-ca)*(1-ml_prob_cust_b_purchase_from_b(pbb,pab,lb,F))

def ml_get_market_shares(paa,pba,pbb,pab,la,lb,F,theta): 
	new_market_share_a = theta*ml_prob_cust_a_purchase_from_a(paa,pba,la,F) + (1-theta)*(1-ml_prob_cust_b_purchase_from_b(pbb,pab,lb,F))
	new_market_share_b = (1-theta)*ml_prob_cust_b_purchase_from_b(pbb,pab,lb,F) + theta*(1-ml_prob_cust_a_purchase_from_a(paa,pba,la,F))
	return (new_market_share_a,new_market_share_b)

def ml_get_total_profits(paa,pba,pbb,pab,la,lb,ca,cb,F,theta):
	total_profit_a = theta*ml_get_payoff_aa(paa,ca,F,pba,la) + (1-theta)*ml_get_payoff_ab(pab,ca,F,pbb,lb) # see Eq 1 in paper
	total_profit_b = (1-theta)*ml_get_payoff_bb(pbb,cb,F,pab,lb) + theta*ml_get_payoff_ba(pba,cb,F,paa,la)
	return (total_profit_a,total_profit_b)

def ml_get_ss_prices_theory(ca,cb,la,lb):
	#From the four propositions in the paper: ml-ss

	if (lb <= ca-cb) and (ca-cb < 2*la): #Region I
		paa_t = 0.33*(2*ca+cb+2*la) #suffix '_t' means theoretical/analytical
		pba_t = 0.33*(2*cb+ca+la)
		pbb_t = ca
		pab_t = ca
	elif ca-cb > min(2*la,lb): # Region II
		paa_t = ca
		pba_t = ca-la
		pbb_t = ca
		pab_t = ca
	elif ca-cb < min(2*la,lb): # Region III
		paa_t = 0.33*(2*ca+cb+2*la)
		pba_t = 0.33*(2*cb+ca+la)
		pbb_t = 0.33*(2*cb+ca+2*lb)
		pab_t = 0.33*(2*ca+cb+lb)
	elif (2*la < ca-cb) and (ca-cb < lb): # Region IV
		paa_t = ca
		pba_t = ca-la
		pbb_t = 0.33*(2*cb+ca+2*lb)
		pab_t = 0.33*(2*ca+cb+lb)
	else:
		print('region not defined')
		paa_t,pba_t,pbb_t,pab_t = [0]*4

	temp = [paa_t,pba_t,pbb_t,pab_t]
	return (np.round(x,3) for x in temp)

def ml_get_example_in_region(region=1,dist='uniform',deltaf=0):
	if dist != 'uniform':
		return NotImplementedError()

	#From the four propositions in the paper: ml-ss
	if region==1: 	# Region I:  lb < ca-cb < 2la
		ca,cb,la,lb    = 2,0,2,1
	elif region==2: # Region II: ca-cb > min(2la,lb)
		ca,cb,la,lb    = 5,0,2,1
	elif region==3: # Region III:  ca-cb < min(2la,lb)
		ca,cb,la,lb    = 1,0,2,2
	elif region==4: # Region IV:  2la < ca-cb < lb
		ca,cb,la,lb    = 3,0,1,4
	else:
		return NotImplementedError()

	instance = {'ca':ca,'cb':cb,'la':la,'lb':lb}
	print(instance)

	if deltaf<1e-2 and dist=='uniform':
		paa_t,pba_t,pbb_t,pab_t = ml_get_ss_prices_theory(ca,cb,la,lb)
		result_ssa = {'paa_sst':paa_t,'pba_sst':pba_t}
		result_ssb = {'pab_sst':pab_t,'pbb_sst':pbb_t}
		print({**result_ssa,**result_ssb})

	return (np.round(instance[x],3) for x in instance)

def ml_get_payoff_matrices_state_a(ca,cb,maxpx,npts,dist,la):

	F,f = get_xi_dist(dist)

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
			if ll_constraint(paa,pba,la,0,dist) and firm_constraint_cost(paa,ca) and firm_constraint_cost(pba,cb):
				constraint_state_a[i,j] = 1 
				obja_state_a[i,j] = ml_get_payoff_aa(paa,ca,F,pba,la)
				objb_state_a[i,j] = ml_get_payoff_ba(pba,cb,F,paa,la)

	return pa_state_a_arr,pb_state_a_arr,obja_state_a,objb_state_a,constraint_state_a

def ml_get_payoff_matrices_state_b(ca,cb,maxpx,npts,dist,lb):

	F,f = get_xi_dist(dist)

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
			if ll_constraint(pbb,pab,lb,0,dist) and firm_constraint_cost(pbb,cb) and firm_constraint_cost(pab,ca):
				constraint_state_b[i,j] = 1 
				obja_state_b[i,j] = ml_get_payoff_ab(pab,ca,F,pbb,lb)
				objb_state_b[i,j] = ml_get_payoff_bb(pbb,cb,F,pab,lb)

	return pa_state_b_arr,pb_state_b_arr,obja_state_b,objb_state_b,constraint_state_b

def ml_get_metric_arrs_vs_camcb(ca_arr,cb,la,lb,dist='uniform',theta=0.5,flag_theory=True):
	'''
	this function iterates over ca. cb is an input.
	'''
	if dist != 'uniform':
		return NotImplementedError()

	F,f = get_xi_dist(dist)

	paa_arr = np.zeros(ca_arr.size)
	pba_arr = np.zeros(ca_arr.size)
	pbb_arr = np.zeros(ca_arr.size)
	pab_arr = np.zeros(ca_arr.size)
	objaa_arr = np.zeros(ca_arr.size)
	objba_arr = np.zeros(ca_arr.size)
	objbb_arr = np.zeros(ca_arr.size)
	objab_arr = np.zeros(ca_arr.size)
	marketshare_a_arr = np.zeros(ca_arr.size)
	marketshare_b_arr = np.zeros(ca_arr.size)
	total_profit_a_arr = np.zeros(ca_arr.size)
	total_profit_b_arr = np.zeros(ca_arr.size)
	prob_purchase_a_from_a_arr = np.zeros(ca_arr.size)
	prob_purchase_b_from_b_arr = np.zeros(ca_arr.size)

	for i,ca in enumerate(ca_arr):
		if flag_theory is True:
			paa_arr[i],pba_arr[i],pbb_arr[i],pab_arr[i] = ml_get_ss_prices_theory(ca,cb,la,lb)
		else:
			return NotImplementedError()

		objaa_arr[i] = ml_get_payoff_aa(paa_arr[i],ca,F,pba_arr[i],la)
		objba_arr[i] = ml_get_payoff_ba(pba_arr[i],cb,F,paa_arr[i],la)
		objbb_arr[i] = ml_get_payoff_bb(pbb_arr[i],cb,F,pab_arr[i],lb)
		objab_arr[i] = ml_get_payoff_ab(pab_arr[i],ca,F,pbb_arr[i],lb)
		marketshare_a_arr[i],marketshare_b_arr[i] = ml_get_market_shares(paa_arr[i],pba_arr[i],pbb_arr[i],pab_arr[i],la,lb,F,theta)
		total_profit_a_arr[i],total_profit_b_arr[i] = ml_get_total_profits(paa_arr[i],pba_arr[i],pbb_arr[i],pab_arr[i],la,lb,ca,cb,F,theta)
		prob_purchase_a_from_a_arr[i] = ml_prob_cust_a_purchase_from_a(paa_arr[i],pba_arr[i],la,F)
		prob_purchase_b_from_b_arr[i] = ml_prob_cust_b_purchase_from_b(pbb_arr[i],pab_arr[i],lb,F)


	return pd.DataFrame({'paa':paa_arr,'pba':pba_arr,'pbb':pbb_arr,'pab':pab_arr,
				'objaa':objaa_arr,'objba':objba_arr,'objbb':objbb_arr,'objab':objab_arr,
				'marketshare_a':marketshare_a_arr,'marketshare_b':marketshare_b_arr,
				'total_profit_a':total_profit_a_arr,'total_profit_b':total_profit_b_arr,
				'prob_purchase_a_from_a':prob_purchase_a_from_a_arr,
				'prob_purchase_b_from_b':prob_purchase_b_from_b_arr})

### Linear loyalty model (subsumes multiplicative and additive)

def ll_constraint(p_firm,p_rival,l_firm,s_firm = 0,dist='uniform'):
	''' 
	let _firm suffix represent the firm for which state a implies customer is in its strong market
	e.g., in state where cust is in B's strong mkt and ml:  (0 <= pbb-pab) and  (pbb-pab <= lb) 
	'''
	if (s_firm <= p_firm-p_rival) and  \
		(p_firm-p_rival <= l_firm + s_firm):
		return True
	return False

def ll_prob_cust_a_purchase_from_a(paa,pba,la,F): return NotImplementedError
def ll_prob_cust_b_purchase_from_b(pbb,pab,lb,F): return NotImplementedError
def ll_get_payoff_aa(paa,ca,F,pba,la): return NotImplementedError
def ll_get_payoff_ba(pba,cb,F,paa,la): return NotImplementedError
def ll_get_payoff_bb(pbb,cb,F,pab,lb): return NotImplementedError
def ll_get_payoff_ab(pab,ca,F,pbb,lb): return NotImplementedError

def ll_get_market_shares(paa,pba,pbb,pab,la,lb,F,theta):
	return NotImplementedError
def ll_get_total_profits(paa,pba,pbb,pab,la,lb,ca,cb,F,theta):
	return NotImplementedError

def ll_get_ss_prices_theory(ca,cb,la,lb):
	return NotImplementedError

def ll_get_example_in_region(region=1,dist='uniform',deltaf=0):
	return NotImplementedError

def ll_get_payoff_matrices_state_a(ca,cb,maxpx,npts,dist,la):
	return NotImplementedError

def ll_get_payoff_matrices_state_b(ca,cb,maxpx,npts,dist,lb):
	return NotImplementedError

def ll_get_metric_arrs_vs_camcb(ca_arr,cb,la,lb,dist='uniform',theta=0.5,flag_theory=True):
	return NotImplementedError





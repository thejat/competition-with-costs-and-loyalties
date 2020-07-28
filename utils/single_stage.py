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

def ll_prob_cust_a_purchase_from_a(paa,pba,F,la,sa=0): return 1-F((paa-pba-sa)/la)
def ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb,sb=0): return 1-F((pbb-pab-sb)/lb)
def ll_get_individual_payoff_aa(paa,pba,ca,F,la,sa=0): return (paa-ca)*ll_prob_cust_a_purchase_from_a(paa,pba,F,la,sa)
def ll_get_individual_payoff_ab(pbb,pab,ca,F,lb,sb=0): return (pab-ca)*(1-ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb,sb))
def ll_get_individual_payoff_ba(paa,pba,cb,F,la,sa=0): return (pba-cb)*(1-ll_prob_cust_a_purchase_from_a(paa,pba,F,la,sa))
def ll_get_individual_payoff_bb(pbb,pab,cb,F,lb,sb=0): return (pbb-cb)*ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb,sb)

def ll_get_market_shares(paa,pba,pbb,pab,F,theta,la,lb,sa=0,sb=0): 
	new_market_share_a = theta*ll_prob_cust_a_purchase_from_a(paa,pba,F,la,sa) \
						+ (1-theta)*(1-ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb,sb))
	new_market_share_b = (1-theta)*ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb,sb) \
						+ theta*(1-ll_prob_cust_a_purchase_from_a(paa,pba,F,la,sa))
	return (new_market_share_a,new_market_share_b)

def ll_get_total_profits(paa,pba,pbb,pab,F,theta,ca,cb,la,lb,sa=0,sb=0):
	total_profit_a = theta*ll_get_individual_payoff_aa(paa,pba,ca,F,la,sa) \
					+ (1-theta)*ll_get_individual_payoff_ab(pbb,pab,ca,F,lb,sb)
	total_profit_b = (1-theta)*ll_get_individual_payoff_bb(pbb,pab,cb,F,lb,sb) \
					+ theta*ll_get_individual_payoff_ba(paa,pba,cb,F,la,sa)
	return (total_profit_a,total_profit_b)

##tbd
def ll_get_ss_prices_theory(ca,cb,la,lb,sa=0,sb=0):
	return NotImplementedError

def ll_get_example_in_region(region=1,dist='uniform',deltaf=0):
	return NotImplementedError

def ll_get_payoff_matrices(maxpx,npts,dist,ca,cb,la,lb,sa=0,sb=0):
	'''
	There are two states in the game. Depending on which market the customer is in (A's strong sub-market vs weak sub-market)
	'''
	F,f = get_xi_dist(dist)
	pa_arr,pb_arr = get_common_price_spaces(ca,cb,maxpx,npts)

	'''
	Generic price arrays pa_arr and pb_arr represent the following:
	 pa_state_a_arr : array of A's prices for its strong sub-market
	 pb_state_a_arr: array of B's prices for its weak sub-market

 	 pa_state_b_arr : array of A's prices for its weak sub-market
	 pb_state_b_arr: array of B's prices for its strong sub-market
	'''

	obja_state_a = np.zeros((pa_arr.size,pb_arr.size)) #A's obj for its strong sub-market
	objb_state_a = np.zeros((pa_arr.size,pb_arr.size)) #B's obj for its weak sub-market

	obja_state_b = np.zeros((pa_arr.size,pb_arr.size)) #A's obj for its weak sub-market
	objb_state_b = np.zeros((pa_arr.size,pb_arr.size)) #B's obj for its strong sub-market

	constraint_state_a = np.zeros((pa_arr.size,pb_arr.size)) #px constraints related to A's strong sub-market
	constraint_state_b = np.zeros((pa_arr.size,pb_arr.size)) #px constraints related to B's strong sub-market

	for i,paa in enumerate(pa_arr): #firm A is the row player
		for j,pba in enumerate(pb_arr):

			#Computing payoffs in state a
			if ll_constraint(paa,pba,la,sa,dist) and firm_constraint_cost(paa,ca) and firm_constraint_cost(pba,cb):
				constraint_state_a[i,j] = 1 
				obja_state_a[i,j] = ll_get_individual_payoff_aa(paa,pba,ca,F,la,sa)
				objb_state_a[i,j] = ll_get_individual_payoff_ba(paa,pba,cb,F,la,sa)

			#Computing payoffs in state a
			if ll_constraint(pbb,pab,lb,sb,dist) and firm_constraint_cost(pbb,cb) and firm_constraint_cost(pab,ca):
				constraint_state_b[i,j] = 1 
				obja_state_b[i,j] = ll_get_individual_payoff_ab(pbb,pab,ca,F,lb,sb)
				objb_state_b[i,j] = ll_get_individual_payoff_bb(pbb,pab,cb,F,lb,sb)

	payoff_matrices = [ ## s = \alpha
						np.array([obja_state_a,objb_state_a]),
						## s = \beta
						np.array([obja_state_b,objb_state_b])]

	constraint_matrices = [ ## s = \alpha
							constraint_state_a,
							## s = \beta
							constraint_state_b]

	return pa_arr,pb_arr,payoff_matrices,constraint_matrices

def ll_get_metric_arrs_vs_camcb(ca_arr,cb,la,lb,sa=0,sb=0,dist='uniform',theta=0.5,flag_theory=True):
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
			# paa_arr[i],pba_arr[i],pbb_arr[i],pab_arr[i] = ll_get_ss_prices_theory(ca,cb,la,lb,sa,sb)
		else:
			return NotImplementedError()

		objaa_arr[i] = ll_get_individual_payoff_aa(paa_arr[i],pba_arr[i],ca,F,la,sa)
		objba_arr[i] = ll_get_individual_payoff_ba(paa_arr[i],pba_arr[i],cb,F,la,sa)
		objbb_arr[i] = ll_get_individual_payoff_bb(pbb_arr[i],pab_arr[i],cb,F,lb,sb)
		objab_arr[i] = ll_get_individual_payoff_ab(pbb_arr[i],pab_arr[i],ca,F,lb,sb)
		marketshare_a_arr[i],marketshare_b_arr[i] \
			= ll_get_market_shares(paa_arr[i],pba_arr[i],pbb_arr[i],pab_arr[i],F,theta,la,lb,sa,sb)
		total_profit_a_arr[i],total_profit_b_arr[i] \
			= ll_get_total_profits(paa_arr[i],pba_arr[i],pbb_arr[i],pab_arr[i],F,theta,ca,cb,la,lb,sa,sb)
		prob_purchase_a_from_a_arr[i] = ll_prob_cust_a_purchase_from_a(paa_arr[i],pba_arr[i],F,la,sa)
		prob_purchase_b_from_b_arr[i] = ll_prob_cust_b_purchase_from_b(pbb_arr[i],pab_arr[i],F,lb,sb)


	return pd.DataFrame({'paa':paa_arr,'pba':pba_arr,'pbb':pbb_arr,'pab':pab_arr,
				'objaa':objaa_arr,'objba':objba_arr,'objbb':objbb_arr,'objab':objab_arr,
				'marketshare_a':marketshare_a_arr,'marketshare_b':marketshare_b_arr,
				'total_profit_a':total_profit_a_arr,'total_profit_b':total_profit_b_arr,
				'prob_purchase_a_from_a':prob_purchase_a_from_a_arr,
				'prob_purchase_b_from_b':prob_purchase_b_from_b_arr})




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

def compute_single_stage_equilibrium(payoff_matrices,pa_arr,pb_arr,show_progress=False,plot_path=False):
	#TODO: make payoff_matrices_w_info dict as the input
	def get_computed_equilibrium(payoffs,p1_arr,p2_arr):
		equilibrium = dsSolve(payoffs)
		eq_indices = np.argmax(equilibrium['strategies'][0],axis=1)
		temp = [p1_arr[eq_indices[0]],p2_arr[eq_indices[1]],equilibrium['stateValues'][0][0],equilibrium['stateValues'][0][1]]
		return (np.round(x,3) for x in temp)

	paa_c,pba_c,objaa_c,objba_c = get_computed_equilibrium([payoff_matrices[0]],pa_arr,pb_arr)
	pab_c,pbb_c,objab_c,objbb_c = get_computed_equilibrium([payoff_matrices[1]],pa_arr,pb_arr)

	return {'paa':paa_c,'pba':pba_c,'pbb':pbb_c,'pab':pab_c,'objaa':objaa_c,'objba':objba_c,'objbb':objbb_c,'objab':objab_c}

### Multiplicative loyalty model

def ml_get_example_in_region(region=1,dist='uniform'):
	if dist != 'uniform':
		return NotImplementedError()

	#From the four propositions in the paper: ml-ss
	if region==1: 	# Region I:  lb < ca-cb < 2la
		ca,cb,la,lb,sa,sb    = 2,0,2,1,0,0
	elif region==2: # Region II: ca-cb > min(2la,lb)
		ca,cb,la,lb,sa,sb    = 5,0,2,1,0,0
	elif region==3: # Region III:  ca-cb < min(2la,lb)
		ca,cb,la,lb,sa,sb    = 1,0,2,2,0,0
	elif region==4: # Region IV:  2la < ca-cb < lb
		ca,cb,la,lb,sa,sb    = 3,0,1,4,0,0
	else:
		return NotImplementedError()

	return ca,cb,la,lb,sa,sb

### Additive loyalty model

def al_get_example_in_region(region=1,dist='uniform'):
	if dist != 'uniform':
		return NotImplementedError()

	#From the four propositions in the paper: ml-ss
	if region==1: 	# Region I:  ca-cb < sa-1, ca-cb > 1-sb
		ca,cb,la,lb,sa,sb    = 2,0,1,1,3.5,.5
	elif region==2: # Region II: sa-1 < ca-cb < sa+2, ca-cb > 1-sb
		ca,cb,la,lb,sa,sb    = 2,0,1,1,1,.5
	elif region==3: # Region III:  ca-cb > sa+2, ca-cb > 1-sb
		ca,cb,la,lb,sa,sb    = 3,0,1,1,.5,.5
	elif region==4: # Region IV:  ca-cb < sa-1, ca-cb < 1-sb
		ca,cb,la,lb,sa,sb    = .5,0,1,1,1.8,.1
	elif region==5: # Region V:  sa-1 < ca-cb < sa+2, ca-cb < 1-sb
		ca,cb,la,lb,sa,sb    = 1,.6,1,1,1.1,.5
	elif region==6: # Region VI: ca-cb > sa+2, ca-cb < 1-sb
		print('INFEASIBLE assuming sa>=0 and sb >=0')
		return NotImplementedError()	
	else:
		return NotImplementedError()

	return ca,cb,la,lb,sa,sb

### Linear loyalty model (subsumes multiplicative and additive)

def ll_constraint(p_firm,p_rival,l_firm=1,s_firm = 0,dist='uniform'):
	''' 
	let _firm suffix represent the firm for which state a implies customer is in its strong market
	e.g., in state where cust is in B's strong mkt and ml:  (0 <= pbb-pab) and  (pbb-pab <= lb) 
	'''
	if dist != 'uniform':
		return NotImplementedError

	if (s_firm <= p_firm-p_rival) and  \
		(p_firm-p_rival <= l_firm + s_firm):
		return True
	return False

def ll_prob_cust_a_purchase_from_a(paa,pba,F,la=1,sa=0): return 1-F((paa-pba-sa)/la)

def ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb=1,sb=0): return 1-F((pbb-pab-sb)/lb)

def ll_get_individual_payoff_aa(paa,pba,ca,F,la=1,sa=0): return (paa-ca)*ll_prob_cust_a_purchase_from_a(paa,pba,F,la,sa)

def ll_get_individual_payoff_ab(pbb,pab,ca,F,lb=1,sb=0): return (pab-ca)*(1-ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb,sb))

def ll_get_individual_payoff_ba(paa,pba,cb,F,la=1,sa=0): return (pba-cb)*(1-ll_prob_cust_a_purchase_from_a(paa,pba,F,la,sa))

def ll_get_individual_payoff_bb(pbb,pab,cb,F,lb=1,sb=0): return (pbb-cb)*ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb,sb)

def ll_get_market_shares(paa,pba,pbb,pab,F,theta,la=1,lb=1,sa=0,sb=0): 
	new_market_share_a = theta*ll_prob_cust_a_purchase_from_a(paa,pba,F,la,sa) \
						+ (1-theta)*(1-ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb,sb))
	new_market_share_b = (1-theta)*ll_prob_cust_b_purchase_from_b(pbb,pab,F,lb,sb) \
						+ theta*(1-ll_prob_cust_a_purchase_from_a(paa,pba,F,la,sa))
	return (new_market_share_a,new_market_share_b)

def ll_get_total_profits(paa,pba,pbb,pab,F,theta,ca,cb,la=1,lb=1,sa=0,sb=0):
	total_profit_a = theta*ll_get_individual_payoff_aa(paa,pba,ca,F,la,sa) \
					+ (1-theta)*ll_get_individual_payoff_ab(pbb,pab,ca,F,lb,sb)
	total_profit_b = (1-theta)*ll_get_individual_payoff_bb(pbb,pab,cb,F,lb,sb) \
					+ theta*ll_get_individual_payoff_ba(paa,pba,cb,F,la,sa)
	return (total_profit_a,total_profit_b)

def ll_get_metrics_theory(dist,ca,cb,la=1,lb=1,sa=0,sb=0): #TODO

	if dist != 'uniform':
		return NotImplementedError

	if sa==0 and sb==0: #TODO: remove repetition of formulae, consider xxa and xxb separately, low priority
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

	elif la==1 and lb ==1: #TODO: the conditions seem brittle, for instance in the ML case la=lb=1 is also possible

		if (ca-cb < sa-1): #Region I and IV
			paa_t = cb+sa
			pba_t = cb
		elif (sa-1 <= ca-cb) and (ca-cb <= sa+2): # Region II and V
			paa_t = (2*ca+cb+sa+2)/3
			pba_t = (ca+2*cb-sa+1)/3
		elif (ca-cb > sa+2): # Region III (Region VI is not feasible)
			paa_t = ca
			pba_t = ca-sa-1
		else:
			print('region not defined')
			paa_t,pba_t = [0]*2

		if (ca-cb >= 1 - sb): #Region I, II and III
			pbb_t = ca+sb
			pab_t = ca
		elif (ca-cb < 1-sb): #Region IV and V
			pbb_t = (2*cb+ca+sb+2)/3
			pab_t = (cb+2*ca-sb+1)/3			
		else:
			print('region not defined')
			pbb_t,pab_t = [0]*2

	else:
		return NotImplementedError

	temp = [paa_t,pba_t,pbb_t,pab_t] #TODO: Change to dict
	return (np.round(x,3) for x in temp)

def ll_get_example_in_region(region=1,dist='uniform',deltaf=0): #TODO
	return NotImplementedError

def ll_get_metrics_computed(dist,ca,cb,la=1,lb=1,sa=0,sb=0,maxpx=None,npts=20,show_progress=False,plot_path=False):

	if maxpx is None:
		maxpx = ca+5

	def ll_get_payoff_matrices(dist,maxpx,npts,ca,cb,la=1,lb=1,sa=0,sb=0):
		'''
		There are two states in the game, but there is no transition. Thus, there are two independent games.
		Depending on which market the customer is in (A's strong sub-market vs weak sub-market)
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

				#Computing payoffs in the first game
				if ll_constraint(paa,pba,la,sa,dist) and firm_constraint_cost(paa,ca) and firm_constraint_cost(pba,cb):
					constraint_state_a[i,j] = 1 
					obja_state_a[i,j] = ll_get_individual_payoff_aa(paa,pba,ca,F,la,sa)
					objb_state_a[i,j] = ll_get_individual_payoff_ba(paa,pba,cb,F,la,sa)

				#Computing payoffs for the second game
				pbb,pab = pba,paa #no meaning here, these are temporary variables.
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


	#data for solver: payoff matrices
	pa_arr,pb_arr,payoff_matrices,constraint_matrices = ll_get_payoff_matrices(dist,maxpx,npts,ca,cb,la,lb,sa,sb)
	# pdb.set_trace()

	result_c = compute_single_stage_equilibrium(payoff_matrices,pa_arr,pb_arr,show_progress=False,plot_path=False)

	return result_c['paa'],result_c['pba'],result_c['pbb'],result_c['pab']

def ll_get_metric_arrs_vs_camcb(ca_arr,cb,la=1,lb=1,sa=0,sb=0,maxpx=10,npts=20,dist='uniform',theta=0.5,flag_theory=True):
	'''
	this function iterates over ca. cb is an input.
	'''
	if dist != 'uniform':
		return NotImplementedError()

	print('ll_get_metric_arrs_vs_camcb start: ',datetime.datetime.now())

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
		print('i,ca,time: ',i,np.round(ca,3),datetime.datetime.now())

		if flag_theory is True:
			paa_arr[i],pba_arr[i],pbb_arr[i],pab_arr[i] = ll_get_metrics_theory(dist,ca,cb,la,lb,sa,sb)
		else:
			paa_arr[i],pba_arr[i],pbb_arr[i],pab_arr[i] = ll_get_metrics_computed(dist,ca,cb,la,lb,sa,sb,maxpx,npts,show_progress=False,plot_path=False)

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

	print('ll_get_metric_arrs_vs_camcb end: ',datetime.datetime.now())

	return pd.DataFrame({'paa':paa_arr,'pba':pba_arr,'pbb':pbb_arr,'pab':pab_arr,
				'objaa':objaa_arr,'objba':objba_arr,'objbb':objbb_arr,'objab':objab_arr,
				'marketshare_a':marketshare_a_arr,'marketshare_b':marketshare_b_arr,
				'total_profit_a':total_profit_a_arr,'total_profit_b':total_profit_b_arr,
				'prob_purchase_a_from_a':prob_purchase_a_from_a_arr,
				'prob_purchase_b_from_b':prob_purchase_b_from_b_arr})


def ll_ss_is_equlibrium_exhaustive(paa_arr,pba_arr,paa_c,pba_c,obja,objb,F,ca,cb,la,sa=0,debug=False):
	assert paa_c in paa_arr
	assert pba_c in pba_arr
	
	if firm_constraint_cost(paa_c,ca) and firm_constraint_cost(pba_c,cb) and ll_constraint(paa_c,pba_c,la,sa):
	
		obja_c = ll_get_individual_payoff_aa(paa_c,pba_c,ca,F,la,sa)
		objb_c = ll_get_individual_payoff_ba(paa_c,pba_c,cb,F,la,sa)
		paa_c_idx = np.argmin(np.abs(paa_arr-paa_c)) #assumes paa_c is in the index
		pba_c_idx = np.argmin(np.abs(pba_arr-pba_c)) #assumes pba_c is in the index
	#     print('x_tAy_t >= xAy_t for all x:')
		for i,paa in enumerate(paa_arr):
			if obja[i,pba_c_idx]>obja_c:
				if debug: print('strictly better price by A: i,paa,obja,obja_t:',i,paa_arr[i],obja[i,pba_c_idx],obja_c)
				return False
	#     print('x_tBy_t >= x_tBy for all y:')
		for j,pba in enumerate(pba_arr):
			if objb[paa_c_idx,j]>objb_c:
				if debug: print('strictly better price by B: j,pba,objb,objb_t:',j,pba_arr[j],objb[paa_c_idx,j],objb_c)
				return False
		return True
	else:
		return False


#Compute the A and B payoff matrices for exhaustive experimentation
def ll_get_payoff_matrices_for_exhaustive(dist,npts,c1,c2,p1_t,p2_t,l,s,market='A-strong-sub-market'):

	maxpx = c1+5
	if p1_t > c1:
		p1_arr = np.concatenate((np.linspace(c1,p1_t,npts,endpoint=False),np.linspace(p1_t,maxpx,npts)))
	else:
		p1_arr = np.linspace(c1,maxpx,npts)
	if p2_t > c2:
		p2_arr = np.concatenate((np.linspace(c2,p2_t,npts,endpoint=False),np.linspace(p2_t,maxpx,npts)))
	else:
		p2_arr = np.linspace(c2,maxpx,npts)
	obj1 = np.zeros((p1_arr.size,p2_arr.size))
	obj2 = np.zeros((p1_arr.size,p2_arr.size))
	constraintmat = np.zeros((p1_arr.size,p2_arr.size))

	F,f = get_xi_dist(dist)
	if market=='A-strong-sub-market':
		temp_func1 = ll_get_individual_payoff_aa
		temp_func2 = ll_get_individual_payoff_ba
	elif market=='B-strong-sub-market':
		temp_func1 = ll_get_individual_payoff_bb
		temp_func2 = ll_get_individual_payoff_ab
	else:
		return NotImplementedError

	for i,p1 in enumerate(p1_arr):
		for j,p2 in enumerate(p2_arr):
			if firm_constraint_cost(p1,c1) and firm_constraint_cost(p2,c2) and ll_constraint(p1,p2,l,s):
				constraintmat[i,j] = 1 
				obj1[i,j] = temp_func1(p1,p2,c1,F,l,s)
				obj2[i,j] = temp_func2(p1,p2,c2,F,l,s)
	return p1_arr,p2_arr,obj1,obj2,constraintmat

def ll_get_objective_vals_at(dist,p1_t,p2_t,c1,c2,l,s,market='A-strong-sub-market'):
	F,f = get_xi_dist(dist)
	if market=='A-strong-sub-market':
		temp_func1 = ll_get_individual_payoff_aa
		temp_func2 = ll_get_individual_payoff_ba
	else:
		temp_func1 = ll_get_individual_payoff_bb
		temp_func2 = ll_get_individual_payoff_ab
		
	obj1_t = temp_func1(p1_t,p2_t,c1,F,l,s)
	obj2_t = temp_func2(p1_t,p2_t,c2,F,l,s)
	
	flag_constraint = firm_constraint_cost(p1_t,c1) and firm_constraint_cost(p2_t,c2) \
			and ll_constraint(p1_t,p2_t,l,s)
	
	return obj1_t,obj2_t,flag_constraint


def get_plot_constraint_using_plotly(p1_arr,p2_arr,constraintmat,xname='pba',yname='paa'):#TODO: separate model, compute and visualization functions
    #Constraint space
    fig = go.Figure(data =go.Contour( z=constraintmat, x=p1_arr, y=p2_arr))
    fig.update_layout( xaxis_title=xname, yaxis_title=yname,font=dict(size=18))
    fig.show()

def get_plot_payoffs_using_plotly(obj1,obj2,p1_arr,p2_arr):
	#A and B's payoffs
	fig = make_subplots(rows=1, cols=2)
	fig.add_trace(
		go.Contour( z=obj1, x=p2_arr, y=p1_arr),
		row=1, col=1)
	fig.add_trace(
		go.Contour( z=obj2, x=p2_arr, y=p1_arr),
		row=1, col=2)
	fig.update_layout(height=600, width=800, title_text="A and B's payoffs \(top view\)")
	fig.show()

def get_3d_plot_payoffs_using_plotly():
	return NotImplementedError
	# 3D plot of obja
	# fig = go.Figure(data=[go.Surface(z=obja, x=pba_arr, y=paa_arr)])
	# fig.update_layout(autosize=False, width=500, height=500, scene = dict(xaxis = dict(title='pba'),yaxis = dict(title='paa')))
	# fig.show()
	# 3D plot of objb
	# fig = go.Figure(data=[go.Surface(z=objb, x=pba_arr, y=paa_arr)])
	# fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90), scene = dict(xaxis = dict(title='pba'),yaxis = dict(title='paa')))
	# fig.show()
from utils.imports import *
from utils.gameSolver import dsSolve
from utils.single_stage import get_xi_dist,\
					ll_constraint,\
					firm_constraint_cost,\
					firm_constraint_across                    

'''
In the below variables, first index is always firm, and second index is the strong sub-market
For instance, in paa_arr, first 'a' represents firm A's pricing, second 'a' represents A's strong sub-market 'alpha'

There are two states: customer is either in set alpha, or in set beta.
If the customer is in state alpha, the two firms show one price each. 
Similarly they show one price each when the customer is in state beta.

'''

### General loyalty model

def compute_infinite_horizon_equilibrium(payoff_matrices,pa_arr,pb_arr,transition_prob_matrices,discount_factors,show_progress=True,plot_path=True):
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


### Multiplicative loyalty model

def ml_prob_cust_a_purchase_from_a(paa,pba,la,F): return 1-F((paa-pba)/la) #TODO: replace, import from ss ll
def ml_prob_cust_b_purchase_from_b(pbb,pab,lb,F): return 1-F((pbb-pab)/lb)

def ml_constraints_state_a(paa,pba,ca,cb,la):
	if paa >= ca and pba >= cb and (0 <= paa-pba) and  (paa-pba <= la):
		return True
	return False

def ml_constraints_state_b(pbb,pab,ca,cb,lb):
	if pbb >= cb and pab >= ca and (0 <= pbb-pab) and  (pbb-pab <= lb):
		return True
	return False

def ml_get_metrics_theory(ca,cb,la,lb,F,f,deltaf,dist='uniform'):


	def xi_equations_ml(candidates,ca,cb,la,lb,F,f,deltaf):
		gammaa = (ca-cb)/la
		gammab = (cb-ca)/lb
		delbydel = (1-deltaf)/deltaf
		
		xia, xib = candidates
		if xia<0 or xia>1 or xib <0 or xib > 1: #hardcoded for cdf between 0 and 1: TODO
			return (math.inf,math.inf)

		residual1 = (xia-gammaa)*(delbydel+F(xib)+1) \
					+ ((2*F(xia)-1)/f(xia))*(delbydel + F(xib) +F(xia)) \
					+ (F(xia)/f(xia)) \
					- (((1-F(xib))*lb)/(la*f(xib)) -F(xib)*(gammaa + xib*lb/la ))
		residual2 = (xib-gammab)*(delbydel+F(xia)+1) \
					+ ((2*F(xib)-1)/f(xib))*(delbydel + F(xib) +F(xia)) \
					+ (F(xib)/f(xib)) \
					- (((1-F(xia))*la)/(lb*f(xia)) -F(xia)*(gammab + xia*la/lb ))
		
		return (residual1,residual2)


	def pax_equations_ml(candidates,xia,xib,ca,cb,la,lb,F,f,deltaf):

		def vaopt_diff_ml(paa,pab,xia,xib,ca,F,deltaf):
			return ((1-F(xia))*(paa-ca) - F(xib)*(pab-ca))/(1-deltaf+deltaf*(F(xia)+F(xib)))

		paa,pab = candidates
		if xia<0 or xia>1 or xib <0 or xib > 1:
			return (math.inf,math.inf)
		vaopt = vaopt_diff_ml(paa,pab,xia,xib,ca,F,deltaf)
		residual1 = paa -(ca + (1-F(xia))*la/f(xia) -deltaf*vaopt)
		residual2 = pab -(ca + F(xib)*lb/f(xib) -deltaf*vaopt)
		return (residual1,residual2)

	def pbx_equations_ml(candidates,xia,xib,ca,cb,la,lb,F,f,deltaf):

		def vbopt_diff_ml(pbb,pba,xia,xib,cb,F,deltaf):
			return ((1-F(xib))*(pbb-cb) - F(xia)*(pba-cb))/(1-deltaf+deltaf*(F(xia)+F(xib)))

		pbb,pba = candidates
		if xia<0 or xia>1 or xib <0 or xib > 1:
			return (math.inf,math.inf)
		vbopt = vbopt_diff_ml(pbb,pba,xia,xib,cb,F,deltaf)
		residual1 = pbb - (cb + (1-F(xib))*lb/f(xib) -deltaf*vbopt)
		residual2 = pba - (cb + F(xia)*la/f(xia) -deltaf*vbopt)
		return (residual1,residual2)

	def get_vopts_ml(paa,pab,pbb,pba,xia,xib,da,db,ca,cb,F):

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


	xia,xib = fsolve(xi_equations_ml, (0.5, 0.5),(ca,cb,la,lb,F,f,deltaf)) #hardcoded initial point for fsolve
	paa,pab = fsolve(pax_equations_ml, (0.5, 0.5),(xia,xib,ca,cb,la,lb,F,f,deltaf)) #hardcoded initial point for fsolve
	pbb,pba = fsolve(pbx_equations_ml, (0.5, 0.5),(xia,xib,ca,cb,la,lb,F,f,deltaf)) #hardcoded initial point for fsolve
	vaao,vabo,vbbo,vbao = get_vopts_ml(paa,pab,pbb,pba,xia,xib,deltaf,deltaf,ca,cb,F)

	return paa,pab,pbb,pba,xia,xib,vaao,vabo,vbbo,vbao

def ml_get_metrics_computed(ca,cb,la,lb,F,f,deltaf,dist='uniform',maxpx=10,npts=20,show_progress=False,plot_path=False):

	def ml_get_transition_prob_matrices(ca,cb,maxpx,npts,F,f,la,lb):

		pa_arr = np.linspace(ca,maxpx,npts)
		pb_arr = np.linspace(cb,maxpx,npts)

		transition_prob_matrices = []
		#state:customer is in set alpha
		transition_prob_matrix = np.zeros((len(pa_arr),len(pb_arr),2))
		for i,pa in enumerate(pa_arr):
			for j,pb in enumerate(pb_arr):
				transition_prob_matrix[i,j,0] = ml_prob_cust_a_purchase_from_a(pa,pb,la,F)  #from alpha to set alpha
				transition_prob_matrix[i,j,1] = 1-transition_prob_matrix[i,j,0] #transitioning to set beta

		transition_prob_matrices.append(transition_prob_matrix)
		#state:customer is in set beta
		transition_prob_matrix = np.zeros((len(pa_arr),len(pb_arr),2))
		for i,pa in enumerate(pa_arr):
			for j,pb in enumerate(pb_arr):
				transition_prob_matrix[i,j,1] = ml_prob_cust_b_purchase_from_b(pb,pa,lb,F)#transitioning to set beta
				transition_prob_matrix[i,j,0] = 1-transition_prob_matrix[i,j,1]  #from beta to set alpha

		transition_prob_matrices.append(transition_prob_matrix)

		return transition_prob_matrices


	#data for solver: payoff matrices	
	pa_arr,pb_arr,obja_state_a,objb_state_a,\
	constraint_state_a = ml_get_payoff_matrices_state_a(ca,cb,maxpx,npts,dist,la)
	_,_,obja_state_b,objb_state_b,\
	constraint_state_b = ml_get_payoff_matrices_state_b(ca,cb,maxpx,npts,dist,lb)
	payoff_matrices = [ ## s = \alpha
						np.array([obja_state_a,objb_state_a]),
						## s = \beta
						np.array([obja_state_b,objb_state_b])]

	transition_prob_matrices = ml_get_transition_prob_matrices(ca,cb,maxpx,npts,F,f,la,lb)
	# set_trace()
	result1_c,result2_c = compute_infinite_horizon_equilibrium(payoff_matrices,pa_arr,pb_arr,transition_prob_matrices,deltaf,
							 show_progress,plot_path)

	xia = (result1_c['paa'] - result1_c['pba'])/la
	xib = (result2_c['pbb'] - result2_c['pab'])/lb

	return result1_c['paa'],result2_c['pab'],result2_c['pbb'],result1_c['pba'],xia,xib,result1_c['vaa'],result2_c['vab'],result2_c['vbb'],result1_c['vba']

def ml_get_market_shares(paa,pba,pbb,pab,la,lb,F): 

	praa = ml_prob_cust_a_purchase_from_a(paa,pba,la,F)
	prbb = ml_prob_cust_b_purchase_from_b(pbb,pab,lb,F)

	A = np.array([[praa-1,1-prbb],[1,1]])
	b = np.array([0,1])
	thetavec = np.linalg.solve(A, b)
	new_market_share_a = thetavec[0]
	new_market_share_b = thetavec[1]
	return (new_market_share_a,new_market_share_b)

def ml_get_total_profits(vaa,vab,vbb,vba,theta):
	total_profit_a = theta*vaa + (1-theta)*vab
	total_profit_b = (1-theta)*vbb + theta*vba
	return (total_profit_a,total_profit_b)

def ll_get_metric_arrs_vs_camcb(ca_arr,cb,la,lb,sa,sb,dist,deltaf,flag_theory=True,maxpx=10,npts=20,show_progress=False,plot_path=False):

	paa_arr,pba_arr,pbb_arr,pab_arr,xia_arr,xib_arr,vaao_arr,vabo_arr,vbbo_arr,vbao_arr,\
	constraint_aa_ba_arr,constraint_bb_ab_arr,constraint_cross_a_arr,constraint_cross_b_arr,\
	marketshare_a_arr,marketshare_b_arr,total_profit_a_arr,total_profit_b_arr,\
	prob_purchase_a_from_a_arr,prob_purchase_b_from_b_arr \
		= ml_get_metric_arrs_vs_camcb_nodf(ca_arr,cb,la,lb,dist,deltaf,flag_theory,maxpx,npts,show_progress,plot_path)

	return pd.DataFrame({'paa':paa_arr,'pba':pba_arr,'pbb':pbb_arr,'pab':pab_arr,
		'vaa':vaao_arr,'vba':vbao_arr,'vbb':vbbo_arr,'vab':vabo_arr,
		'marketshare_a':marketshare_a_arr,'marketshare_b':marketshare_b_arr,
		'total_profit_a':total_profit_a_arr,'total_profit_b':total_profit_b_arr,
		'prob_purchase_a_from_a':prob_purchase_a_from_a_arr,
		'prob_purchase_b_from_b':prob_purchase_b_from_b_arr,
		'constraint_aa_ba_arr':constraint_aa_ba_arr,'constraint_bb_ab_arr':constraint_bb_ab_arr,
		'constraint_cross_a_arr':constraint_cross_a_arr,'constraint_cross_b_arr':constraint_cross_b_arr,
		'xia':xia_arr,'xib':xib_arr
		})	

def ml_get_metric_arrs_vs_camcb_nodf(ca_arr,cb,la,lb,dist,deltaf,flag_theory=True,maxpx=10,npts=20,show_progress=False,plot_path=False):

	#make this function output a list of arrays needed for ml_compare_two_solutions
	# and made a simple wrapper that outputs a df, similar to the single stage setting


	if dist != 'uniform':
		return NotImplementedError()

	print('ml_get_metric_arrs_vs_camcb_nodf start: ',datetime.datetime.now())

	F,f    = get_xi_dist(dist)

	paa_arr = np.zeros(ca_arr.size) #first index is firm, second index is customer type
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


	for i,ca in enumerate(ca_arr):
		print('i,ca,time: ',i,np.round(ca,3),datetime.datetime.now())

		if flag_theory:
			paa_arr[i],pab_arr[i],pbb_arr[i],pba_arr[i], xia_arr[i],xib_arr[i],vaao_arr[i],vabo_arr[i],vbbo_arr[i],vbao_arr[i]  \
				= ml_get_metrics_theory(ca,cb,la,lb,F,f,deltaf,dist)
		else:
			paa_arr[i],pab_arr[i],pbb_arr[i],pba_arr[i], xia_arr[i],xib_arr[i],vaao_arr[i],vabo_arr[i],vbbo_arr[i],vbao_arr[i]  \
				= ml_get_metrics_computed(ca,cb,la,lb,F,f,deltaf,dist,maxpx,npts,show_progress,plot_path)

		if ll_constraint(paa_arr[i],pba_arr[i],la,0,dist) and firm_constraint_cost(paa_arr[i],ca) and firm_constraint_cost(pba_arr[i],cb):
			constraint_aa_ba_arr[i] = 1
		if ll_constraint(pbb_arr[i],pab_arr[i],lb,0,dist) and firm_constraint_cost(pbb_arr[i],cb) and firm_constraint_cost(pab_arr[i],ca):
			constraint_bb_ab_arr[i] = 1
		if firm_constraint_across(paa_arr[i],pab_arr[i]): #this constraint is not being imposed while defining feasible actions
			constraint_cross_a_arr[i] = 1 
		if firm_constraint_across(pbb_arr[i],pba_arr[i]):
			constraint_cross_b_arr[i] = 1


		marketshare_a_arr[i],marketshare_b_arr[i] = ml_get_market_shares(paa_arr[i],pba_arr[i],pbb_arr[i],pab_arr[i],la,lb,F)
		total_profit_a_arr[i],total_profit_b_arr[i] = ml_get_total_profits(vaao_arr[i],vabo_arr[i],vbbo_arr[i],vbao_arr[i],marketshare_a_arr[i])
		prob_purchase_a_from_a_arr[i] = ml_prob_cust_a_purchase_from_a(paa_arr[i],pba_arr[i],la,F)
		prob_purchase_b_from_b_arr[i] = ml_prob_cust_b_purchase_from_b(pbb_arr[i],pab_arr[i],lb,F)


	print('ml_get_metric_arrs_vs_camcb_nodf end: ',datetime.datetime.now())

	return paa_arr,pba_arr,pbb_arr,pab_arr,xia_arr,xib_arr,vaao_arr,vabo_arr,vbbo_arr,vbao_arr,\
		constraint_aa_ba_arr,constraint_bb_ab_arr,constraint_cross_a_arr,constraint_cross_b_arr,\
		marketshare_a_arr,marketshare_b_arr,total_profit_a_arr,total_profit_b_arr,\
		prob_purchase_a_from_a_arr,prob_purchase_b_from_b_arr

def ll_compare_two_solutions(ca,cb,la,lb,maxpx,npts,deltaf,dist):

	#ml ih theory
	paa_arr,pba_arr,pbb_arr,pab_arr,xia_arr,xib_arr,vaao_arr,vabo_arr,vbbo_arr,vbao_arr,\
	constraint_aa_ba_arr,constraint_bb_ab_arr,constraint_cross_a_arr,constraint_cross_b_arr, \
	marketshare_a_arr,marketshare_b_arr,total_profit_a_arr,total_profit_b_arr,\
	prob_purchase_a_from_a_arr,prob_purchase_b_from_b_arr \
		= ml_get_metric_arrs_vs_camcb_nodf(np.array([ca]),cb,la,lb,dist,deltaf,flag_theory=True)
	result1_t = {'paa':paa_arr[0],'pba':pba_arr[0],'vaa':vaao_arr[0],'vba':vbao_arr[0], 'xia':xia_arr[0],
		'constraint_aa_ba':constraint_aa_ba_arr[0],'constraint_cross_a':constraint_cross_a_arr[0] }
	result2_t = {'pab':pab_arr[0],'pbb':pbb_arr[0],'vab':vabo_arr[0],'vbb':vbbo_arr[0], 'xib':xib_arr[0], 
		'constraint_bb_ab':constraint_bb_ab_arr[0],'constraint_cross_b': constraint_cross_b_arr[0]}
	result1_t = {x:np.round(y,3) for x,y in result1_t.items()}
	result2_t = {x:np.round(y,3) for x,y in result2_t.items()}

	#ml ih computational
	paa_arr,pba_arr,pbb_arr,pab_arr,xia_arr,xib_arr,vaao_arr,vabo_arr,vbbo_arr,vbao_arr,\
	constraint_aa_ba_arr,constraint_bb_ab_arr,constraint_cross_a_arr,constraint_cross_b_arr, \
	marketshare_a_arr,marketshare_b_arr,total_profit_a_arr,total_profit_b_arr,\
	prob_purchase_a_from_a_arr,prob_purchase_b_from_b_arr \
	 	= ml_get_metric_arrs_vs_camcb_nodf(np.array([ca]),cb,la,lb,dist,deltaf,False,maxpx,npts,False,False)
	result1_c = {'paa':paa_arr[0],'pba':pba_arr[0],'vaa':vaao_arr[0],'vba':vbao_arr[0], 'xia':xia_arr[0],
		'constraint_aa_ba':constraint_aa_ba_arr[0],'constraint_cross_a':constraint_cross_a_arr[0] }
	result2_c = {'pab':pab_arr[0],'pbb':pbb_arr[0],'vab':vabo_arr[0],'vbb':vbbo_arr[0], 'xib':xib_arr[0], 
		'constraint_bb_ab':constraint_bb_ab_arr[0],'constraint_cross_b': constraint_cross_b_arr[0]}
	result1_c = {x:np.round(y,3) for x,y in result1_c.items()}
	result2_c = {x:np.round(y,3) for x,y in result2_c.items()}
	
	return pd.DataFrame({'type':['theory','computed'],
		'paa':np.array([result1_t['paa'],result1_c['paa']]),
		'pba':np.array([result1_t['pba'],result1_c['pba']]),
		'pbb':np.array([result2_t['pbb'],result2_c['pbb']]),
		'pab':np.array([result2_t['pab'],result2_c['pab']]),
		'xia':np.array([result1_t['xia'],result1_c['xia']]),
		'xib':np.array([result2_t['xib'],result2_c['xib']]),
		'vaa':np.array([result1_t['vaa'],result1_c['vaa']]),
		'vab':np.array([result2_t['vab'],result2_c['vab']]),
		'vbb':np.array([result2_t['vbb'],result2_c['vbb']]),
		'vba':np.array([result1_t['vba'],result1_c['vba']]),
		'constraint_aa_ba':np.array([result1_t['constraint_aa_ba'],result1_c['constraint_aa_ba']]),
		'constraint_bb_ab':np.array([result2_t['constraint_bb_ab'],result2_c['constraint_bb_ab']]),
		'constraint_cross_a':np.array([result1_t['constraint_cross_a'],result1_c['constraint_cross_a']]),
		'constraint_cross_b':np.array([result2_t['constraint_cross_b'],result2_c['constraint_cross_b']])
		})


### Linear loyalty model (subsumes multiplicative and additive)

def ll_get_metrics_computed(ca,cb,la,lb,sa,sb,F,f,deltaf,maxpx=None,npts=20,show_progress=False,plot_path=False):

	return NotImplementedError()

	if maxpx is None:
		maxpx = ca+5

	def ml_get_payoff_matrices_state_a(ca,cb,maxpx,npts,F,f,la):

		pa_state_a_arr = np.linspace(ca,maxpx,npts)
		pb_state_a_arr = np.linspace(cb,maxpx,npts)
		'''
		 pa_state_a_arr : array of A's prices for its strong sub-market
		 pb_state_a_arr: array of B's prices for its weak sub-market
		'''

		obja_state_a = np.zeros((pa_state_a_arr.size,pb_state_a_arr.size)) #A's obj for its strong sub-market
		objb_state_a = np.zeros((pa_state_a_arr.size,pb_state_a_arr.size)) #B's obj for its weak sub-market
		constraint_state_a = np.zeros((pa_state_a_arr.size,pb_state_a_arr.size)) #px constraints related to A's strong sub-market


		for i,paa in enumerate(pa_state_a_arr): #firm A is the row player
			for j,pba in enumerate(pb_state_a_arr):
				if ml_constraints_state_a(paa,pba,ca,cb,la):
					constraint_state_a[i,j] = 1 
					obja_state_a[i,j] = get_payoff_aa(paa,ca,F,pba,la)
					objb_state_a[i,j] = get_payoff_ba(pba,cb,F,paa,la)

		return pa_state_a_arr,pb_state_a_arr,obja_state_a,objb_state_a,constraint_state_a

	def ml_get_payoff_matrices_state_b(ca,cb,maxpx,npts,F,f,lb):

		pa_state_b_arr = np.linspace(ca,maxpx,npts)
		pb_state_b_arr = np.linspace(cb,maxpx,npts)
		'''
		 pa_state_b_arr : array of A's prices for its weak sub-market
		 pb_state_b_arr: array of B's prices for its strong sub-market
		'''

		obja_state_b = np.zeros((pa_state_b_arr.size,pb_state_b_arr.size)) #A's obj for its weak sub-market
		objb_state_b = np.zeros((pa_state_b_arr.size,pb_state_b_arr.size)) #B's obj for its strong sub-market
		constraint_state_b = np.zeros((pa_state_b_arr.size,pb_state_b_arr.size)) #px constraints related to B's strong sub-market


		for i,pab in enumerate(pa_state_b_arr): #firm A is the row player
			for j,pbb in enumerate(pb_state_b_arr):
				if ml_constraints_state_b(pbb,pab,ca,cb,lb):
					constraint_state_b[i,j] = 1 
					obja_state_b[i,j] = get_payoff_ab(pab,ca,F,pbb,lb)
					objb_state_b[i,j] = get_payoff_bb(pbb,cb,F,pab,lb)

		return pa_state_b_arr,pb_state_b_arr,obja_state_b,objb_state_b,constraint_state_b

	def ml_get_transition_prob_matrices(ca,cb,maxpx,npts,F,f,la,lb):

		pa_arr = np.linspace(ca,maxpx,npts)
		pb_arr = np.linspace(cb,maxpx,npts)

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


	#data for solver: payoff matrices
	pa_arr,pb_arr,obja_state_a,objb_state_a,\
	constraint_state_a = ml_get_payoff_matrices_state_a(ca,cb,maxpx,npts,F,f,la)
	_,_,obja_state_b,objb_state_b,\
	constraint_state_b = ml_get_payoff_matrices_state_b(ca,cb,maxpx,npts,F,f,lb)
	payoff_matrices = [ ## s = \alpha
						np.array([obja_state_a,objb_state_a]),
						## s = \beta
						np.array([obja_state_b,objb_state_b])]

	transition_prob_matrices = ml_get_transition_prob_matrices(ca,cb,maxpx,npts,F,f,la,lb)
	result1_c,result2_c = compute_equilibrium(payoff_matrices,transition_prob_matrices,deltaf,
							 pa_arr,pb_arr,show_progress,plot_path)

	xia = (result1_c['paa'] - result1_c['pba'])/la
	xib = (result2_c['pbb'] - result2_c['pab'])/lb

	return result1_c['paa'],result2_c['pab'],result2_c['pbb'],result1_c['pba'],xia,xib,result1_c['vaa'],result2_c['vab'],result2_c['vbb'],result1_c['vba']




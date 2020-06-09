from utils.imports import *

'''
In the below variables, first index is always firm, and second index is the strong sub-market
For instance, in paa_arr, first 'a' represents firm A's pricing, second 'a' represents A's strong sub-market 'alpha'
'''

def init_single_stage_a(ca,cb,maxpx,npts):

	paa_arr = np.linspace(ca,maxpx,npts) #A's price for its strong sub-market
	pba_arr = np.linspace(cb,maxpx,npts) #B's price for its weak sub-market
	objaa = np.zeros((paa_arr.size,pba_arr.size)) #A's obj for its strong sub-market
	objba = np.zeros((paa_arr.size,pba_arr.size)) #B's obj for its weak sub-market
	constraintmatA = np.zeros((paa_arr.size,pba_arr.size)) #px constraints related to A's sub-market
	return paa_arr,pba_arr,objaa,objba,constraintmatA

def init_single_stage_b(ca,cb,maxpx,npts):
	pbb_arr = np.linspace(cb,maxpx,npts)
	pab_arr = np.linspace(ca,maxpx,npts)
	objbb = np.zeros((pbb_arr.size,pab_arr.size))
	objab = np.zeros((pbb_arr.size,pab_arr.size))
	constraintmatB = np.zeros((pbb_arr.size,pab_arr.size))
	return pbb_arr,pab_arr,objbb,objab,constraintmatB

def get_xi_dist(dist='normal'):

    if dist=='uniform':
        return (np.arange(0.0, 1.0, 0.1),uniform.cdf,uniform.pdf)
    return (np.arange(-1.0, 1.0, 0.1),norm.cdf,norm.pdf)


def constraints_a(paa,pba,ca,cb,la): return (paa >= ca) and (pba >= cb) and (0 <= paa-pba) and (paa-pba <= la)
def constraints_b(pbb,pab,cb,ca,lb): return (pbb >= cb) and (pab >= ca) and (0 <= pbb-pab) and (pbb-pab <= lb)

def get_payoff_aa(paa,ca,F,pba,la): return (paa-ca)*(1-F((paa-pba)/la))
def get_payoff_ba(pba,cb,F,paa,la): return (pba-cb)*F((paa-pba)/la)
def get_payoff_bb(pbb,cb,F,pab,lb): return (pbb-cb)*(1-F((pbb-pab)/lb))
def get_payoff_ab(pab,ca,F,pbb,lb): return (pab-ca)*F((pbb-pab)/lb)

def get_payoffs_a(ca,cb,maxpx,npts,dist,la):

	_,F,f = get_xi_dist(dist)

	paa_arr,pba_arr,objaa,objba,cmata = init_single_stage_a(ca,cb,maxpx,npts)


	for i,paa in enumerate(paa_arr):
	    for j,pba in enumerate(pba_arr):
	        if constraints_a(paa,pba,ca,cb,la):
	            cmata[i,j] = 1 
	            objaa[i,j] = get_payoff_aa(paa,ca,F,pba,la)
	            objba[i,j] = get_payoff_ba(pba,cb,F,paa,la)

	return paa_arr,pba_arr,objaa,objba,cmata

def get_payoffs_b(ca,cb,maxpx,npts,dist,lb):

	_,F,f = get_xi_dist(dist)

	pbb_arr,pab_arr,objbb,objab,cmatb = init_single_stage_b(ca,cb,maxpx,npts)


	for i,pbb in enumerate(pbb_arr):
	    for j,pab in enumerate(pab_arr):
	        if constraints_b(pbb,pab,cb,ca,lb):
	            cmatb[i,j] = 1 
	            objbb[i,j] = get_payoff_bb(pbb,cb,F,pab,lb)
	            objab[i,j] = get_payoff_ab(pab,ca,F,pbb,lb)

	return pbb_arr,pab_arr,objbb,objab,cmatb


def get_objs(paa,pba,pbb,pab,ca,cb,la,lb,dist):

	_,F,f = get_xi_dist(dist)
	objaa = get_payoff_aa(paa,ca,F,pba,la)
	objba = get_payoff_ba(pba,cb,F,paa,la)
	objbb = get_payoff_bb(pbb,cb,F,pab,lb)
	objab = get_payoff_ab(pab,ca,F,pbb,lb)

	temp  = [objaa, objba, objbb, objab]
	return (np.round(x,3) for x in temp)


def get_example(region=1):
	#From the four propositions in the paper: ml-ss

	if region==1:
		# Region I:  lb < ca-cb < 2la
		ca,cb,la,lb    = 2,0,2,1

		paa_t = 0.33*(2*ca+cb+2*la) #suffix 'T' means theoretical/analytical
		pba_t = 0.33*(2*cb+ca+la)
		pbb_t = ca
		pab_t = ca
	elif region==2:
		# Region II: ca-cb > min(2la,lb)
		ca,cb,la,lb    = 5,0,2,1
		paa_t = ca
		pba_t = ca-la
		pbb_t = ca
		pab_t = ca
	elif region==3:
		# Region III:  ca-cb < min(2la,lb)
		ca,cb,la,lb    = 1,0,2,2
		paa_t = 0.33*(2*ca+cb+2*la)
		pba_t = 0.33*(2*cb+ca+la)
		pbb_t = 0.33*(2*cb+ca+2*lb)
		pab_t = 0.33*(2*ca+cb+lb)
	elif region==4:
		# Region IV:  2la < ca-cb < lb
		ca,cb,la,lb    = 3,0,1,4
		paa_t = ca
		pba_t = ca-la
		pbb_t = 0.33*(2*cb+ca+2*lb)
		pab_t = 0.33*(2*ca+cb+lb)
	else:
		print('region not defined')
		ca,cb,la,lb,paa_t,pba_t,pbb_t,pab_t = [0]*8

	temp = [ca,cb,la,lb,paa_t,pba_t,pbb_t,pab_t]
	return (np.round(x,3) for x in temp)


def get_theoretical_prices(ca,cb,la,lb):
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

def get_opt_prices_a(cb,la):
	ca_arr = np.linspace(cb,10,100) #ensure ca >= cb
	ca_m_cb_arr = ca_arr - cb
	paa_t_arr = np.zeros(ca_arr.size)
	pba_t_arr = np.zeros(ca_arr.size)
	for i,ca in enumerate(ca_arr):
		paa_t_arr[i],pba_t_arr[i],_,_ = get_theoretical_prices(ca,cb,la,la)
	return paa_t_arr, pba_t_arr,ca_m_cb_arr


def get_opt_prices_b(cb,lb):
	ca_arr = np.linspace(cb,10,100) #ensure ca >= cb
	ca_m_cb_arr = ca_arr - cb
	pbb_t_arr = np.zeros(ca_arr.size)
	pab_t_arr = np.zeros(ca_arr.size)
	for i,ca in enumerate(ca_arr):
		_,_,pbb_t_arr[i],pab_t_arr[i] = get_theoretical_prices(ca,cb,lb,lb)
	return pbb_t_arr, pab_t_arr,ca_m_cb_arr

from utils.imports import *

#simple one liner functions
def perc_75(x): return np.percentile(x,75) #Percentile calculations
def perc_25(x): return np.percentile(x,25)


def plot_ml_ss(p1_t_arr,p2_t_arr,ca_m_cb_arr,vert_line_loc=None,labels=None,fname=None):
	if labels is None or len(labels) < 2:
		print('plot_ml_ss: cannot plot without labels')
		return

	fig, ax = plt.subplots(figsize=(8,6), dpi=300)
	pltsettings ={'linestyle':'-','alpha':0.5}
	ax.plot(ca_m_cb_arr, p1_t_arr,**pltsettings,label=labels[0])
	ax.plot(ca_m_cb_arr, p2_t_arr,**pltsettings,label=labels[1])
	if vert_line_loc is not None and len(labels) >= 3:
		ax.axvline(x=vert_line_loc,color='k', linestyle='--',label=labels[2])
	ax.set(xlabel=r'$c_a - c_b$', ylabel='Equilibrum Prices')
	ax.legend(loc="best")
	ax.grid()
	plt.tight_layout()
	if fname is not None:
		fig.savefig(fname)
	plt.show()


def plot_ml_ss_metrics(metric_a_arr,metric_b_arr,ca_m_cb_arr,vert_line_locs=None,metric='Market Share',fname=None):
	fig, ax = plt.subplots(figsize=(8,6), dpi=300)
	pltsettings ={'linestyle':'-','alpha':0.5}
	ax.plot(ca_m_cb_arr, metric_a_arr,**pltsettings,label='Firm A')
	ax.plot(ca_m_cb_arr, metric_b_arr,**pltsettings,label='Firm B')
	if vert_line_locs is not None:
		for vert_line_loc in vert_line_locs:
			ax.axvline(x=vert_line_loc,color='k', linestyle='--')
	ax.set(xlabel=r'$c_a - c_b$', ylabel=metric)
	ax.legend(loc="best")
	ax.grid()
	plt.tight_layout()
	if fname is not None:
		fig.savefig(fname)
	plt.show()


if __name__=='__main__':
	data = np.array(list(range(1,101)))
	print(perc_25(data),perc_75(data))
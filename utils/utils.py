from utils.imports import *

#simple one liner functions
def perc_75(x): return np.percentile(x,75) #Percentile calculations
def perc_25(x): return np.percentile(x,25)


def plot_curves_vs_camcb(metric_1_arr,metric_2_arr,ca_m_cb_arr,ylabel='',labels=None,
	vert_line_locs=None,vert_line_locs_labels=None,fname=None):

	fig, ax = plt.subplots(figsize=(8,6), dpi=300)
	pltsettings ={'linestyle':'-','alpha':0.5}

	if labels is None:
		ax.plot(ca_m_cb_arr, metric_1_arr,**pltsettings)
		ax.plot(ca_m_cb_arr, metric_2_arr,**pltsettings)
	elif len(labels)>= 2:
		ax.plot(ca_m_cb_arr, metric_1_arr,**pltsettings,label=labels[0])
		ax.plot(ca_m_cb_arr, metric_2_arr,**pltsettings,label=labels[1])
	if vert_line_locs is not None:
		if vert_line_locs_labels is not None and len(vert_line_locs_labels)==len(vert_line_locs):
			for i,vert_line_loc in enumerate(vert_line_locs):
				ax.axvline(x=vert_line_loc,color='k', linestyle='--',label=vert_line_locs_labels[i])
		else:
			for vert_line_loc in vert_line_locs:
				ax.axvline(x=vert_line_loc,color='k', linestyle='--')

	ax.set(xlabel=r'$c_a - c_b$', ylabel=ylabel)
	ax.legend(loc="best")
	ax.grid()
	plt.tight_layout()
	if fname is not None:
		fig.savefig(fname)
	plt.show()
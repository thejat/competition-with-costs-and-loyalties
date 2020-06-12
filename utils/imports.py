#Core scientific modules
import math
import numpy as np
from scipy.stats import uniform, norm
from scipy.optimize import fsolve, minimize


#Pandas, fileIO
import os, glob
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = 100

#Plotting, display
from IPython.display import display, Markdown
from matplotlib import pyplot as plt, rcParams
from cycler import cycler
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
sns.set(style = "white", context="paper")
notebook_default_rcparams = { #Set default RC parameters
    "axes.labelsize": 30,
    "legend.fontsize":24,
    "xtick.labelsize":28,
    "ytick.labelsize":28,
    "axes.grid":True,
    "legend.framealpha":0.5,
    "lines.linewidth":5,
    "legend.loc":'upper left'}
rcParams.update(notebook_default_rcparams)

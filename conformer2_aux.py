# Collection of functions need to run conformer2.py
import math
import random
import matplotlib
import numpy as np
import pickle as PK
from inspect import isfunction # Need this to test whether something is a function!
import matplotlib.pyplot as plt
from tensorflow import keras as KER

def pickle_dump(data,fname,ftype,default=None):
    fn = fname if fname else default
    fid = fname + ".txt"
    f = open(fid, 'wb')  # w => for writing, b => binary
    PK.dump(data, f)

def pickle_load(fname,ftype,default=None):
    fn = fname if fname else default
    fid = fname + ".txt"
    f = open(fid,'rb')
    return PK.load(f)

def n_of(count, item):
    if isfunction(item):
        return [item() for _ in range(count)]
    else:
        return [item]*count

def biased_coin_toss(prob=.5):
    return random.uniform(0, 1) <= prob

def randab(a, b):
    return a + (b - a) * random.uniform(0, 1)

def randab_int(a,b,size=1):
    if size == 1:
        return random.randint(a,b)
    else:
        return np.random.randint(a,b,size=size)

def randelem(elems,size=1):
    def gen_one():
        return elems[random.randint(0, len(elems) - 1)] if elems else None
    if size == 1:   return gen_one()
    else:   return  np.array([gen_one() for i in range(size)])

def general_avg(elems, prop_func=(lambda x: x)):
    if elems:
        return sum(list(map(prop_func, elems))) / len(elems)

def general_variance(elems, prop_func=(lambda x: x), avg=None):
    if not (avg): avg = general_avg(elems, prop_func=prop_func)
    if len(elems) > 1:
        sum = 0
        for elem in elems:
            sum += (prop_func(elem) - avg) ** 2
        return (sum / len(elems))
    else:
        return 0

def general_stdev(elems, prop_func=(lambda x: x), avg=None):
    return math.sqrt(general_variance(elems, prop_func=prop_func, avg=avg))

def list_average(L):    return sum(L) / len(L)

# Calc avg and stdev for each group of corresponding elems in multiple lists
def lists_avg_and_stdev(ls):
    ls2 = lists_transpose(ls)
    avgs = [list_average(l) for l in ls2]
    stdevs = [general_stdev(sublist,avg=a) for sublist,a in zip(ls2,avgs)]
    return avgs, stdevs

# ls is a list of sublists, with each sublist having the same length.  This treats the data structure like an
# array and tranposes, turning an M x N collection into an N x M collection.

def lists_transpose(ls):
    s1 = len(ls); s2 = len(ls[0])
    ls2 = [n_of(s1,0) for k in range(s2)]
    for i in range(s1):
        for j in range(s2):
            ls2[j][i] = ls[i][j]
    return ls2

def first_n_fast(elems,n,prop_func=(lambda x: x),dir='decrease'):
    rev = True if dir == 'decrease' else False
    comparator = (lambda x,y: x > y) if dir == 'decrease' else (lambda x,y: y > x)
    grp = elems[0:n]
    grp = sorted(grp,key=prop_func,reverse=rev)
    for elem in elems[n:]:
        if comparator(prop_func(elem),prop_func(grp[n-1])):  # compare to last elem of group
            grp = insert_into_sorted_list(elem,grp,prop_func=prop_func,comparator=comparator)
            grp = grp[0:n] # remove end
    return grp

# Comparator is > for a list sorted into decreasing order (largest first).
def insert_into_sorted_list(elem,elems,prop_func=(lambda x: x),comparator=(lambda x,y: x > y)):
    target = prop_func(elem)
    for i in range(len(elems)):
        if comparator(target,prop_func(elems[i])):
            elems.insert(i,elem)
            return elems
    return elems

def vector_len(v): return math.sqrt(sum([elem**2 for elem in v]))

def random_realvectors(count,size,min,max,norm=False):
    return [random_realvector(size,min,max,norm=norm) for i in range(count)]

def random_realvector(size,min,max,norm=False):
    v = np.random.uniform(min,max,size)
    return normalize_list(v) if norm else v  # normalize_list is in prims1.py

def normalize_list(elems,geometric=False):
    if geometric:
        s = math.sqrt(sum(map((lambda x: x**2), elems)))
    else:
        s = sum(elems)
    if s != 0:
        return [elem / s for elem in elems]
    else:
        return n_of(len(elems),0)

def vector_angle(v1,v2):
    l1 = vector_len(v1); l2 = vector_len(v2)
    if l1 == 0 or l2 == 0: return 0
    else:
        frac = np.dot(v1,v2)/(l1*l2)
        return math.acos(np.dot(v1,v2)/(l1*l2)) if abs(frac) <= 1 else 0

def entropy(distr, normalize=False,base=2):
    distribution = normalize_list(distr) if normalize else distr
    return -sum(d*safelog(d,base) for d in distribution)

def safelog(x,base=2): return math.log(x,base) if x > 0 else 0

# ***** GRAPHICS *******

def start_interactive_mode(): plt.ion()
def end_interactive_mode(): plt.ioff()
def set_current_figure(fig): plt.figure(fig.number)
def get_current_figure(): return plt.gcf()
def get_current_axes(): return plt.gcf().gca()
# This also sets the new figure to the current figure.
def newfig(num=None,figsize=None):
    return plt.figure(num=num,figsize=figsize)

# Block = true will leave control hanging until the figures are deleted.
def showall(block=False):
    plt.show(block=block)

def simple_plot(yvals,xvals=None,fig=None,xtitle='X',ytitle='Y',title='Y = F(X)',stdevs=None,colors=['k','b'],
                show=True,dump=False,fname=None,ftype=None,ylims=None):
    start_interactive_mode()
    if fig:  set_current_figure(fig)
    else: fig = newfig()
    xvals = xvals if xvals is not None else list(range(len(yvals)))
    plt.plot(xvals,yvals,color=colors[0])
    if stdevs is not None:  # Plot a stdev REGION
        plt.fill_between(xvals,[y-s for y,s in zip(yvals,stdevs)],[y+s for y,s in zip(yvals,stdevs)],color=colors[1])
    plt.xlabel(xtitle); plt.ylabel(ytitle); plt.title(title)
    if ylims is not None:  # Must do AFTER the call to plt.plot
        fig.gca().set_ylim(*ylims)
    if show: showall()
    if dump:    pickle_dump([xvals,yvals,stdevs],fname,ftype)
    end_interactive_mode()
    return fig

# Here, yvals is a list of lists, each of which is a series to be plotted.  Xvals should also be a list of
# lists.  'balance' => try to get each series to span the same range of x values such that all plots cover the
# same width of the graph.  This entails that the x's.  Options for xscale and yscale: linear, log, symlog, logit...

def multi_plot(yvals,xvals=None,fig=None,xtitle='X',ytitle='Y',title='Y = F(X)',xscale='linear',yscale='linear',
               show=True,legend=None,legloc='upper right',balance=True):
    start_interactive_mode()
    if fig:  set_current_figure(fig)
    else: fig = newfig()
    fig.gca().set_xscale(xscale); fig.gca().set_yscale(yscale)
    maxlen = max([len(yv) for yv in yvals])
    # xvals = xvals if xvals is not None else list(range(len(yvals[0])))
    for i, yvect in enumerate(yvals):
        if xvals is not None: xvect = xvals[i]
        elif balance:
            step = round(maxlen / len(yvect))
            xvect = list(range(0,step*len(yvect),step))
        else:
            xvect = list(range(len(yvect)))
        plt.plot(xvect,yvect)
    plt.xlabel(xtitle); plt.ylabel(ytitle); plt.title(title)
    if legend: fig.gca().legend(legend,loc=legloc)
    if show: showall()
    end_interactive_mode()
    return fig

def quickplot_pca(features,rounds=1000,dims=2,note=None,show_notes=True,title='PCA'):
    Y = pca(features,dims)
    note = note if note is not None else [0]*len(Y)
    quickplot_scatter(Y,note,dims=dims,note=show_notes,title=title)

# ****** Principle Component Analysis (PCA) ********
# This performs the basic operations outlined in "Python Machine Learning" (pp.128-135).  It begins with
# an N x K array whose rows are cases and columns are features.  It then computes the covariance matrix, which is
# then used to compute the eigenvalues and eigenvectors.  The eigenvectors corresponding to the largest (absolute
# value) eigenvalues are then combined to produce a transformation matrix, which is applied to the original
# N cases to produce N new cases, each with J (ideally J << K) features.  This is UNSUPERVISED dimension reduction.

def pca(features,target_size,bias=True,rowvar=False):
    farray = features if isinstance(features,np.ndarray) else np.array(features)
    cov_mat = np.cov(farray,rowvar=rowvar,bias=bias) # rowvar=False => each var's values are in a COLUMN.
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    return gen_dim_reduced_data(farray,target_size,eigen_vals, eigen_vecs)

# Use the highest magnitude eigenvalues (and their eigenvectors) as the basis for feature-vector transformations that
# reduce the dimensionality of the data.  feature_array is N x M, where N = # cases, M = # features

def gen_dim_reduced_data(feature_array,target_size,eigen_values,eigen_vectors):
    eigen_pairs = [(np.abs(eigen_values[i]),eigen_vectors[:,i]) for i in range(len(eigen_values))]
    eigen_pairs.sort(key=(lambda p: p[0]),reverse=True)  # Sorts tuples by their first element = abs(eigenvalue)
    best_vectors = [pair[1] for pair in eigen_pairs[ : target_size]]
    w_transform = np.array(best_vectors).transpose()
    return np.dot(feature_array,w_transform)


# Generate a simple predator neural network
def gen_pred_net(num_features, num_behaviors,hids=[10,10],lrate=0.01,act='sigmoid',lastact='sigmoid'):
    opt = KER.optimizers.SGD
    input = KER.layers.Input(shape=num_features, name='input_layer')
    x = input
    for hidden_layer_size in hids:
        x = KER.layers.Dense(hidden_layer_size,activation=act)(x)
    output = KER.layers.Dense(num_behaviors,activation=lastact)(x)
    model = KER.models.Model(input, output)
    model.compile(optimizer=opt(lr=lrate),loss='MSE')
    return model


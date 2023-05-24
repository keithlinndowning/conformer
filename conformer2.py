# This is a slightly simplified version of the file conformer.py, which houses the model used
# in my ALife Journal paper (2023).  The only difference is that the original uses my entire
# deepnet2.py code for running the neural networks (of the predictor model), whereas this version
# uses KERAS directly.  The neural networks are VERY simple, so this should not make a difference
# to the results.  However, the pure tensorflow used in deepnet2.py does run faster.  It was
# simply too much work to include all of the dependencies from deepnet2.py in this version, which I'm
# putting out on github.
# To run this code, you will need to install tensorflow and matplotlib.  These are used in the
# auxiliary file, conformer2_aux.py, which contains all of the supporting functions and classes
# from my other support files, such as prims1.py, kdstats.py, grapher.py, etc.

# To test this code, once loaded, do one of the 3 "pubruns" functions near the bottom of the file, or just
# call dopop() directly.  You will need to read my Alife Journal article to understand this model, since the
# comments are primarily there to help ME remember what I've done.

import numpy as np
import random
import math
import conformer2_aux as AUX
import copy

## ******* EMERGENCE OF CONFORMITY *******
# This takes a population of distributions and gradually modifies them in the following manner:
# 1) Take a random distribution d, 2) Find the distribution d* that is most similar to it (this represents a
# "recommendation" or "influencer", 3) modify d[k] to be more similar to d*[k] for a random k, 4) Renormalize d.
# NOTE:  The population converges as long as epsilon > 0.  When epsilon = 0, the use of ONLY best-match influencers
# leads to a few clusters of similar individuals (i.e. distributions).  But just a small epsilon is enough to push
# the population to full convergence.  Even when alpha = 0.5 (instead of 1.0), this still converges (albeit more slowly)
# as long as epsilon > 0. The partial convergence, to a few clusters, represents polarization.

class DistAgent():
    def __init__(self,dpop,init_vector):
        self.dpop = dpop
        self.distance = 0 # A slot used during distance comparisons = distance to a particular individual.
        self.set_distributions(init_vector)

    def set_distributions(self,d):
        self.features = AUX.normalize_list(d[0:self.dpop.num_features],geometric=True)
        self.behaviors = AUX.normalize_list(d[self.dpop.num_features:],geometric=True)

    def compare(self,d2):    return AUX.vector_angle(self.norm_merged_distr(),d2.norm_merged_distr())
    def get_features(self): return self.features
    def get_behaviors(self):    return self.behaviors
    def norm_merged_distr(self): return AUX.normalize_list(self.merged_distr(),geometric=True)
    def merged_distr(self):  return self.features + self.behaviors

    # For a DistAgent, interaction means being influenced by
    def interact(self,agent2):
        dist = self.compare(agent2)
        if dist < (self.dpop.beta1*np.pi/2):
            self.dpop.incr_peer_count(pos=True)  # Keep track of number of positive influences
            self.modify_dists(agent2,self.dpop.alpha,influence=+1)  # Make source more similar to target
        elif dist > (1-self.dpop.beta2)*np.pi/2:
            self.dpop.incr_peer_count(pos=False) # Keep track of number of negative influences
            self.modify_dists(agent2,self.dpop.alpha,influence=-1)  # Make source more DIFFERENT from target

    # When influence is positive, move source toward the target, otherwise away from it.
    def modify_dists(self,agent2,rate,influence=+1):
        if AUX.biased_coin_toss(0.5):
            source = self.features; target = agent2.features; flag = True
        else:
            source = self.behaviors; target = agent2.behaviors; flag = False
        k = AUX.randab_int(0,len(source)-1)
        source[k] += (target[k] - source[k])*rate*influence
        source[k] = max(0,source[k])
        if flag:
            self.features = AUX.normalize_list(source,geometric=True)
        else:   self.behaviors = AUX.normalize_list(source,geometric=True)

#  The predator agent wants to influence all agents to match its distribution.
class Predator(DistAgent):

    def __init__(self,dpop,goal_vals,strength,scale):
        self.dpop = dpop
        self.strength = strength # Max strength of influence (when error = 0)
        self.memory = []
        self.error_history = []  # Only add a new value (avg of current_errors) at end of each epoch
        self.current_errors = []
        self.error_threshold = (1/(dpop.predator_error_scale*math.sqrt(dpop.num_behaviors)))**2
        self.error_scale = scale
        self.set_distributions(goal_vals)  # These become the goal distributions (for features and behaviors)
        self.brain = dpop.gen_predator_brain()

    # A predator will NEVER change its distribution, but it can influence the other agent WHEN it is able
    # to predict that agent's behaviors based on its features.  The ability to make this prediction comes from
    # storing the feature-behavior distributions of the agents that it meets and then using those as training
    # cases for its neural network.

    def interact(self,agent2):
        if type(agent2) is not Predator:
            self.predict_and_influence(agent2)
            self.cache_info(agent2)

    def predict_and_influence(self,agent2):
        prediction = self.brain(np.array([np.array(agent2.get_features())]),training=False)[0]
        error = np.mean(np.square(np.subtract(agent2.get_behaviors(), prediction)))  # Error is small, in (0,1) or (0, 0.1)
        self.current_errors.append(error)
        if error < self.error_threshold:
            agent2.modify_dists(self,self.strength / (1 + self.dpop.num_behaviors**2 * error ** 0.5),influence=+1)

    def cache_info(self,agent2):
        self.memory.insert(0,copy.copy(agent2.merged_distr()))  # Add distribution as the newest memory
        self.memory = self.memory[:self.dpop.predator_memory_size]  # Truncate old memories if over-sized

    # Produce training cases for the neural net based on the remembered distributions
    def learn(self):
        if len(self.memory) > 0:
            flen = self.dpop.num_features
            features = [v[0:flen] for v in self.memory]
            targets = [v[flen:] for v in self.memory]
            # Train the KERAS net
            self.brain.fit(features, targets, epochs=1, batch_size=len(self.memory), validation_split=0, verbose=0)

    def update_error_history(self):
        val = np.average(self.current_errors) if len(self.current_errors) > 0 else 0
        self.error_history.append(val)
        self.current_errors = []

class DistPop():
    def __init__(self,size,length,behavs=4,alpha=0.25,eps=0.5,beta1=0.25,beta2=0.1,globe=True,preds=1,pmem=50,hood=1,
                 hids=[50,25],plrate=0.01,pstr=1.0,pes=2,prev=True,pbias=None,name='dpop'):
        self.popsize = size  # number(agents) + number(predators)
        self.basename = name  # Used for creating file names for data dumps
        self.num_predators = preds
        self.num_basic_agents = size - preds
        self.predator_memory_size = pmem  # size of each predator's memory buffer (old stuff is removed)
        self.predator_lrate = plrate
        self.predator_strength = (pstr if pstr is not None else alpha) # strength of predator's (max) influence
        self.predator_error_scale = pes # factor applied to error when computing influence
        self.hidden_layer_sizes = hids
        self.length = length
        self.num_behaviors = behavs   # Each agent's prob dist consists of features and behaviors.
        self.num_features = length - behavs
        self.alpha = alpha # degree to which a source prob will be modified in direction of the influencer
        self.epsilon = eps # prob of choosing a random indiv (instead of the best matching) as an influencer
        self.beta1 = beta1 # threshold for influencing positively (toward convergence)
        self.beta2 = beta2 # threshold for influencing negatively (toward divergence)
        self.hood_size = hood  # When using global evolution (the matchmaker), this is k = neighborhood size.
        self.peer_review = prev  # Are we saving and plotting info on when peer-to-peer influence occurs?
        self.meta_distance_history = []
        self.meta_predator_history = []
        if prev:
            self.meta_peer_pos_history = []
            self.meta_peer_neg_history = []
        self.global_compare = globe
        self.predator_bias = pbias if (pbias and preds > 0) else preds / size

    def run(self,epochs=100,steps=100,show=True,meshow=True,medump=True,reps=1):
        for r in range(reps):
            print("Starting run ", r)
            self.reset_local_caches()
            self.gen_agents()
            self.evolve(epochs=epochs,steps=steps,show=show)
            self.update_meta_caches()
        if meshow:  self.show_meta_results(dump=medump)

    def reset_local_caches(self):
        self.distance_history = []
        if self.peer_review:
            self.peer_counts = [0, 0] # number of pos and neg peer influences in the epoch
            self.peer_history = []  # This will hold pairs (pos-peer-influences neg-peer-influences)
        # Each new run creates new predator objects, so their error histories need not be reset.

    def update_meta_caches(self):
        self.meta_distance_history.append(self.distance_history)
        for pred in self.predators:
            self.meta_predator_history.append(pred.error_history)
        if self.peer_review:
            series = AUX.lists_transpose(self.peer_history)
            self.meta_peer_pos_history.append(series[0])
            self.meta_peer_neg_history.append(series[1])

    def update_histories(self):
        self.distance_history.append(self.compare_many())  # compare_all is computationally expensive
        if self.peer_review:
            self.peer_history.append(copy.copy(self.peer_counts))
            self.peer_counts = [0,0]

    def crunch_basic_meta_caches(self):
        phist_avgs,phist_stdevs = AUX.lists_avg_and_stdev(self.meta_predator_history) if self.num_predators > 0 else (None,None)
        dist_avgs,dist_stdevs = AUX.lists_avg_and_stdev(self.meta_distance_history)
        return phist_avgs, phist_stdevs, dist_avgs, dist_stdevs

    def crunch_peer_meta_caches(self):
        pos_avgs,pos_stdevs = AUX.lists_avg_and_stdev(self.meta_peer_pos_history)
        neg_avgs, neg_stdevs = AUX.lists_avg_and_stdev(self.meta_peer_neg_history)
        return pos_avgs, pos_stdevs,neg_avgs,neg_stdevs

    # peer counts is a pair: [num-positive-peer-influences, num-negative-peer-influences]
    def incr_peer_count(self,amt=1,pos=True):
        if self.peer_review:
            index = 0 if pos else 1
            self.peer_counts[index] += amt

    def gen_agents(self):
        n0 = self.popsize - self.num_predators
        self.predators = [Predator(self,AUX.random_realvector(self.length,0,1),
                                   self.predator_strength,self.predator_error_scale)
                          for _ in range(self.num_predators)]
        self.agents = [DistAgent(self,AUX.random_realvector(self.length,0,1))
                       for _ in range(self.popsize-self.num_predators)]
        self.agents.extend(self.predators)
        self.scatter_plot_labels = [1]*n0 + [0]*self.num_predators

    # Generate the neural network for a predator.  This network should map an agents features to its
    # behaviors, thus enabling a predator to PREDICT an agent's behavior from its features.
    # steps = # mini-batches to run each time it trains, tf=test fraction, vf=validation fraction,
    # tm = test mode (0 => no testing, just training).

    def gen_predator_brain(self):
        return AUX.gen_pred_net(self.num_features,self.num_behaviors,self.hidden_layer_sizes)

    def compare_all(self):
        s = 0
        for d1 in self.agents:
            for d2 in self.agents:
                if d1 != d2:    s += (1 - math.cos(d1.compare(d2)))
        return s / (self.popsize*(self.popsize-1))

    def compare_many(self,n=500):
        s = 0; count = 0
        for _ in range(n):
            i,j = np.random.randint(0,self.popsize,2)  #np.random.randint(a,b) gens from a to b-1
            if i != j:                                  # but random.randinit(a,b) gens from a to b !!
                s += (1 - math.cos(self.agents[i].compare(self.agents[j])))
                count += 1
        return s / count

    def find_best_match(self,d0):
        if self.hood_size == 1:
            min_dist = self.length; best = None
            for d1 in self.agents:
                if d0 != d1:
                    dist = d0.compare(d1)
                    if dist  < min_dist:
                        best =d1; min_dist = dist
        else:
            hood = self.find_nearest_neighbors(d0)
            best = random.choice(hood)  # Randomly choose a neighbor
        return best

    # Find the k agents closest to agent d0.  This is used during global evolution using a matchmaker.
    def find_nearest_neighbors(self,d0):
        d0.distance = math.pi  # This insures that d0 is not chosen as one of it's own nearest k neighbors.
        for d1 in self.agents:
            d1.distance = d0.compare(d1)
        return AUX.first_n_fast(self.agents,self.hood_size,prop_func=(lambda agent: agent.distance),dir='increase')

    def evolve(self,epochs=100,steps=100,show=True):
        for _ in range(epochs):
            if self.global_compare:
                self.evolve_one_epoch_global(steps)
            else:   self.evolve_one_epoch_local(steps)
            self.update_histories()
            for p in self.predators:
                p.update_error_history()
                p.learn()
        if show:   self.show_results()

    def evomore(self,epochs,steps=100,show=True):
        return self.evolve(epochs=epochs,steps=steps,show=show)

    def evolve_one_epoch_global(self,steps):
        for _ in range(steps):
            source_index = random.randint(0, self.popsize - 1)
            d0 = self.agents[source_index]
            d1 = AUX.randelem(self.agents) if AUX.biased_coin_toss(self.epsilon) else self.find_best_match(d0)
            d0.interact(d1)

    def evolve_one_epoch_local(self,steps):
        for _ in range(steps):
            if AUX.biased_coin_toss(self.predator_bias):
                d0 = np.random.choice(self.predators)
                d1 = np.random.choice(self.agents[0:self.num_basic_agents])
            else:
                source_index, target_index = np.random.choice(self.num_basic_agents,2,replace=False) # two unique ints in [0,popsize-1]
                d0 = self.agents[source_index]; d1 = self.agents[target_index]
            d0.interact(d1)

    def show_results(self):
        AUX.simple_plot(self.distance_history,xtitle='Epoch',ytitle='Diversity',title='Evolving Diversity')
        AUX.start_interactive_mode()
        AUX.quickplot_pca([np.array(d.norm_merged_distr()) for d in self.agents],note=self.scatter_plot_labels,show_notes=False)
        for p in self.predators:
            AUX.simple_plot(p.error_history,xtitle='Epoch',ytitle='Error',title='Predictor Error History')
        if self.peer_review:
            series = AUX.lists_transpose(self.peer_history)
            AUX.multi_plot(series,xtitle='Epoch',ytitle='Number',title='Peer Influence',legend=['Pos','Neg'])
        AUX.end_interactive_mode()
        print('Average Entropy = ',sum([AUX.entropy(d.norm_merged_distr()) for d in self.agents])/self.popsize)

    # Meta results are those averaged over 1 or more runs.
    def show_meta_results(self,dump=False):
        error_avg,error_stdev,div_avg,div_stdev = self.crunch_basic_meta_caches()
        fid = self.basename; ftype='conformity'
        AUX.simple_plot(div_avg, xtitle='Epoch', ytitle='Diversity', stdevs=div_stdev,
                       title='Run-Averaged Diversity', dump=dump,ftype=ftype,fname=fid+"-div")
        if error_avg:
            AUX.simple_plot(error_avg, xtitle='Epoch', ytitle='Error', stdevs=error_stdev,
                           title='Run-Averaged Predictor Error',dump=dump,ftype=ftype,fname=fid+"-err")
        if self.peer_review:
            pos_avg, pos_stdev, neg_avg,neg_stdev = self.crunch_peer_meta_caches()
            AUX.simple_plot(pos_avg,xtitle='Epoch',ytitle='Number',stdevs=pos_stdev,
                           title='Run-Averaged Pos Peer Influence',dump=dump,ftype=ftype,fname=fid+"-posinf")
            AUX.simple_plot(neg_avg, xtitle='Epoch', ytitle='Number', stdevs=neg_stdev,
                           title='Run-Averaged Neg Peer Influence',dump=dump,ftype=ftype,fname=fid+"-neginf")

# **** Debugging ****

    def print_dists(self):
        for a in self.agents:   print(a.merged_distr())

# This loads a pickled vector (xvalues, yvalues, stdevs) and plots it
def load_show(fid,gtype='div', xtitle='Epochs',ytitle=None,title=' ',ylims=None):
    ytitle = {'div': 'Diversity', 'err': 'Error','inf':'Count'}[gtype]
    ylims = ylims or {'div': [0,0.6], 'err': [0,0.1],'inf':[0,100]}[gtype]
    triple = AUX.pickle_load(fid,'conformity')
    AUX.simple_plot(triple[1],triple[0],stdevs=triple[2],xtitle=xtitle,ytitle=ytitle,title=title,ylims=ylims)

def diversity_to_angle(diversity):    return 180 * (math.acos(1-diversity) / math.pi)

# Find the distribution of angles among a large set of randomly-generated vectors
def angle_test(count=100,len1=10,len2=5,resolution=10,bias=0.2,deg=True):
    v1 = AUX.random_realvectors(count,len1,0,1);  v2 = AUX.random_realvectors(count,len2,0,1)
    nv1 = [AUX.normalize_list(v,geometric=True) for v in v1]; nv2 = [AUX.normalize_list(v,geometric=True) for v in v2];
    normvects = [AUX.normalize_list(n1+n2,geometric=True) for n1,n2 in zip(nv1,nv2)]
    angles = []
    for v1 in normvects:
        for v2 in normvects:
            if v1 is not v2:    angles.append(AUX.vector_angle(v1,v2))
    avg,stdev = list_avg_and_stdev(angles)
    k = 180 / math.pi if deg else 1
    print("Average angle: {0}, Stdev: {1}".format(avg*k,stdev*k))

# **** MAIN ****
# These are the high-level functions used to set all of this in motion.

def dopop(size=50,length=15,behavs=5,epochs=2500,steps=100,show=False,meshow=True,medump=True,name='dpop',reps=1,globe=False,hood=5,
          b1=.1,b2=.3,alpha=0.25,pbias=.1,eps=0.0,preds=1,plr=0.01,pstr=0.25,pes=.1,hids=[10,10],pmem=25,prev=True):
    p = DistPop(size,length,behavs=behavs,globe=globe,beta1=b1,beta2=b2,preds=preds,plrate=plr,eps=eps,hood=hood,
                name=name,alpha=alpha,pstr=pstr,pbias=pbias,pes=pes,hids=hids,pmem=pmem,prev=prev)
    p.run(epochs=epochs,steps=steps,show=show,meshow=meshow,medump=medump,reps=reps)
    # p.print_dists()
    return p

# Parameters:  Using plr = 0.01 with pstr=4, epochs=2500 and predator-error-scale = 25, you get a nice transition to
# conformity, but that transition slows down a lot when plr = 0.0001, thus illustrating the effect of a learning
# predator.  Even with plr=0, the predator still has a very gradual effect, but convergence is much slower.

# Without any predator, convergence does not occur unless b1 increases beyond .1.  When b1=b2=.1 (default), the
# tendency to converge is too weak, but the addition of the predator creates convergence.  When b1=b2=.1, there is
# still no convergence to 0 without a predator.  At b1=.3, b2=.1, we get convergence to a polarized state (several
# points), with a diversity value of 0.07 or lower.

# Using pes=.1 insures that the predator error threshold is essentially meaningless: the predator ALWAYS gets
# a little influence, even when error is high.  A default value of pes is 2.0.  See the predator model for
# the calculation of the threshold based on pes.

# These are the runs for the first section of my ALife Journal article (2023)
def pubruns1(reps=25,epochs=2500):
    dopop(name='basic-3-3',size=50,length=15,epochs=epochs,preds=0,alpha=0.25,reps=reps,globe=False,
          steps=100,b1=.3,b2=.3)
    dopop(name='basic-3-2', size=50, length=15, epochs=epochs, preds=0, alpha=0.25, reps=reps, globe=False,
          steps=100,b1=.3, b2=.2)
    dopop(name='basic-4-3', size=50, length=15, epochs=epochs, preds=0, alpha=0.25, reps=reps, globe=False,
          steps=100, b1=.4, b2=.3)

# These are the matchmaker runs for the ALife paper
def pubruns2(reps=25,epochs=2500):
    dopop(name='matchmaker-1', b1=.4, b2=.3,globe=True, preds=0, hood=1, epochs=epochs,reps=reps, eps=0)
    dopop(name='matchmaker-8',b1=.4, b2=.3,globe=True, preds=0, hood=8, epochs=epochs,reps=reps,eps=0)
    dopop(name='matchmaker-1-ep05',b1=.4, b2=.3,globe=True, preds=0, hood=1, epochs=epochs,reps=reps, eps=0.05)

# These are the predator runs for the ALife paper
def pubruns3(reps=25,epochs=2500,pes=0.1):
    dopop(name='pred-1-3',b1=.1,b2=.3,globe=False,preds=1,epochs=epochs,reps=reps,plr=0.01,pes=pes)
    dopop(name='pred-2-3', b1=.2, b2=.3, globe=False, preds=1, epochs=epochs, reps=reps, plr=0.01,pes=pes)
    dopop(name='pred-3-3', b1=.3, b2=.3, globe=False, preds=1, epochs=epochs, reps=reps, plr=0.01,pes=pes)
    dopop(name='pred-1-3-fast', b1=.1, b2=.3, globe=False, preds=1, epochs=epochs, reps=reps, plr=0.1,pes=pes)

def multipred(preds=2,reps=2,epochs=2500,pes=0.1):
    dopop(name='pred-1-3-fast', b1=.1, b2=.3, globe=False, preds=preds, epochs=epochs, reps=reps, plr=0.1, pes=pes)

def doall(reps=25,epochs=2500):
    pubruns1(reps,epochs)
    pubruns2(reps,epochs)
    pubruns3(reps,epochs)

"""
Run model of the full SCN consisting of 60 Gonze oscillators. The period
is found using the deterministic model to normalize the resulting trajectories.
The initial conditions are random--these were hard-coded in to return an 
identical figure every time, but they can be commented out for random start.

Generates for final figure of example trajectories from the paper with a 
2.5:1 AVP:VIP coupling strength ratio, and a 2:1 AVP:VIP cell count ratio.

John Abel
"""

from __future__ import division
import sys
assert sys.version[0]=='2', "This file must be run in python 2"
import pickle
from itertools import combinations

import numpy as np
import scipy as sp
import casadi as cs
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import minepy as mp

from local_imports import LimitCycle as lc
from local_imports import PlotOptions as plo
from local_models.gonze_model import y0in, param, ODEmodel, EqCount


#note: up to a point, this is the same as the file for the deterministic model!

# find original period
single_osc = lc.Oscillator(ODEmodel(), param, y0=np.ones(EqCount))
ts, sol = single_osc.int_odes(500)
single_osc.approx_y0_T(trans=2000)
wt_T = single_osc.T
single_osc.limit_cycle()

# number of each celltype
# these and the kav are hard-coded into the model
AVPcells = 53; VIPcells=27; NAVcells = 40
totcells = AVPcells+VIPcells+NAVcells

# initial phases
init_conditions_AV  = [single_osc.lc(wt_T*np.random.rand()) 
                        for i in range(AVPcells+VIPcells)]
init_conditions_NAV = [single_osc.lc(wt_T*np.random.rand())[:-1]
                        for i in range(NAVcells)]

y0_random = np.hstack(init_conditions_AV+init_conditions_NAV)
# so that figure is identical use the one actually generated



# switch to the multicellular stochastic model
from local_models.stoch_model_final import param, GonzeModelManyCells

# note that these relative strengths only are about IN THE ABSENCE OF THE OTHER

# make the stochastic figure
model = GonzeModelManyCells(param, initial_values=y0_random)
wt_trajectories = model.run(show_labels=False, seed=0)
wt_ts = wt_trajectories[0][:,0]
wt_avpsol = wt_trajectories[0][:,1:(AVPcells*4+1)]
wt_vipsol = wt_trajectories[0][:,(AVPcells*4+1):(AVPcells*4+VIPcells*4+1)]
wt_navsol = wt_trajectories[0][:,(AVPcells*4+VIPcells*4+1):]

# avp bmalko
avp_model = GonzeModelManyCells(param, initial_values=y0_random, bmalko='AVP')
avp_trajectories = avp_model.run(show_labels=False, seed=0)
avp_ts = avp_trajectories[0][:,0]
avp_avpsol = avp_trajectories[0][:,1:(AVPcells*4+1)]
avp_vipsol = avp_trajectories[0][:,(AVPcells*4+1):(AVPcells*4+VIPcells*4+1)]
avp_navsol = avp_trajectories[0][:,(320+1):]

vip_model = GonzeModelManyCells(param, initial_values=y0_random, bmalko='VIP')
vip_trajectories = vip_model.run(show_labels=False, seed=0)
vip_ts = vip_trajectories[0][:,0]
vip_avpsol = vip_trajectories[0][:,1:(AVPcells*4+1)]
vip_vipsol = vip_trajectories[0][:,(AVPcells*4+1):(AVPcells*4+VIPcells*4+1)]
vip_navsol = vip_trajectories[0][:,(AVPcells*4+VIPcells*4+1):]


# figure
plo.PlotOptions(ticks='in')
plt.figure(figsize=(3.5,3))
gs = gridspec.GridSpec(3,1)

ax = plt.subplot(gs[0,0])

ax.plot(wt_ts/wt_T, wt_vipsol[:,::4], 'darkorange', alpha=0.13, ls='--')
ax.plot(wt_ts/wt_T, wt_navsol[:,::3], 'goldenrod', alpha=0.13, ls='-.')
ax.plot(wt_ts/wt_T, wt_avpsol[:,::4], 'purple', alpha=0.13)
ax.plot(wt_ts/wt_T, wt_vipsol[:,::4].mean(1), 'darkorange', alpha=1, ls='--')
ax.plot(wt_ts/wt_T, wt_navsol[:,::3].mean(1), 'goldenrod', alpha=1, ls='-.')
ax.plot(wt_ts/wt_T, wt_avpsol[:,::4].mean(1), 'purple', alpha=1)

ax.set_xticks([0,1,2,3,4,5,6])
ax.set_xlim([0,6])
ax.set_xticklabels([])
ax.legend()
ax.set_ylim([0,300])
ax.set_yticks([0,100,200,300])

bx = plt.subplot(gs[1,0])
bx.plot(avp_ts/wt_T, avp_avpsol[:,::4], 'purple', alpha=0.13)
bx.plot(avp_ts/wt_T, avp_vipsol[:,::4], 'darkorange', alpha=0.13, ls='--')
bx.plot(avp_ts/wt_T, avp_navsol[:,::3], 'goldenrod', alpha=0.13, ls='-.')
bx.plot(avp_ts/wt_T, avp_avpsol[:,::4].mean(1), 'purple', alpha=1)
bx.plot(avp_ts/wt_T, avp_vipsol[:,::4].mean(1), 'darkorange', alpha=1, ls='--')
bx.plot(avp_ts/wt_T, avp_navsol[:,::3].mean(1), 'goldenrod', alpha=1, ls='-.')

bx.set_xticks([0,1,2,3,4,5,6])
bx.set_xticklabels([])
bx.set_xlim([0,6])
bx.set_ylim([0,300])
bx.set_yticks([0,100,200,300])


cx = plt.subplot(gs[2,0])
cx.plot(vip_ts/wt_T, vip_vipsol[:,::4], 'darkorange', alpha=0.13, ls='--')
cx.plot(vip_ts/wt_T, vip_navsol[:,::3], 'goldenrod', alpha=0.13, ls='-.')
cx.plot(vip_ts/wt_T, vip_avpsol[:,::4], 'purple', alpha=0.13)
cx.plot(vip_ts/wt_T, vip_vipsol[:,::4].mean(1), 'darkorange', alpha=1, ls='--')
cx.plot(vip_ts/wt_T, vip_navsol[:,::3].mean(1), 'goldenrod', alpha=1, ls='-.')
cx.plot(vip_ts/wt_T, vip_avpsol[:,::4].mean(1), 'purple', alpha=1)

cx.set_xticks([0,1,2,3,4,5,6])
cx.set_xlim([0,6])
cx.set_ylim([0,300])
cx.set_yticks([0,100,200,300])
cx.set_xlabel('Day')

plt.legend()
plt.tight_layout(**plo.layout_pad)
plt.savefig('results/model_figure_data.pdf')
plt.show()


# export data
header = ['TimesH', 'Frame']+['AVP'+str(i) for i in range(AVPcells)]+['VIP'+str(i) for i in range(VIPcells)]+['NAV'+str(i) for i in range(NAVcells)]

# wt traces
output_data = np.hstack([np.array([wt_ts*24/wt_T]).T,  np.array([np.arange(len(wt_ts))]).T, wt_avpsol[:,::4], wt_vipsol[:,::4], wt_navsol[:,::3] ])
output_df = pd.DataFrame(data=output_data,
            columns = header)
output_df.to_csv('results/WT_finalmodel_trajectories.csv', index=False)

# avp traces
output_data = np.hstack([np.array([avp_ts*24/wt_T]).T, np.array([np.arange(len(avp_ts))]).T, avp_avpsol[:,::4], avp_vipsol[:,::4], avp_navsol[:,::3] ])
output_df = pd.DataFrame(data=output_data,
            columns = header)
output_df.to_csv('results/AVPBmalKO_finalmodel_trajectories.csv', index=False)

# vip traces
output_data = np.hstack([np.array([vip_ts*24/wt_T]).T, np.array([np.arange(len(vip_ts))]).T, vip_avpsol[:,::4], vip_vipsol[:,::4], vip_navsol[:,::3] ])
output_df = pd.DataFrame(data=output_data,
            columns = header)
output_df.to_csv('results/VIPBmalKO_finalmodel_trajectories.csv', index=False)




##############################################################################
# perform simulations and statistical tests
##############################################################################

# try to load the simulation
try:    
    with open("data/wt_final.pickle", "rb") as read_file:
        wt_trajectories = pickle.load(read_file)
    with open("data/avp_final.pickle", "rb") as read_file:
        avp_trajectories = pickle.load(read_file)
    with open("data/vip_final.pickle", "rb") as read_file:
        vip_trajectories = pickle.load(read_file)
    print "Loaded final simulation."
    traj = {'wt': wt_trajectories,
              'avp': avp_trajectories,
              'vip': vip_trajectories}
except IOError:
    print "Final simulation does not exist yet."
    print "Simulating 100 iterations of final model."
    wt_trajectories = []
    avp_trajectories = []
    vip_trajectories = []
    for tn in range(100):
        print tn,
        # get random initial condition
        # initial phases
        init_conditions_AV  = [single_osc.lc(wt_T*np.random.rand()) 
                                for i in range(AVPcells+VIPcells)]
        init_conditions_NAV = [single_osc.lc(wt_T*np.random.rand())[:-1]
                                for i in range(NAVcells)]
        y0_random = np.hstack(init_conditions_AV+init_conditions_NAV)

        # do the simulation
        model = GonzeModelManyCells(param, initial_values=y0_random)
        wt_trajectories.append(model.run(show_labels=False, seed=0))

        # avp bmalko
        avp_model = GonzeModelManyCells(param, bmalko='AVP', 
                                        initial_values=y0_random)
        avp_trajectories.append(avp_model.run(show_labels=False, seed=0))

        # vip bmalko
        vip_model = GonzeModelManyCells(param, bmalko='VIP', 
                                        initial_values=y0_random)
        vip_trajectories.append(vip_model.run(show_labels=False, seed=0))

    # save results
    with open("data/wt_final.pickle", "wb") as output_file:
            pickle.dump(wt_trajectories, output_file)
    with open("data/avp_final.pickle", "wb") as output_file:
            pickle.dump(avp_trajectories, output_file)
    with open("data//vip_final.pickle", "wb") as output_file:
            pickle.dump(vip_trajectories, output_file)

    traj = {'wt': wt_trajectories,
              'avp': avp_trajectories,
              'vip': vip_trajectories}


# MIC Calculations
try:
    # load the results
    with open("results/finalmodel_mic_results.pickle", "rb") as input_file:
        results = pickle.load(input_file)
except IOError:
    print "MIC has not been calculated yet."
    print "Performing MIC calcualtion."
    # now perform MIC calculation
    def mic_of_simulation(trajectories):
        """
        returns the MIC values for one set of the SCN trajectories in question
        """

        avpvipsol = trajectories[:, 1:(160+1)]
        navsol = trajectories[:, (160+1):]

        per2 = np.hstack([avpvipsol[:, ::4], navsol[:, ::3]])
        numcells = per2.shape[1]

        # set up mic calculator
        mic = mp.MINE(alpha=0.6, c=15, est='mic_approx')
        mic_values = []
        for combo in combinations(range(numcells), 2):
            mic.compute_score(per2[:, combo[0]], per2[:, combo[1]])
            mic_values.append(mic.mic())

        return mic_values

    # process wt
    print "WT"
    wt_traj = traj['wt']
    wt = []
    for idx, ti in enumerate(wt_traj):
        print idx,
        mic_mean = np.mean(mic_of_simulation(ti[0]))
        wt.append(mic_mean)

    # process avp
    print "AVP-BmalKO"
    avp_traj = traj['avp']
    avp = []
    for idx, ti in enumerate(avp_traj):
        print idx,
        mic_mean = np.mean(mic_of_simulation(ti[0]))
        avp.append(mic_mean)

    # process vip
    print "VIP-BamelKO"
    vip_traj = traj['vip']
    vip = []
    for idx, ti in enumerate(vip_traj):
        print idx,
        mic_mean = np.mean(mic_of_simulation(ti[0]))
        vip.append(mic_mean)

    results = [wt, avp, vip]

    with open("results/finalmodel_mic_results.pickle", "wb") as output_file:
        pickle.dump(results, output_file)



# run the stats
# compare avpbmalko and vipbmalko vs. wt for all cases
# running a mann-whitney U test for nonparametric comparison
# recap of phenotypes is: AVPBmal1ko < VIPBmal1KO = WT
# signifiance of p < 0.05
# correct p-values with Bonferroni correction
# at each level, we are comparing: WT:AVP, WT:VIP, VIP:AVP so 3*7=21
# comparisons

r = results
wa = stats.mannwhitneyu(r[0], r[1], alternative='two-sided')[1]
wv = stats.mannwhitneyu(r[0], r[2], alternative='two-sided')[1]
av = stats.mannwhitneyu(r[1], r[2], alternative='two-sided')[1]
u_results = [wa, wv, av]
print u_results
correct_phenotype1 = [np.all([ur[0]<0.05, ur[1]>0.05, ur[2]<0.05]) 
                        for ur in u_results]








#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:02:54 2019

@author: maria
Dirichlet process mixtures-- Poisson
"""

#matplotlib inline
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt


with np.load('my_discete.npz') as data:
    d_train= data['train']

K = 50
N = d_train.shape[0]
x_plot = np.arange(250)
blue, *_ = sns.color_palette()
SEED = 5132290 # from random.org
np.random.seed(SEED)

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

    return beta * portion_remaining


with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 1, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))

    mu = pm.Uniform('mu', 0., 300., shape=K)
    obs = pm.Mixture('obs', w, pm.Poisson.dist(mu), observed=d_train[:,0])


with model:
    step = pm.Metropolis()
    trace = pm.sample(5000, step=step, random_seed=SEED)
    
# plots
pm.traceplot(trace, varnames=['alpha']);

fig, ax = plt.subplots(figsize=(8, 6))

plot_w = np.arange(K) + 1

ax.bar(plot_w - 0.5, trace['w'].mean(axis=0), width=1., lw=0);
ax.set_xlim(0.5, K);
ax.set_xlabel('Component');
ax.set_ylabel('Posterior expected mixture weight');   


post_pmf_contribs = sp.stats.poisson.pmf(np.atleast_3d(x_plot),
                                         trace['mu'][:, np.newaxis, :])
post_pmfs = (trace['w'][:, np.newaxis, :] * post_pmf_contribs).sum(axis=-1)
post_pmf_low, post_pmf_high = np.percentile(post_pmfs, [2.5, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(8, 6))

ax.hist(d_train[:,0], bins=40, normed=True, lw=0, alpha=0.75);
ax.fill_between(x_plot, post_pmf_low, post_pmf_high,
                 color='gray', alpha=0.45)
ax.plot(x_plot, post_pmfs[0],
        c='gray', label='Posterior sample densities');
ax.plot(x_plot, post_pmfs[::200].T, c='gray');
ax.plot(x_plot, post_pmfs.mean(axis=0),
        c='k', label='Posterior expected density');

ax.set_xlabel('Observartions');
ax.set_yticklabels([]);
ax.legend(loc=1);


fig, ax = plt.subplots(figsize=(8, 6))

ax.hist(d_train[:,0], bins=40, normed=True, lw=0, alpha=0.75);
ax.plot(x_plot, post_pmfs.mean(axis=0),
        c='k', label='Posterior expected density');
ax.plot(x_plot, (trace['w'][:, np.newaxis, :] * post_pmf_contribs).mean(axis=0)[:, 0],
        '--', c='k', label='Posterior expected\nmixture components\n(weighted)');
ax.plot(x_plot, (trace['w'][:, np.newaxis, :] * post_pmf_contribs).mean(axis=0),
        '--', c='k');

ax.set_xlabel('Yearly sunspot count');
ax.set_yticklabels([]);
ax.legend(loc=1);

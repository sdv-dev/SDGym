#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: maria
Dirichlet process mixtures
"""
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt

with np.load('gaussian_grid.npz') as data:
    g_train= data['train']

N = g_train.shape[0]
K = 30
x_plot = np.linspace(-10, 10, 500)
blue, *_ = sns.color_palette()
SEED = 54321
np.random.seed(SEED)


def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

    return beta * portion_remaining

with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 1., alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))
    tau = pm.Gamma('tau', 1., 1., shape=K)
    lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
    mu = pm.Normal('mu', 0, tau=lambda_ * tau, shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=lambda_ * tau,
                           observed=g_train[:,0])
    
    
with model:
    trace = pm.sample(5000, random_seed=SEED)
    
pm.traceplot(trace, varnames=['alpha']);


# plots
fig, ax = plt.subplots(figsize=(8, 6))
plot_w = np.arange(K) + 1
ax.bar(plot_w - 0.5, trace['w'].mean(axis=0), width=1., lw=0);

ax.set_xlim(0.5, K);
ax.set_xlabel('Component');
ax.set_ylabel('Posterior expected mixture weight');

post_pdf_contribs = sp.stats.norm.pdf(np.atleast_3d(x_plot),
                                      trace['mu'][:, np.newaxis, :],
                                      1. / np.sqrt(trace['lambda'] * trace['tau'])[:, np.newaxis, :])
post_pdfs = (trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)
post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [2.5, 97.5], axis=0)
fig, ax = plt.subplots(figsize=(8, 6))

# histogram
n_bins = 20
ax.hist(g_train, bins=n_bins, normed=True,
        color=blue, lw=0, alpha=0.5);

ax.fill_between(x_plot, post_pdf_low, post_pdf_high,
                color='gray', alpha=0.45);
ax.plot(x_plot, post_pdfs[0],
        c='gray', label='Posterior sample densities');
ax.plot(x_plot, post_pdfs[::100].T, c='gray');
ax.plot(x_plot, post_pdfs.mean(axis=0),
        c='k', label='Posterior expected density');

ax.set_xlabel('Observations');
ax.set_yticklabels([]);
ax.set_ylabel('Density');

ax.legend(loc=2);

fig, ax = plt.subplots(figsize=(8, 6))

n_bins = 20
ax.hist(g_train, bins=n_bins, normed=True,
        color=blue, lw=0, alpha=0.5);

ax.plot(x_plot, post_pdfs.mean(axis=0),
        c='k', label='Posterior expected density');
ax.plot(x_plot, (trace['w'][:, np.newaxis, :] * post_pdf_contribs).mean(axis=0)[:, 0],
        '--', c='k', label='Posterior expected mixture\ncomponents\n(weighted)');
ax.plot(x_plot, (trace['w'][:, np.newaxis, :] * post_pdf_contribs).mean(axis=0),
        '--', c='k');

ax.set_xlabel('Observations');
ax.set_yticklabels([]);
ax.set_ylabel('Density');
ax.legend(loc=2);


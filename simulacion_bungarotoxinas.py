# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:12:58 2026

@author: Luis1
"""



#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(0)
tau2 = 1.0
tau1 = 3.0
R0 = 5.5
r = 8.0
E = 1.0 / (1.0 + (r / R0)**6)

t = np.linspace(0, 10, 100)      # time vector
bin_width = t[1] - t[0]          # correct bin width
N_total = 7000
frac1 = 0.5
N1 = int(N_total * frac1)
N2 = N_total - N1

def exp_pdf(t_array, tau):
    pdf = (1.0 / tau) * np.exp(-t_array / tau)
    pdf /= (np.sum(pdf) * bin_width)
    return pdf

pdf1 = exp_pdf(t, tau1)
pdf2 = exp_pdf(t, tau2)
mix_counts_before = (N1 * pdf1 + N2 * pdf2) * bin_width
counts_before = np.random.poisson(mix_counts_before)

# FRET: shorten donor lifetime and transfer photons to acceptor
tau1_eff = tau1 * (1.0 - E)
N_transferred = int(round(N1 * E))
N1_after = N1 - N_transferred
N2_after = N2 + N_transferred
pdf1_fret = exp_pdf(t, tau1_eff)
pdf2_fret = pdf2
mix_counts_after = (N1_after * pdf1_fret + N2_after * pdf2_fret) * bin_width
counts_after = np.random.poisson(mix_counts_after)
print("tau1 before:", tau1, "tau1_eff after FRET:", tau1_eff)

# MLE (Poisson) fit for single exponential tail: lambda_i = A * exp(-t/tau) + C
def neglogL_poisson(params, tvals, counts):
    A, tau, C = params
    if A < 0 or tau <= 0 or C < 0:
        return 1e12
    lam = A * np.exp(-tvals / tau) + C
    lam = np.maximum(lam, 1e-12)
    # Poisson negative log-likelihood (up to constant): sum(lam - k*log(lam))
    return np.sum(lam - counts * np.log(lam))

cut_time = 3
mask = t >= cut_time
t_tail = t[mask]

# Fit BEFORE FRET
y_before_tail = counts_before[mask].astype(float)
if y_before_tail.sum() > 0:
    p0 = [y_before_tail.max(), 3.0, np.median(y_before_tail[-5:])]
    res_b = minimize(neglogL_poisson, p0, args=(t_tail, y_before_tail),
                     bounds=[(0, None), (1e-3, 50.0), (0, None)])
    if res_b.success:
        A_b, tau_b, C_b = res_b.x
    else:
        A_b = tau_b = C_b = np.nan
else:
    A_b = tau_b = C_b = np.nan

# Fit AFTER FRET
y_after_tail = counts_after[mask].astype(float)
if y_after_tail.sum() > 0:
    p0 = [y_after_tail.max(), 3.0, np.median(y_after_tail[-5:])]
    res_a = minimize(neglogL_poisson, p0, args=(t_tail, y_after_tail),
                     bounds=[(0, None), (1e-3, 50.0), (0, None)])
    if res_a.success:
        A_a, tau_a, C_a = res_a.x
    else:
        A_a = tau_a = C_a = np.nan
else:
    A_a = tau_a = C_a = np.nan

print(f"MLE tail fit (t>={cut_time} ns) BEFORE FRET tau = {tau_b:.4f} ns")
print(f"MLE tail fit (t>={cut_time} ns) AFTER  FRET tau = {tau_a:.4f} ns")

# quick plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(t, counts_before, width=bin_width, color='C0', alpha=0.6, label='No fret')
if not np.isnan(tau_b):
    plt.plot(t_tail, A_b*np.exp(-t_tail/tau_b)+C_b, 'k--', label=f'fit tau={tau_b:.3f}')
plt.xlabel('Time (ns)'); plt.ylabel('Counts/bin'); plt.legend()

plt.subplot(1,2,2)
plt.bar(t, counts_after, width=bin_width, color='C1', alpha=0.6, label='Fret')
if not np.isnan(tau_a):
    plt.plot(t_tail, A_a*np.exp(-t_tail/tau_a)+C_a, 'r--', label=f'fit tau={tau_a:.3f}')
plt.xlabel('Time (ns)'); plt.ylabel('Counts/bin'); plt.legend()
plt.tight_layout(); plt.show()




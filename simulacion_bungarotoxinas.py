# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:12:58 2026

@author: Luis1
"""



#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit

np.random.seed(0)
tau1 = 3
tau2 = 1.5

R0 = 6.5
r = 8
E = 1.0 / (1.0 + (r / R0)**6)
t = np.linspace(0, 15, 100)      # time vector
bin_width = t[1] - t[0]          # correct bin width
#%%


#datos 1 (con y sin fret)
N_total = 10000
frac1 = 0.4
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
#counts_before  = np.random.poisson(N1*pdf1)


# FRET: 
tau1_eff = tau1 * (1.0 - E)
N_transferred = int(round(N1 * E))
N1_after = N1 - N_transferred
N2_after = N2 + N_transferred
pdf1_fret = exp_pdf(t, tau1_eff)
pdf2_fret = pdf2
mix_counts_after = (N1_after * pdf1_fret + N2_after * pdf2_fret) * bin_width
counts_after = np.random.poisson(mix_counts_after)
print("tau1 before:", tau1, "tau1_eff after FRET:", tau1_eff)

#%%
rng = np.random.default_rng(0)
tau1, tau2 = 3.0, 1.5      # ns
N1, N2 = 4000, 6000        # número de fotones por población
irf_sigma = 0.08           # ns, jitter IRF (0 para no aplicar)

# sample tiempos continuos
t1 = -tau1 * np.log(rng.random(N1))
t2 = -tau2 * np.log(rng.random(N2))

# opcional: añadir jitter IRF
# if irf_sigma > 0:
#     t1 = t1 + rng.normal(0, irf_sigma, size=N1)
#     t2 = t2 + rng.normal(0, irf_sigma, size=N2)

# combinar y (opcional) plegar modulo periodo de láser
t_all = np.concatenate([t1, t2])
# t_all = np.mod(t_all, laser_period)   # si hace falta

# binning a histograma
t_min, t_max, n_bins = 0.0, 15.0, 100
bins = np.linspace(t_min, t_max, n_bins+1)
hist, edges = np.histogram(t_all, bins=bins)

# resultados
bin_centers = 0.5*(edges[:-1]+edges[1:])
print("Total photons simulated:", t_all.size)
# hist contiene los conteos por bin

# FRET: 
tau1_eff = tau1 * (1.0 - E)
N_transferred = int(round(N1 * E))
N1_after = N1 - N_transferred
N2_after = N2 + N_transferred
t1_after =  -tau1_eff* np.log(rng.random(N1_after))
t2_after = -tau2 * np.log(rng.random(N2_after))
t_after_all = np.concatenate([t1_after, t2_after])
hist_after, edges_after = np.histogram(t_after_all, bins = bins)

#%%


cut_time = 0
mask_tail = t >= cut_time
t_tail_before = t[mask_tail]
#y_tail_before = counts_before[mask_tail]
y_tail_before = hist
t_tail_after = t[mask_tail]
#y_tail_after = counts_after[mask_tail]
y_tail_after = hist_after
def exponencial(t, A, tau, C):
    return A * np.exp(-t / tau) + C


def ajuste_exponencial(x, y, tau_0, fondo):
    p0 = [max(y.max(), 1.0), tau_0, max(fondo, 0.0)]
    try:
        popt, pcov = curve_fit(exponencial, x, y, p0=p0, maxfev=5000, bounds=([0,1e-3,0],[np.inf,100, np.inf]))
    except Exception as e:
        print("Fit failed:", e); return
    A_fit, tau_fit, C_fit = popt
    if pcov is None:
        tau_err = np.nan
    else:
        tau_err = np.sqrt(np.abs(np.diag(pcov)))[1]

    plt.figure(figsize=(7,5))
    plt.plot(x, y, 'o', label='Datos', color="salmon")
    t_model = np.linspace(x.min(), x.max(), 400)
    plt.plot(t_model, exponencial(t_model, *popt), '-', label=f'Ajuste: τ = {tau_fit:.2f} ± {tau_err:.2f} ns', color="slategray")
    plt.xlabel("Tiempo [ns]"); plt.ylabel("Cuentas")
    plt.grid(); plt.legend(); plt.tight_layout(); plt.show()
    print(f"Tiempo de vida: {tau_fit:.3f} ± {tau_err:.3f} ns")

# call on tail data (do not pass full t unless you intend to)
ajuste_exponencial(t_tail_before, y_tail_before, tau_0=3.0, fondo=np.median(y_tail_before[-5:]))
ajuste_exponencial(t_tail_after, y_tail_after, tau_0=2.33, fondo=np.median(y_tail_after[-5:]))

#%%


tau1_true = tau1      # del script previo
tau2_true = tau2

def fit_fixed_taus(x, y, tau1_fix, tau2_fix, p0=None):
    """
    Ajusta y = A1*exp(-t/tau1_fix) + A2*exp(-t/tau2_fix) + C
    Solo ajusta A1, A2 y C (taus fijos).
    Entradas:
      x, y        : arrays de datos
      tau1_fix    : tau1 fijo (float)
      tau2_fix    : tau2 fijo (float)
      p0          : lista inicial opcional [A1, A2, C]
    Devuelve:
      popt (A1,A2,C), perr (errores), y_fit (modelo en x)
    """
   

    def model_fixed(x, A1, A2, C):
        return A1 * np.exp(-x / tau1_fix) + A2 * np.exp(-x / tau2_fix) + C

    # semillas razonables si no se proveen
    if p0 is None:
        A_guess = max(y.max(), 1.0) / 2.0
        p0 = [A_guess, A_guess, max(np.median(y[-5:]) if len(y) >= 5 else 0.0, 0.0)]

    bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])  # A1,A2,C >= 0
    try:
        popt, pcov = curve_fit(model_fixed, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception as e:
        print("Fit failed:", e)
        return None, None, None

    perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else [np.nan, np.nan, np.nan]
    A1_fit, A2_fit, C_fit = popt
    y_fit = model_fixed(x, *popt)

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(x, y, 'o', label='data', color="salmon")
    xfine = np.linspace(x.min(), x.max(), 300)
    yfine = A1_fit * np.exp(-xfine / tau1_fix) + A2_fit * np.exp(-xfine / tau2_fix) + C_fit
    plt.plot(xfine, yfine, '-', label=f'fit fixed taus: τ1={tau1_fix:.2f}, τ2={tau2_fix:.2f}', color = "slategray")
    plt.grid()
    plt.xlabel('Tiempo [ns]'); plt.ylabel('Cuentas'); plt.legend(); plt.tight_layout(); plt.show()

    print(f"A1={A1_fit:.6g} ± {perr[0]:.6g}")
    print(f"A2={A2_fit:.6g} ± {perr[1]:.6g}")
    print(f"C ={C_fit:.6g} ± {perr[2]:.6g}")

    return popt, perr, y_fit


# Aplicar a before y after (usar los arrays que ya tienes)
fit_fixed_taus(t_tail_before, y_tail_before, tau1_true, tau2_true)
fit_fixed_taus(t_tail_after,y_tail_after, tau1_eff, tau2_true)
# para after, tau1 se reduce por FRET (tau1*(1-E)), si querés usar ese tau
#tau1_after = tau1_true * (1.0 - E)
#fit_fixed_taus(t_tail_after, y_tail_after, tau1_after, tau2_true)
#%% Sin Fret

# supuestos: popt = [A1, A2, C] si taus fijos; o popt_free = [A1,tau1,A2,tau2,C]
# perr correspondiente; bin_width definido (ej: bin_width = t[1]-t[0])

x = t_tail_before
y = hist
tau1_fix = tau1
tau2_fix = tau2
# Caso 1: taus fijos, popt_fixed returned by fit_fixed_taus
popt_fixed, perr_fixed, yfit_fixed = fit_fixed_taus(x, y, tau1_fix, tau2_fix)
A1_fit, A2_fit, C_fit = popt_fixed
errA1, errA2, errC = perr_fixed

N1_est = A1_fit * tau1_fix / bin_width
N2_est = A2_fit * tau2_fix / bin_width

# errores (si tau fijos, sólo de A)
errN1 = errA1 * tau1_fix / bin_width
errN2 = errA2 * tau2_fix / bin_width

print(f"N1_est = {N1_est:.1f} ± {errN1:.1f}")
print(f"N2_est = {N2_est:.1f} ± {errN2:.1f}")


#%% Con Fret (corregimos)
x = t_tail_after
y = hist_after
tau1_fix = tau1_eff
tau2_fix = tau2
# Caso 1: taus fijos, popt_fixed returned by fit_fixed_taus
popt_fixed, perr_fixed, yfit_fixed = fit_fixed_taus(x, y, tau1_fix, tau2_fix)
A1_fit, A2_fit, C_fit = popt_fixed
errA1, errA2, errC = perr_fixed

N1_fret = A1_fit * tau1_fix / bin_width
N1_est = N1_fret + N_transferred
N2_fret = A2_fit * tau2_fix / bin_width
N2_est = N2_fret - N_transferred

# errores (si tau fijos, sólo de A)
errN1 = errA1 * tau1_fix / bin_width
errN2 = errA2 * tau2_fix / bin_width

print(f"N1_est = {N1_est:.1f} ± {errN1:.1f}")
print(f"N2_est = {N2_est:.1f} ± {errN2:.1f}")


#%% Ahora, partiendo de N_tot, N1 y N2, quiero recuperar tau 1 y tau 2
N_tot = 10000
frac = 0.5
N_1 = frac*N_tot
N_2 = N_tot - N_1

x = t_tail_before
y = hist_after

def model_from_N_and_taus(x, N1, N2, tau1, tau2, bin_width, C):
    A1 = N1 * (bin_width / tau1)
    A2 = N2 * (bin_width / tau2)
    return A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + C

def fit_taus_with_fixed_N(x, y, N1, N2, bin_width,
                          tau1_p0=3.0, tau2_p0=1.5, fit_C=True,
                          tau_bounds=(0.05, 100.0), maxfev=200000):
    """
    Ajusta tau1, tau2 (y opcionalmente C) manteniendo N1,N2 fijos.
    Devuelve dict con taus ajustados, errores y y_fit.
    """
    x = np.asarray(x); y = np.asarray(y)
    if fit_C:
        def model_vars(x, tau1, tau2, C):
            return model_from_N_and_taus(x, N1, N2, tau1, tau2, bin_width, C)
        p0 = [tau1_p0, tau2_p0, max(np.median(y[-5:]), 0.0)]
        lower = [tau_bounds[0], tau_bounds[0], 0.0]
        upper = [tau_bounds[1], tau_bounds[1], np.inf]
        bounds = (lower, upper)
    else:
        def model_vars(x, tau1, tau2):
            return model_from_N_and_taus(x, N1, N2, tau1, tau2, bin_width, 0.0)
        p0 = [tau1_p0, tau2_p0]
        lower = [tau_bounds[0], tau_bounds[0]]
        upper = [tau_bounds[1], tau_bounds[1]]
        bounds = (lower, upper)

    try:
        popt, pcov = curve_fit(model_vars, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
    except Exception as e:
        print("Fit failed:", e)
        return None

    perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.array([np.nan]*len(popt))

    if fit_C:
        tau1_fit, tau2_fit, C_fit = popt
        tau1_err, tau2_err, C_err = perr
        y_fit = model_vars(x, tau1_fit, tau2_fit, C_fit)
    else:
        tau1_fit, tau2_fit = popt
        tau1_err, tau2_err = perr
        C_fit, C_err = 0.0, 0.0
        y_fit = model_vars(x, tau1_fit, tau2_fit)

    # imprimir y plot
    print(f"Fitted tau1 = {tau1_fit:.4f} ± {tau1_err:.4f}  ns")
    print(f"Fitted tau2 = {tau2_fit:.4f} ± {tau2_err:.4f}  ns")
    if fit_C:
        print(f"Fitted C    = {C_fit:.4g} ± {C_err:.4g}")

    plt.figure(figsize=(6,4))
    plt.plot(x, y, 'o', label='data', color='C1')
    xf = np.linspace(x.min(), x.max(), 400)
    if fit_C:
        yf = model_from_N_and_taus(xf, N1, N2, tau1_fit, tau2_fit, bin_width, C_fit)
    else:
        yf = model_from_N_and_taus(xf, N1, N2, tau1_fit, tau2_fit, bin_width, 0.0)
    plt.plot(xf, yf, '-', label=f'fit taus: τ1={tau1_fit:.2f}, τ2={tau2_fit:.2f}')
    plt.xlabel('t'); plt.ylabel('counts'); plt.legend(); plt.tight_layout(); plt.show()

    return {"tau1":tau1_fit, "tau1_err":tau1_err,
            "tau2":tau2_fit, "tau2_err":tau2_err,
            "C":C_fit, "C_err":C_err, "y_fit":y_fit}

bin_width = t[1] - t[0]
res = fit_taus_with_fixed_N(x, y, N_1, N_2, bin_width, tau1_p0=2.33, tau2_p0=1.5, fit_C=True)
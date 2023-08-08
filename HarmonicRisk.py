#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:19:12 2022

@author: jac
"""

from numpy import exp, pi, sqrt, arctan, sin, cos, linspace, sign
from numpy import min as npmin

from scipy.integrate import solve_ivp
from matplotlib.pylab import subplots


def HarmonicR_RHS( t, Y, w0, eta):
    """Harmonic ODE."""
    r, h, H = Y
    return [-2*eta*w0 * r - w0**2 * h , r, h]

def HarmonicR_ODE( tN, hb, hI, r0, w0, eta, t_eval=None):
    """Harmonic risk function using the ODE solver."""
    ### Solve using the LSODA solver, note that H(0) = 0
    rt = solve_ivp( fun=HarmonicR_RHS, t_span=(0, tN), t_eval=t_eval, y0=[ r0, hI, 0],\
          method='LSODA', vectorized=True, args=( w0, eta))
    ###      t      r           h              H
    return rt.t, rt.y[0,:], hb+rt.y[1,:], rt.t*hb+rt.y[2,:] 


def HarmonicR_ODE_Pars( tau, w2, hb, h0, r0):
    """Returns hI, w0 and eta."""
    hI = h0-hb
    
    w1 = 2*pi*w2/tau #angular frequency
    eta = 1/sqrt((tau*w1)**2 + 1) #damping ratio < 1
    w0 = 1/(eta*tau) #natural frequency
    return hI, w0, eta

def PlotHarmonicRiskODE( tau=1, w2=1/2, hb=1, h0=1.5, r0=0.0,\
        trange_mult=3, time_unit="", color="black", ax=None):
    """  
    Plot the harmonic risk function:

    tau (= 1) # Mean decay time 
    w2  (= 1/2) # Number of cycles in one mean decay time tau
    hb  (= 1.0) # Limit asymptotic risk    
    h0  (= 1.5) # Inicial value for risk
    r0  (= 0.0) # inicial "velocity" for risk
    trange_mult (=3) # tN = trange_mult*tau, time range to solve and plot h,
             number of mean decay times x tau
    time_unit = "" #e.g. "yr"
    color (='black')`# for plotting h
    ax # Axes to plot to, if None, create one
    returns ax.
    """
    
    if ax is None:
        fig, ax = subplots() #(nrows=1,ncols=2)

    tN = trange_mult*tau

    hI, w0, eta = HarmonicR_ODE_Pars( tau, w2, hb, h0, r0)
    t, r, h, H = HarmonicR_ODE( tN, hb, hI, r0, w0, eta)
    ax.plot( t, h, '-', color=color)
    ax.set_xlabel(r"$t$ "+time_unit)
    ax.set_ylabel(r"$h(t)$")
    ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
    ax.axhline(y=hb, color='black')
    return ax


def HarmonicR_AnalyticPars( tau, w2, hb, h0, r0):
    """
    Returns the initial value for h (hI), angular frequancy (w1), amplitude (A), phase (phi)
    and minimum (tm), of the analytic solution of the harmonic risk function:

    tau  # Mean decay time 
    w2   # Number of cycles in one mean decay time tau
    hb   # Limit asymptotic risk    
    h0   # Inicial value for risk
    r0   # inicial "velocity" for risk
    
    """
    w1 = 2*pi*w2/tau  # angular frequancy = w_0 \sqrt{1 - eta^2}
    hI = h0-hb
    tmp = (r0 + hI/tau)
    if hI == 0.0:
        phi = 0.0
        A = r0/w1
    elif tmp == 0.0:
        phi = sign(hI)*pi/2
        A = hI
    else:
        phi = arctan((w1*hI)/tmp)
        A = hI/sin(phi)
    tm1 = (arctan(tau*w1)      - phi)/w1
    tm2 = (arctan(tau*w1) + pi - phi)/w1
    h1 = HarmonicR_Analytic_h( tm1, tau, w1, A, phi, hb)
    h2 = HarmonicR_Analytic_h( tm2, tau, w1, A, phi, hb)
    if h1 < h2:
        tm = tm1
        hm = h1
    else:
        tm = tm2
        hm = h2
    #print( w1, A, phi, tm, hm)
    return w1, A, phi, tm, hm

def HarmonicR_Analytic_h( t, tau, w1, A, phi, hb):
    """Analytic solution for risk function h.
        tau: mean decay time
        w1 : angular frequancy = w_0 \sqrt{1 - eta^2}
        A  : amplitude
        phi: phase
        hb : asymptotic value
    """
    ###                           2*pi*w2/tau
    return hb + A*exp(-t/tau)*sin(w1*t + phi) #

def HarmonicR_Analytic_H( t, tau, w1, A, phi, hb):
    """Analytic solution for cummulative risk function H.
        tau: mean decay time
        w1 : angular frequancy = w_0 \sqrt{1 - eta^2}
        A  : amplitude
        phi: phase
        hb : asymptotic value
    """
    a = tau*w1
    return hb*t + A*tau*((sin(phi)+a*cos(phi)) - \
      exp(-t/tau)*(sin(w1*t + phi)+a*cos(w1*t+phi)))/(a**2 + 1)

def PlotHarmonicRiskAnalytic( tau=1, w2=1/2, hb=1, h0=1.5, r0=0.0,\
        trange_mult=3, time_unit="", color="black", N=100, ax=None):
    """  
    Plot the harmonic risk function:

    tau (= 1) # Mean decay time 
    w2  (= 1/2) # Number of cycles in one mean decay time tau
    hb  (= 1.0) # Limit asymptotic risk    
    h0  (= 1.5) # Inicial value for risk
    r0  (= 0.0) # inicial "velocity" for risk
    trange_mult (=3) # tN = trange_mult*tau, time range to solve and plot h,
             number of mean decay times x tau
    time_unit = "" #e.g. "yr"
    color (='black')`# for plotting h
    N   (= 100) # Grid size for plottimng
    ax # Axes to plot to, if None, create one
    returns ax.
    """
    
    if ax is None:
        fig, ax = subplots() #(nrows=1,ncols=2)

    tN = trange_mult*tau

    w1, A, phi, tm, hm = HarmonicR_AnalyticPars( tau, w2, hb, h0, r0)
    if (tm < tN) and (hm < 0.0):
        print("WARNIG: negative h.")
    t = linspace( 0, tN, num=N)
    h = HarmonicR_Analytic_h( t, tau=tau, w1=w1, A=A, phi=phi, hb=hb)
    ax.plot( t, h, '-', color=color) #
    ax.set_xlabel(r"$t$ "+time_unit)
    ax.set_ylabel(r"$h(t)$")
    ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
    ax.axhline(y=hb, color='black')
    ### Mark the minimum
    ax.axvline(x=tm, color=color, linestyle='--')
    return ax

def HarmonicR( t, tau, w2, hb, h0, r0, ode=False):
    
    tN = max(t)
    if ode:
        hI, w0, eta = HarmonicR_ODE_Pars( tau, w2, hb, h0, r0)
        t, r, h, H = HarmonicR_ODE( tN, hb, hI, r0, w0, eta, t_eval=t)
        if npmin(h) < 0.0:
            ### Invalid parameters, negative risk:
            return None, None
    else:
        w1, A, phi, tm, hm = HarmonicR_AnalyticPars( tau, w2, hb, h0, r0)
        if (tm < tN) and (hm < 0.0):
            ### Invalid parameters, negative risk:
            return None, None
        h = HarmonicR_Analytic_h( t, tau, w1, A, phi, hb)
        H = HarmonicR_Analytic_H( t, tau, w1, A, phi, hb)
    return h, H
        

if __name__ == "__main__":
    """Plot some examples of different shapes."""
    ax =\
    PlotHarmonicRiskODE(      tau=5, w2=1/2, hb=1, h0=1.5, r0=0.0, color="blue", trange_mult=3, time_unit="")
    PlotHarmonicRiskODE(      tau=5, w2=1,   hb=1, h0=1.5, r0=0.0, color="lightblue", ax=ax, trange_mult=3, time_unit="")
    PlotHarmonicRiskAnalytic( tau=5, w2=1,   hb=1, h0=1.5, r0=0.0, color="lightblue", ax=ax, trange_mult=3, time_unit="")
    PlotHarmonicRiskODE(      tau=5, w2=1/6, hb=1, h0=1.0, r0=-0.5, color="green", ax=ax, trange_mult=3, time_unit="")
    PlotHarmonicRiskAnalytic( tau=5, w2=1/6, hb=1, h0=1.0, r0=-0.5, color="green", ax=ax, trange_mult=3, time_unit="")
    PlotHarmonicRiskODE(      tau=5, w2=1/4, hb=1, h0=0.5, r0=0.0, color="orange", ax=ax, trange_mult=3, time_unit="")
    PlotHarmonicRiskODE(      tau=5, w2=1/4, hb=1, h0=0.5, r0=0.0, color="orange", ax=ax, trange_mult=3, time_unit="")
    yl = ax.get_ylim()
    ax.set_ylim((0,yl[1]))
    ax.get_figure().savefig("Figs/HarmonicRisck1.eps")
    
    ax =\
    PlotHarmonicRiskODE(     tau=5, w2=1/5.5, hb=0.2, h0=0.3, r0=0.2, color="blue", trange_mult=3, time_unit="")
    PlotHarmonicRiskAnalytic(tau=5, w2=1/5.5, hb=0.2, h0=0.3, r0=0.2, color="blue", ax=ax, trange_mult=3, time_unit="")
    PlotHarmonicRiskODE(     tau=5, w2=1/5.5, hb=0.2, h0=0.3, r0=0.8, color="blue", ax=ax, trange_mult=3, time_unit="")
    PlotHarmonicRiskODE(     tau=5, w2=1/8,   hb=2.0, h0=0.3, r0=0.0, color="red", ax=ax, trange_mult=3, time_unit="")
    PlotHarmonicRiskODE(     tau=5, w2=1/8,   hb=0.2, h0=0.3, r0=-0.1, color="green", ax=ax, trange_mult=3, time_unit="")
    yl = ax.get_ylim()
    ax.set_ylim((0,yl[1]))
    ax.get_figure().savefig("Figs/HarmonicRisck2.eps")
    
    fig, ax = subplots(nrows=1, ncols=2)
    t = linspace( 0, 14, num=100)
    h, H = HarmonicR( t, tau=5, w2=1,   hb=1, h0=1.5, r0=0.0, ode=False)
    ax[0].plot( t, h, '-', color="blue")
    ax[1].plot( t, H, '-', color="blue")
    h, H = HarmonicR( t, tau=5, w2=1,   hb=1, h0=1.5, r0=0.0, ode=True)
    ax[0].plot( t, h, '-', color="blue")
    ax[1].plot( t, H, '-', color="blue")
    h, H = HarmonicR( t, tau=5, w2=1/5.5, hb=0.2, h0=0.3, r0=0.8, ode=False)
    ax[0].plot( t, h, '-', color="green")
    ax[1].plot( t, H, '-', color="green")
    h, H = HarmonicR( t, tau=5, w2=1/5.5, hb=0.2, h0=0.3, r0=0.8, ode=True)
    ax[0].plot( t, h, '-', color="green")
    ax[1].plot( t, H, '-', color="green")
    ax[0].set_xlabel(r"$t$ ")
    ax[0].set_ylabel(r"$h(t)$")
    ax[0].grid(which='both', color='grey', linestyle='dotted', linewidth=1)
    ax[1].set_xlabel(r"$t$ ")
    ax[1].set_ylabel(r"$H(t)$")
    ax[1].grid(which='both', color='grey', linestyle='dotted', linewidth=1)




    
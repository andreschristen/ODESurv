#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:19:12 2022

@author: jac
"""

from numpy import exp, log, linspace
from scipy.integrate import solve_ivp
from matplotlib.pylab import subplots, rcParams
rcParams.update({'font.size': 14})

def Logistich( t, la, kappa, h0):
    """Analytic solution for the logistic."""
    return (kappa * h0 * exp(la * t))/(kappa + h0*(exp(la * t)-1.0))

def LogisticH( t, la, kappa, h0):
    return (kappa/la)*log((h0*(exp(la*t)-1) + kappa)/kappa)

def LogisticQ( u, la, kappa, h0):
    """Inverse of cdf, quantile function."""
    return (1/la)*log(1 + (kappa/h0)*(exp(-(la/kappa)*log(1-u)) -1) )

def LogisticRHS( t, h, la, kappa):
    """The rhs for the ODE defining the Logidtics Risk function."""
    return la * h * (1.0 - h/kappa)

def LogisticH_RHS( t, Y, la, kappa):
    """Logistic rhs including the survival function."""
    h, H = Y
    return [la * h * (1.0 - h/kappa), h]

def LogistichODE( t_eval, la, kappa, h0):
    """Return logistic hazard function evaluted at t_eval, but calculated
       using the ODE solver.
    """
    rt = solve_ivp( fun=LogisticRHS, t_span=(t_eval[0],t_eval[-1]), t_eval=t_eval,\
        y0=[h0], method='LSODA', vectorized=True, args=( la, kappa))
    return rt.y[0,:]

def LogisticODE( t_eval, la, kappa, h0):
    """Return logistic hazard and cummulative hazard function evaluted
       at t_eval, but calculated using the ODE solver.
    """
    rt = solve_ivp( fun=LogisticH_RHS, t_span=(t_eval[0],t_eval[-1]), t_eval=t_eval,\
        y0=[h0,0.0], method='LSODA', vectorized=True, args=( la, kappa))
    ###        h        H
    return rt.y[0,:], rt.y[1,:]


if __name__ == "__main__":
    """Plot some examples and comparison using LSODA."""
    fig, ax = subplots(nrows=2,ncols=2)
    
    la = 1.0
    kappa = 10.0
    tN = 10.0/la
    t_eval = linspace( 0.0, tN, num=100)
    ax00 = ax[0,0]
    fig2, ax00 = subplots()
    ### Increasing
    h0 = kappa/10
    rt = solve_ivp( fun=LogisticRHS, t_span=(0, tN), t_eval=t_eval, y0=[h0],\
          method='LSODA', vectorized=True, args=( la, kappa))
    ax00.plot( rt.t, rt.y[0,:], '-', color="orange")
    ### Also plot analytic solution
    ax00.plot( rt.t, Logistich( rt.t, la=la, kappa=kappa, h0=h0), '--', color="black")

    ### Decreasing
    h0 = kappa*2
    rt = solve_ivp( fun=LogisticRHS, t_span=(0, tN), t_eval=t_eval, y0=[h0],\
          method='LSODA', vectorized=True, args=( la, kappa))
    ax00.plot( rt['t'], rt['y'][0,:], '-', color="green")

    ### Constant
    h0 = kappa
    rt = solve_ivp( fun=LogisticRHS, t_span=(0, tN), t_eval=t_eval, y0=[h0],\
          method='LSODA', vectorized=True, args=( la, kappa))
    ax00.plot( rt['t'], rt['y'][0,:], '-', color="red")
    ax00.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
    fig2.tight_layout()
    fig2.savefig("Figs/LogisticRiskhs.eps")
    ax[0,0].set_title(r"$h(t)$")

    ### Same as first example but calculation the cummulative hazard and density function as well
    h0 = kappa/10
    rt = solve_ivp( fun=LogisticH_RHS, t_span=(0, tN), t_eval=t_eval, y0=[h0,0.0],\
          method='LSODA', vectorized=True, args=( la, kappa))
    ax[0,1].plot( rt.t, rt.y[0,:], '-', color="orange")
    ax[1,0].plot( rt.t, rt.y[1,:], '-', color="orange")
    ax[1,1].plot( rt.t, exp(-rt.y[1,:]), '-', color="orange")
    ax[1,0].plot( rt.t, LogisticH(rt.t, la, kappa, h0), '--', color="black")

    h0 = kappa*2
    rt = solve_ivp( fun=LogisticH_RHS, t_span=(0, tN), t_eval=t_eval, y0=[h0,0.0],\
          method='LSODA', vectorized=True, args=( la, kappa))
    ax[0,1].plot( rt.t, rt.y[0,:], '-', color="green") # h 
    ax[1,0].plot( rt.t, rt.y[1,:], '-', color="green") # H
    ax[1,1].plot( rt.t, exp(-LogisticH(rt.t, la, kappa, h0)), '-', color="green")

    h0 = kappa
    rt = solve_ivp( fun=LogisticH_RHS, t_span=(0, tN), t_eval=t_eval, y0=[h0,0.0],\
          method='LSODA', vectorized=True, args=( la, kappa))
    ax[0,1].plot( rt.t, rt.y[0,:], '-', color="red")
    ax[1,0].plot( rt.t, rt.y[1,:], '-', color="red")
    ax[1,1].plot( rt.t, exp(-rt.y[1,:]), '-', color="red")

    #ax[0,1].set_title(r"$h(t)$")
    #ax[1,0].set_title(r"$H(t)$")
    #ax[1,1].set_title(r"$S(t)$")
    ax[1,1].set_xlim((0,2))
    ax[1,1].set_ylim(( 0, 1.0))
    fig.tight_layout()



    
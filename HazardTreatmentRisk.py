#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:19:12 2022

@author: jac
"""

from numpy import exp, log, linspace, array, isnan
from scipy.stats import uniform
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from matplotlib.pylab import subplots


def HazardTreatmentR_RHS( t, Y, la, ka, al, be, de):
    """The Hazard-Treatment risk ODE.
        h'(t)  =  \lambda h(t) \left(1 - \dfrac{h(t)}{\kappa}\right) - \alpha q(t) h(t), & h(0) = h_0 \\
        q'(t) =   \beta q(t) \left( 1- \frac{q(t)}{\kappa} \right)   - \delta q(t) h(t) ,  & \ac q(0) = q_0 \\ 
        H'(t)  =  h(t), & H(0) = 0.
    """
    h, q, H = Y
    if isnan(h):
        raise("h nan.")
    if isnan(q):
        raise("q nan.")
    return array([[ la*h*(1-h/ka) -al*q*h ],
                  [ be*q*(1-q/ka) -de*q*h ],
                  [                     h ]])

def HazardTreatmentR_Jacobian( t, Y, la, ka, al, be, de):
    """The Hazard-Treatment risk ODE.
        h'(t)  =  \lambda h(t) \left(1 - \dfrac{h(t)}{\kappa}\right) - \alpha q(t) h(t), & h(0) = h_0 \\
        q'(t) =  -\delta q(t) h(t) + \beta q(t) \left( 1- \frac{q(t)}{\kappa} \right) ,  & \ac q(0) = q_0 \\ 
        H'(t)  =  h(t), & H(0) = 0.
        
        This defines the Jacobian of the RHS, used by solve_ivp
    """
    h, q, H = Y
    ### partial dev      w.r.t h                w.r.t q         w.r.t H
    return array([[ la*(1-2*h/ka) -al*q,                  -al*h, 0],\
                  [               -de*q,    be*(1-2*q/ka) -de*h, 0],\
                  [                   1,                      0, 0]])

def HazardTreatmentR_ODE( tN, la, ka, al, be, de, h0, q0, t_eval=None, method='LSODA'):
    """Harmonic risk function using the ODE solver.
        teval: time points to return evaluations of h, if None, the solver chooses optimal points. 
    """
    ### Solve using the LSODA solver, note that H(0) = 0
    rt = solve_ivp( fun=HazardTreatmentR_RHS, t_span=(0, tN), t_eval=t_eval, y0=[ h0, q0, 0],\
          method=method, vectorized=True, args=( la, ka, al, be, de),\
              jac=HazardTreatmentR_Jacobian)
    if rt.status != 0:
        print("HazardTreatmentR_ODE: failed: ", rt.status, "\n", la, ka, al, be, de, h0, q0)
        raise("HazardTreatmentR_ODE: failed.")
    for i,h in enumerate(rt.y[0,:]):
        if h < 0:
            rt.y[0,i] = 1e-6 #we do not know what is a small number in general, works fine for the rotterdam data 
    ###      t      h           q          H
    return rt.t, rt.y[0,:], rt.y[1,:], rt.y[2,:] 

def InvH( t, H):
    """Returns the inverse of H. If then this is evaluated in -log(uniform.rvs()),
       this is a simulation of the corresponding risk distribution:
        >>> invH = InvH( t, H)
        >>> sim = invH(-log(uniform.rvs(size=100))) #100 simulations
        
       Uses scipy.interpolate.PchipInterpolator monotonic interpolator and solve.
    """
    
    return PchipInterpolator( H, t)

def PlotRiskDist( t, h, H, color="black", time_unit = "", ax=None):
    """  
    Plot the risk function represented by the risk function h and cumulative
    risk H, using the time points t.

    time_unit = "" #e.g. "yr"
    color (='black')`# for plotting h
    ax # Axes to plot to, if None, create one
    returns ax.
    """
    
    if ax is None:
        fig, ax = subplots(nrows=2,ncols=2, sharex=True)

    invH=InvH( t, H)
    ax[0,0].plot( t, H, '-', color=color)
    ax[0,0].set_title(r"$H(t)$")
    ax[0,0].grid(which='both', color='grey', linestyle='dotted', linewidth=1)

    ax[0,1].plot( t, h, '-', color=color)
    ax[0,1].set_title(r"$h(t)$")
    ax[0,1].grid(which='both', color='grey', linestyle='dotted', linewidth=1)

    ax[1,0].plot( t, exp(-H), '-', color=color)
    ax[1,0].set_title(r"$S(t)$")
    ax[1,0].set_xlabel(r"$t$")
    u = linspace( 0, 1, num=200)
    ax[1,0].plot( invH(-log(u)), u, '--', color=color)
    ax[1,0].grid(which='both', color='grey', linestyle='dotted', linewidth=1)
    
    ax[1,1].plot( t, h*exp(-H), '-', color=color)
    ax[1,1].set_title(r"$f(t)$")
    ax[1,1].set_xlabel(r"$t$")
    sim = invH(-log(uniform.rvs(size=1000)))
    ax[1,1].hist( sim, bins=20, density=True)
    ax[1,1].grid(which='both', color='grey', linestyle='dotted', linewidth=1)
    
    return ax


def PlotHazardTreatmentODE( la, ka, al, be, de, h0, q0, tN=3,\
                         time_unit="", color="orange", ax=None):
    """  
    Plot the harmonic risk function:

    time_unit = "" #e.g. "yr"
    color (='black')`# for plotting h
    ax # Axes to plot to, if None, create one
    returns ax.
    """
    
    if ax is None:
        fig, ax = subplots(nrows=1,ncols=1)

    t, h, q, H = HazardTreatmentR_ODE( tN, la, ka, al, be, de, h0, q0)
    ax.plot( t, h, '-', color=color)
    ax.plot( t, q, '--', color=color)
    ax.set_xlabel(r"$t$ "+time_unit)
    ax.set_ylabel(r"$h(t), q(t)$")
    ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
    ax.axhline(y=ka, color="black")
    """
    ax[1].plot( h, q, '-', color=color)
    ax[1].plot( ka, 0, 'o', color=color)
    ax[1].plot( 0, ga, 'o', color=color)
    ax[1].set_xlabel(r"$h$ "+time_unit)
    ax[1].set_ylabel(r"$q$")
    ax[1].grid(which='both', color='grey', linestyle='dotted', linewidth=1)
    """
    return ax

        

if __name__ == "__main__":
    ### la, ka, al, be, de
    t, h, q, H = HazardTreatmentR_ODE(tN=10, la=1, ka=1, al=2, be=1, de=2, h0=3/2, q0=1/2)
    PlotRiskDist( t, h, H)
    ax =\
    PlotHazardTreatmentODE(la=1, ka=1, al=2, be=1, de=2, h0=0.32, q0=0.34, tN=10, color="blue")
    PlotHazardTreatmentODE(la=1, ka=1, al=2, be=1, de=2, h0=1/3, q0=1/3, tN=10, color="orange", ax=ax)
    PlotHazardTreatmentODE(la=1, ka=1, al=2, be=1, de=2, h0=3/2, q0=1/2, tN=10, color="red", ax=ax)
    PlotHazardTreatmentODE(la=1, ka=1, al=2, be=1, de=2, h0=1/2, q0=0.6, tN=10, color="green", ax=ax)
    yl = ax.get_ylim()
    ax.set_ylim((0,yl[1]))
    ax.get_figure().savefig("HazardTreatmentRisck1.pdf")
    




    
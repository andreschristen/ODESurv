#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 11:33:08 2021

@author: jac
"""

from numpy import zeros, quantile, append, array, isnan, histogram, where, mean
from matplotlib.pylab import subplots


def PlotFuncAtMC( f, x, sample, ax=None, color='blue', histtype='bar', remove_nans=True,
                 **kwargs):
    """Plot a histogram of $f_\theta(x)$
        for every $\theta$ in MC sample.
    """
    if ax == None:
        fig, ax = subplots()
    
    T = sample.shape[0] # sample size
    solns = zeros(T)
    for i in range(T):
        if remove_nans:
            pars=sample[i,where(1-isnan(sample[i,:]))]
        else:
            pars=sample[i,:]
        solns[i] = f( x, pars=pars)
    ax.hist( solns, color=color, histtype=histtype, **kwargs )
    return ax, solns

def CreateSolnsf( f, x, sample, remove_nans=True):
    """Create the solns array for using in PlotSolnsMCq or PlotSolnsMCh:
        to plot the evolution of $f_\theta(x)$ for every $\theta$ in MC sample.
        f: f( x, theta), theta is a vector, to pass the parameters
        sample: array of T x p, T MC sample size, p numbre of parameters.
        q: quantiles below 0.5, these will be repeated with 1-q to make quantiles ranges.
        fill: If True, fill the quantile rages with color, otherwiaw, draw color lines for each quantile.
        med_col: color to plot the median.
        
        returns solns
        
        Do PlotSolnsMCq( x, solns) or PlotSolnsMCh( x, solns) for plotting.
    """

    T = sample.shape[0] # sample size
    solns = zeros((T,x.size))
    for i in range(T):
        if remove_nans:
            if any(isnan(sample[i,:])):
                   continue
        solns[i,:] = f( x, theta=sample[i,:])

    return solns


def PlotSolnsMCq( x, solns, q=[0.10,0.25], flip=False, remove_nans=True,\
                 fill=True, color='blue', med_col='red', mean_col=None, label="", ax=None):
    """Plot evolution of quantiles q of $f_\theta(x)$
        for every $\theta$ in MC sample.
       solns: is a $T \times m$ array, with the $T$ simulations
         $f_{\theta^{(t)}}(x); T=0,1,\ldots,m-1$.
       q: quantiles below 0.5, these will be repeated with 1-q to make quantiles ranges.
       fill: If True, fill the quantile rages with color, otherwiaw, draw color lines for each quantile.
       med_col: color to plot the median.
       mean_col: color to plot the mean of samples, if None, it is not plotted (default).
       
       if flip, use x in the y axis.
    """

    if ax == None:
        fig, ax = subplots()
    
    q = array(q)
    q = append(append( q, [0.5]), 1-q)
    n = q.size
    
    m = x.size #Number of points in x
    T, m1 = solns.shape #sample size and number of points in x
    if not(m == m1):
        print("Size of x (%d) not equal to the number of columns (%d) in solns" %\
              (m,m1))
        return
    
    quan = zeros((n,m))
    if mean_col is not None:
        mn = zeros(m)
    ### Calculate the quantiles
    for j in range(m):
        tmp = solns[:,j]
        if remove_nans:
            tmp = tmp[~isnan(tmp)]
        if tmp.size > 100:
            quan[:,j] = quantile( tmp, q)
        else:
            quan[:,j] = [None]*n
        if mean_col is not None:
            mn[j] = mean(tmp)
    ###Plot the quantiles:
    k = n//2
    if fill:
        for i in range(k):
            if flip:
                ax.fill_betweenx( x, quan[i,:], quan[-1-i,:], color=color, alpha=0.5/k)
            else:
                ax.fill_between( x, quan[i,:], quan[-1-i,:], color=color, alpha=0.5/k)
    else:
        for i in range(n):
            if flip:
                ax.plot( quan[i,:], x, '--', color=color, linewidth=0.5)
            else:
                ax.plot( x, quan[i,:], '--', color=color, linewidth=0.5)
    ###Plot the median and the mean
    if flip:
        ax.plot( quan[k,:], x, '-', color=med_col, linewidth=1.5, label=label)
        if mean_col is not None:
            ax.plot( mn, x, '-', color=mean_col, linewidth=1.5, label=label)
    else:
        ax.plot( x, quan[k,:], '-', color=med_col, linewidth=1.5)
        if mean_col is not None:
            ax.plot( x, mn, '-', color=mean_col, linewidth=1.5)

    return ax, quan


def PlotSolnsMCh( x, solns, bins=20, flip=False, remove_nans=True,\
                 q=[0.10,0.25], fill=False, color='black', med_col='black', ax=None):
    """Plot evolution of vertical histograms of $f_\theta(x)$
        for every $\theta$ in MC sample.
       solns: is a $T \times m$ array, with the $T$ simulations
       $f_{\theta^{(t)}}(x); T=0,1,\ldots,m-1$, where $x$ is an equally space
       size $m$ array.
       bins: Number of bins to use.
       
       If flip, use x in the y axis.
       
       If q is not None, the quantiles bands are be added also, using PlotSolnsMCq:
       q=[0.10,0.25], fill=False, color='black', med_col='black', are passed to PlotSolnsMCq.     
       
       ax: Axes to plot, if None, create one.
       
       Returns: ax 
    """

    if ax == None:
        fig, ax = subplots()
        
    m = x.size #Number of points in x
    ### we assume x represent the center points of the intervals
    Dx = x[1]-x[0] # are eqally space and this is the size of the intervals
    T, m1 = solns.shape #sample size and number of points in x
    if not(m == m1):
        print("Size of x (%d) not equal to the number of columns (%d) in solns" %\
              (m,m1))
        return
    
    ### Calculate and plot the histograms
    for j in range(m):
        tmp = solns[:,j]
        if remove_nans:
            tmp = tmp[~isnan(tmp)]
        hs, bns = histogram( tmp, bins=bins, density=True)
        if any(isnan(hs)): #Nan values may still oaccur if all values in the sample are equal
            hs = array([1.0]) # in that case, plot 1.0
            bns = array([solns[0,j]]) # at the unique value
        hs_max = hs.max()
        for i in range(bns.size-1):
            if flip:
                ax.barh( x[j], height=Dx, align='center',\
                       width=bns[i+1]-bns[i], left=bns[i],\
                           color="black", alpha=hs[i]/hs_max)
            else:
                ax.bar( x[j], width=Dx, align='center',\
                       height=bns[i+1]-bns[i], bottom=bns[i],\
                           color="black", alpha=hs[i]/hs_max)
    if q is not None:
        PlotSolnsMCq( x, solns, flip=flip, q=q, fill=fill, color=color, med_col=med_col, ax=ax)
    
    return ax





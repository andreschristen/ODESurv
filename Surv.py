#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:56:13 2022

@author: jac


class Surv to analyse survuival data, choosing the survival funciton.

For the examples in Christen and Rubio (2023), go to if name == "__main__" below

"""

from numpy import exp, log, linspace, array, arange, ceil, zeros, diag, ones, cumsum, isnan
from numpy import sum as sum_np
from scipy.stats import uniform, gamma, norm
from matplotlib.pylab import subplots, close

from pytwalk import pytwalk
from plotfrozen import PlotFrozenDist
#from PlotCorner import PlotCorner
from pyplotfuncmc import PlotSolnsMCq, CreateSolnsf

from LogisticRisk import Logistich, LogisticH

from HarmonicRisk import HarmonicR_AnalyticPars, HarmonicR_Analytic_h, HarmonicR_Analytic_H

from HazardTreatmentRisk import HazardTreatmentR_ODE


class Surv:

    """
        Generic class for Bayesian analysis of survival a, b, al, betadata using the t-walk.
        This class has as an example the exponential distribution.
        
        surv_times: array with ordered survival data, from time 0
        censor: array with 1 if NOT censor, 0 if censored obs.
        p: Number of parameters in the underlying model (e.g.Weibull, p=2).
        
        Covariates: if None, then there are no covariates and there
         are only p parameters in the model.
         Otherwise is tuple with two entries (is_reg_par, X)
         
         is_reg_par: boolean list of size p.  If True, the parameter is 
         regression parameter, maping (with X) to n values of that parameter.
         If False, then is not regression par and it maps to one value.
         
         X: design matrix
         Example:
         p=3 (e.g. Logistic, with parameters la, kappa and h0)
         is_reg_par = [ True, True, False] #la, kappa are reg. parameters
         X is an array of size ( n + n + 1) x q
         For example, there are 2 covariates (age and sex), the regression
         for la_i = l(th_0 + th_1 age_i + th_2 sex_i), and for
          kappa_i = l(th_3 + th_4 age_i + th_5 sex_i) . For
               h0 = l(th_6)
         If size is 1, l is the identity.
         The design matrix is packed such that
         pars = X @ th, th = (th_0, ... , th_6)'
         pars = (la_0, ... , la_{n-1}, kappa_0, ... ,kappa_{n-1}, h0)'
         q = 7 in this case, n is the sample size.
         If Covariates = None, then internally
         pars = (la, kappa, h0) and X is the 3x3 identity and l(x)=x
    """
    def __init__(self, surv_times, censor, p, Covariates=None):
        self.n = surv_times.size
        ### OJO ESTO ES SOLO PARA QUE FUNCIONE LA EVALUACION DE LA ODE
        ### Para que no haya datos repetidos
        ### multiplica por U(0.99,0.01)
        surv_times *= uniform.rvs(loc=0.99, scale=0.01, size=self.n)
        indx = surv_times.argsort()
        self.surv_times = surv_times[indx]
        self.censor = censor[indx].astype(int)
        self.log_surv_times = log(self.surv_times)
        self.obs_surv_times = self.surv_times[self.censor == 1]
        self.m = self.obs_surv_times.size
        self.log_obs_surv_times = log(self.obs_surv_times)
        self.censor_surv_times = self.surv_times[self.censor == 0]
        self.log_censor_surv_times = log(self.censor_surv_times)

        self.p = p ### Number of parameters in the underlying model (e.g.Weibull=2)
        if Covariates is None:
            self.is_reg_par = [False]*self.p
            ### No covariates, X is the p X p identity
            self.X = diag(ones(self.p))
            ### The par_map is simply one for each parameter
            ### For x, the parameter vector, self.par_map indicates the
            ### sequential indices for each parameter, see self.link 
            self.par_map = cumsum(array([0] + [1]*self.p))
            ### default supp method
            ### Only positive parameters
            self.positive = arange(self.p)
        else:
            self.is_reg_par = Covariates[0]
            self.X = Covariates[1]
            self.par_map = [0]
            for irp in self.is_reg_par:
                if irp:
                    self.par_map += [self.n] #Regresion parameter
                else:
                    self.par_map += [1]
            ### default, al real parameters
            self.positive = array([]) 
            self.par_map = cumsum(array(self.par_map)) #single parameter
        self.q = self.X.shape[1] # Total number of parameters
        ### Default par names
        self.parname = [r"$\theta_{%d}$" % (j) for j in range(self.q)] #with no regresion
        
        self.prior =  [gamma( 2, scale=1/1)] #list of priors, logpdf and pdf are needed
            
        ### parameter names of underlying model, this should be of size p
        self.u_parname = [r"$\lambda$"] #example, exponential

        self.twalk = pytwalk(  n=self.q, U=self.Energy, Supp=self.Supp)
    
    def h(self, surv_times, pars):
        """Hazard function. It needs to be vectorized on surv_times and
           on pars, if using regresion. pars is a list of size self.p,
           each entry may be an arry of size self.n (reg. parameter) or
           a number (no reg. parameter).
        """
        return pars[0] #lambda, constant hazard function for exponential

    def H(self, surv_times, pars):
        """Cummulative hazard function. It needs to be vectorized on surv_times and
           on pars, if using regresion. pars is a list of size self.p,
           each entry may be an arry of size self.n (rea, b, al, betag. parameter) or
           a number (no reg. parameter).
        """
        return pars[0]*surv_times #H of exponential
    
    def actual_link_fuction(self, z):
        return z
    
    def l(self, z):
        """Link function, z is an array."""
        if z.size == 1:
            return z[0] # no reg. par., identity
        else:
            return self.actual_link_fuction(z) #link function here, for regresion parameters

    def unpack(self, th):
        reg = self.X @ th
        ### Unpack the result to map the corresponding parameters
        ### according to the indices in par_map
        return [self.l(reg[self.par_map[i]:self.par_map[i+1]]) for i in range(self.p)]
        
    def LogPrior( self, th):
        return sum([self.prior[i].logpdf(th[i]) for i in range(self.q)])

    def LoghLikelihood(self, th):
        """Log hazard function. It takes args and returns the log hazard
           evaluated on the sum_np( log(self.h(self.obs_surv_times, *args)))
           but the log(h) may be woractual_link_fuctionk out algebraically for numerica stability."""
        largs = self.unpack(th)
        return sum_np( self.censor*log(self.h(self.surv_times, largs)))

    def LogLikelihood( self, th):
        """Log likelihood."""
        largs = self.unpack(th)
        return self.LoghLikelihood(largs) + sum_np(-1.0*self.H( self.surv_times, largs))

    """
    def LogLikelihoodODE( self, *args):
        #Log likelohood for using a ODE.
        largs = self.link(*args)
        h, H = self.ODE( self.surv_times, *largs)
        ###                 h y[1,:]                                 H
        return sum_np(log(h)*self.censor) + sum_np(-1.0*H)
    """
    
    def SimInit(self):
        return array([self.prior[i].rvs() for i in range(self.q)]) #array of seize 1
    
    def Supp( self, th):
        return all(th[self.positive] > 0.0)
    
    def Energy( self, th):
        return -1.0*( self.LogLikelihood(th) + self.LogPrior(th))    

    def RunMCMC( self, T, burn_in=0):
        self.twalk.Run( T=T, x0=self.SimInit(), xp0=self.SimInit())
        if burn_in > 0:
            self.AnaMCMC(burn_in=burn_in)
    
    def AnaMCMC( self, burn_in):
        self.iat = int(ceil(self.twalk.Ana(burn_in=burn_in)))
        self.start = burn_in
    
    def StripParName(self, k):
        """Return parameter name k removing $ and \.  Useful for filenames."""
        return self.parname[k].replace('$','').replace('\\','')
    
    def PlotLogPost( self, ax=None):
        if ax is None:
            fig, ax = subplots()
        ax.plot( arange( self.start, self.twalk.Output.shape[0]), -1*self.twalk.Output[self.start:,-1], '-')
        ax.set_ylabel("LogPost")
        ax.set_xlabel("Iteration")
        return ax

    def PlotPriors(self):
        fig, ax = subplots( nrows=2, ncols=1 + self.n_pars//2)
        ax = ax.flatten()
        for i in range(self.q):
            PlotFrozenDist( self.prior[i], color="green", ax=ax[i+1] )
            ax[i+1].set_xlabel(self.parname[i])
        fig.tight_layout()
    
    def PlotPost( self, par, ax=None, **kwargs):
        if ax is None:
            fig, ax = subplots()
        ax.hist(self.twalk.Output[self.start:,par], density=True, **kwargs)
        ax.set_xlabel(self.parname[par])
        ax.set_ylabel("Density")
        xl = ax.get_xlim()
        x = linspace( xl[0], xl[1], num=100)
        ax.plot( x, self.prior[par].pdf(x), '-', color="green")
        ax.set_xlim(xl)
        return ax

    def PlotAll( self, f_nam_base=None, ext=".png", **kwargs):
        """Plot all the posterior parameters with the prior in one single panel.
           If f_nam_base not None, plot them in separate figures and svae them,
           using f_nam_base + parname + ext.
           
           returns the last fig and ax used.
        """
        if f_nam_base is None:
            fig, ax = subplots( nrows=2, ncols=1 + self.q//2)
            ax = ax.flatten()
            single = True
            self.PlotLogPost(ax=ax[0])
        else:
            single = False
            ax = self.PlotLogPost()
            fig = ax.get_figure()
            fig.savefig(f_nam_base + "LP" + ext)
        
        for i in range(self.q):
            if single:
                self.PlotPost( par=i, ax=ax[i+1], **kwargs)
            else:
                ax = self.PlotPost( par=i, **kwargs)
                fig = ax.get_figure() 
                fig.savefig( f_nam_base + self.StripParName(i) + ext )
        if single:
            fig.tight_layout()
        return fig, ax

    def PlotPredRisk(self, x0=0.0, x1=None, mult=1.15, ssize=500, xnum=100, surv=False, ax=None, **kwargs):
        """Plot the predictive population risk (or survival) funcions, with trajectories from realizations of
           the MCMC, from x0 (=0) to x1 (if None, x1=mult * max(surv_times)).
           kwargs is passed to PlotFuncEvolveMCq.
           Only works for no regresion 
        """
        if any(self.is_reg_par):
            print("PlotPredRisk only meake sense with no regresion.")
            return None, None, None
        if x1 is None:
            x1=mult * self.surv_times[-1]
        self.x = linspace( x0, x1, num=xnum)
        T = self.twalk.Output.shape[0]
        if (T-self.start) // self.iat < ssize:
            print("MCMC sample size not big enough. Effective sample size %d (T=%d, burn_in=%d, IAT=%d)." %\
                  ((T-self.start) // self.iat, T, self.start, self.iat))
            return None, None, None
        tau = (T-self.start) // ssize
        def f( x, theta):
            largs = self.l(theta)
            if surv:
                return exp(-self.H( x, largs))
            else:
                return self.h( x, largs)
        solns = CreateSolnsf(f, self.x, self.twalk.Output[self.start::tau,:-1])
        ax, quan = PlotSolnsMCq( self.x, solns, ax=ax, **kwargs)
        ax.set_xlabel(r"$t$")
        if surv:
            ax.set_ylabel(r"$S(t)$")
        else:
            ax.set_ylabel(r"$h(t)$")
        return ax, quan, solns        


class SurvWeibull(Surv):

    def __init__(self, surv_times, censor):
        super().__init__(surv_times, censor, p=2)
        ### Priors
        self.parname = [r"$\lambda$", r"$\kappa$"]
        self.laprior = gamma( 2, scale=1/1)
        self.kappaprior = gamma( 2, scale=1/1)
        self.prior = [ self.laprior, self.kappaprior] #list of priors, logpdf and pdf are needed
    
    def h( self, surv_times, args):
        """Hazard function. It needs to be vectorized on surv_times."""
        la, kappa = args
        return (kappa/la)*(surv_times/la)**(kappa-1.0)

    def H( self, surv_times, args):
        """Cummulative hazard function. It needs to be vectorized on surv_times."""
        la, kappa = args
        return (surv_times/la)**kappa
    
    def LogPrior( self,  la, kappa):
        return self.laprior.logpdf(la) + self.kappaprior.logpdf(kappa)

    def SimInit(self):
        return array([self.laprior.rvs(), self.kappaprior.rvs()]) #array of seize 1

    def LoghLikelihood(self, *args):
        """Log hazard function. It takes args and retexpon.rvs(size=self.n_pars)urns the log hazard
           evaluated on the sum_np( log(se        self.iat = 1
lf.h(self.obs_surv_times, *args)))
           but the log(h) may be work out algebraically for numerica stability."""
        la, kappa = args
        log_la = log(la)
        return sum_np( log(kappa) - log_la + (kappa-1)*(self.log_obs_surv_times - log_la) )



class SurvLogistic(Surv):

    def __init__(self, surv_times, censor, Covariates=None):
        super().__init__(surv_times, censor, p=3, Covariates=Covariates)
        if Covariates is None:
            self.parname = [r"$\lambda$", r"$\kappa$", r"$h_0$"]
        self.laprior = gamma( 2, scale=2)
        self.kappaprior = gamma( 2, scale=2)
        self.h0prior = gamma( 2, scale=2)
        self.prior = [ self.laprior, self.kappaprior, self.h0prior] #list of priors, logpdf and pdf are needed
    
    def h( self, surv_times, args):
        """Hazard function. It needs to be vectorized on surv_times."""
        la, kappa, h0 = args
        return Logistich( surv_times, la, kappa, h0)

    def H( self, surv_times, args):
        """Cummulative hazard function. It needs to be vectorized on surv_times."""
        la, kappa, h0 = args
        return LogisticH( surv_times, la, kappa, h0)

    def LoghLikelihood(self, args):
        """Log hazard function. It takes args and returns the log hazard
           evaluated on the sum_np( log(self.h(self.obs_surv_times, *args)))
           but the log(h) may be work out algebraically for numerica stability."""
        la, kappa, h0 = args
        la_t = la*self.surv_times
        return sum_np( self.censor*(log(kappa) + log(h0) + la_t -\
                      log(kappa + h0*(exp(la_t) - 1.0))) )


class SurvHarmonic(Surv):

    def __init__(self, surv_times, censor, Covariates=None):
        super().__init__(surv_times, censor, p=5, Covariates=Covariates)
        if Covariates is None:
            self.parname = [r"$\tau$", r"$w_2$", r"$h_b$", r"$h_0$", r"$r_0$"]
        self.positive = array([0, 1, 2, 3]) # all positive, besides r_0
        self.prior = [ gamma( 2, scale=2)]*5 #list of priors, logpdf and pdf are needed
        self.tM = float('inf') #max(surv_times) #default maximum time to use
        self.warning_par = 0
    
    def ParRestrictions( self, th):
        """Default parameter restrictions.  All positive besides r0.
           This may be changed by the user.
        """
        return all(th[self.positive] > 0.0)
    
    def Supp( self, th):
        tau, w2, hb, h0, r0 = th
        if not(self.ParRestrictions(th)):
            return False
        w1, A, phi, tm, hm = HarmonicR_AnalyticPars( tau, w2, hb, h0, r0)
        ### If argmin_{t \in R}h(t) = tm < maximum time self.tM and
        ### min_{t \in R}h(t)= h(tm)=hm <0, then h becomes negative.
        ### Or if h(self.tM) < 0, then parameters out of support.
        if ((tm < self.tM) and (hm < 0.0)) or\
            (HarmonicR_Analytic_h( self.tM, tau, w1, A, phi, hb) < 0):
            ### Invalid parameters, negative risk:
            return False
        ### Calculate here the loglikelihood:
        loghlikelihood = sum_np( log(HarmonicR_Analytic_h(self.obs_surv_times, tau, w1, A, phi, hb)))
        self.loglikelihood = loghlikelihood + sum_np(-1.0*HarmonicR_Analytic_H( self.surv_times, tau, w1, A, phi, hb))
        return True
    
    def LogLikelihood(self, args):
        """args is ignored and the last call to Supp is assumed as the
           desired args to evaluate the log likelihood, as it is the case
           in the twalk.
        """
        return self.loglikelihood

    def RunMCMC( self, T, burn_in=0):
        """Overloaded, we need to check if pars in support."""
        """
        x0=self.SimInit()
        while not(self.Supp(x0)):
            x0=self.SimInit()
        xp0=self.SimInit()
        while not(self.Supp(xp0)):
            xp0=self.SimInit()
        """
        x0=array([ 0.72641173,  0.02170349,  0.08183277,  2.02720063, -2.39099078])
        xp0=array([ 0.90417621,  0.02392637,  0.092748  ,  1.7843021 , -1.97522506])
        self.twalk.Run( T=T, x0=x0, xp0=xp0)
        if burn_in > 0:
            self.AnaMCMC(burn_in=burn_in)
    
    def h( self, surv_times, args):
        """Hazard function. It will be used on its own when simulating from the
           posterior risk in PlotPredRisk."""
        if self.warning_par == 0:
            print("SurvHarmonic: h: Warning, parameters may not be in the supp.")
            self.warning_par = 1
        else:
            self.warning_par +=1
        tau, w2, hb, h0, r0 = args
        w1, A, phi, tm, hm = HarmonicR_AnalyticPars( tau, w2, hb, h0, r0)
        return HarmonicR_Analytic_h( surv_times, tau, w1, A, phi, hb)

    def H( self, surv_times, args):
        """Cummulative hazard function. Not expected to be used."""
        if self.warning_par == 0:
            print("SurvHarmonic: H: Warning, parameters may not be in the supp.")
            self.warning_par = 1
        else:
            self.warning_par +=1
        tau, w2, hb, h0, r0 = args
        w1, A, phi, tm, hm = HarmonicR_AnalyticPars( tau, w2, hb, h0, r0)
        return HarmonicR_Analytic_H( surv_times, tau, w1, A, phi, hb)

    def LoghLikelihood(self, args):
        """Log hazard function. It takes args and returns the log hazard
           evaluated on the sum_np( log(self.h(self.obs_surv_times, *args))).
           During the MCMC (t-walk), this calulation is done inside supp,
           and only if the parameters are in the support.
           Not expected to be used.
        """
        if self.warning_par == 0:
            print("SurvHarmonic: LoghLikelihood: Warning, parameters may not be in the supp.")
            self.warning_par = 1
        else:
            self.warning_par +=1
        tau, w2, hb, h0, r0 = args
        w1, A, phi, tm, hm = HarmonicR_AnalyticPars( tau, w2, hb, h0, r0)
        return sum_np( log(HarmonicR_Analytic_h(self.obs_surv_times, tau, w1, A, phi, hb)))


class SurvHazardTreatment(Surv):

    def __init__(self, surv_times, censor, h0, q0, Covariates=None):
        super().__init__(surv_times, censor, p=4, Covariates=Covariates)
        self.h0 = h0
        self.q0 = q0
        if Covariates is None: #la, ka, al, be, de
            self.parname = [r"$\lambda$", r"$\kappa$", r"$\alpha$", r"$\beta$"]
        self.positive = array([0, 1, 2, 3]) # all positive
        self.prior = [ gamma( 2, scale=2)]*4 #list of priors, logpdf and pdf are needed
        self.tM = max(surv_times) #default maximum time to use
        self.warning_par = 0
        self.twalk.par_names = self.parname

    def ParRestrictions( self, th):
        """Default parameter restrictions.  All positive."""
        return all(th[self.positive] > 0.0)
    
    def Supp( self, th):
        if not(self.ParRestrictions(th)):
            return False
        
        """Return the log likelihood.  \gamma = \kappa"""
        la, ka, al, be = th
        ### \delta = \alpha
        t, h, q, H = HazardTreatmentR_ODE( tN=self.tM, t_eval=self.surv_times, \
                    la=la, ka=ka, al=al, be=be, de=al, h0=self.h0, q0=self.q0)
        ### Calculate here the loglikelihood:self.censor*
        self.loghlikelihood = sum_np( self.censor*log(h))
        loglikelihood = self.loghlikelihood + sum_np(-1.0*H)
        if isnan(loglikelihood):
            print("loglikelihood NaN", self.loghlikelihood, sum_np(-1.0*H))
            return False
        self.loglikelihood = loglikelihood
        return True
    
    def LogLikelihood(self, th):
        return self.loglikelihood

    def RunMCMC( self, T, burn_in=0):
        """Overloaded, we need to check if pars in support."""
        """
        x0=self.SimInit()
        while not(self.Supp(x0)):
            x0=self.SimInit()
        xp0=self.SimInit()
        while not(self.Supp(xp0)):
            xp0=self.SimInit()
        """
        x0 = array([1.69121619, 0.10452575, 6.87585355, 5.08031715])
        xp0 =array([1.52542794, 0.10601504, 6.65617124, 4.8078948 ])
        self.twalk.Run( T=T, x0=x0, xp0=xp0)
        if burn_in > 0:
            self.AnaMCMC(burn_in=burn_in)
    
    def h( self, surv_times, theta):
        """Hazard function. It will be used on its own when simulating from the
           posterior risk in PlotPredRisk."""
        la, ka, al, be = theta
        t, h, q, H = HazardTreatmentR_ODE( tN=self.tM, t_eval=surv_times, \
                    la=la, ka=ka, al=al, be=be, de=al, h0=self.h0, q0=self.q0)
        return h

    def q_SV( self, surv_times, theta):
        """q State Variable function (treatment).
           It will be used on its own when simulating from the posterior
           risk in PlotPredRisk below.
           """
        la, ka, al, be = theta
        t, h, q, H = HazardTreatmentR_ODE( tN=self.tM, t_eval=surv_times, \
                    la=la, ka=ka, al=al, be=be, de=al, h0=self.h0, q0=self.q0)
        return q


    def H( self, surv_times, theta, tN=None):
        """Cummulative hazard function. Not expected to be used."""
        la, ka, al, be = theta
        if tN is None:
            tN=self.tN
        t, h, q, H = HazardTreatmentR_ODE( tN=tN, t_eval=surv_times, \
                    la=la, ka=ka, al=al, be=be, de=al, h0=self.h0, q0=self.q0)
        return H

    def LoghLikelihood(self, args):
        """Log hazard function. It takes args and returns the log hazard
           evaluated on the sum_np( log(self.h(self.obs_surv_times, *args))).
           During the MCMC (t-walk), this calulation is done inside supp,
           and only if the parameters are in the support.
           Not expected to be used.
        """
        la, ka, al, be = args
        t, h, q, H = HazardTreatmentR_ODE( tN=self.tM, t_eval=self.obs_surv_times, \
                    la=la, ka=ka, al=al, be=be, de=al, h0=self.h0, q0=self.q0)
        return sum_np( log(h) )

    def PlotPredRisk(self, x0=0.0, x1=None, mult=1.0, ssize=500, xnum=100, surv=False, ax=None, **kwargs):
        """Plot the predictive risk andq (treatment) with trajectories from realizations of
           the MCMC, from x0 (=0) to x1 (if None, x1=mult * max(surv_times)).
            
        """
        if any(self.is_reg_par):
            print("PlotPredRisk only meake sense with no regresion.")
            return None, None, None
        if x1 is None:
            x1=mult * self.surv_times[-1]
        self.x = linspace( x0, x1, num=xnum)
        T = self.twalk.Output.shape[0]
        if (T-self.start) // self.iat < ssize:
            print("MCMC sample size not big enough. Effective sample size %d (T=%d, burn_in=%d, IAT=%d)." %\
                  ((T-self.start) // self.iat, T, self.start, self.iat))
            return None, None, None
        tau = (T-self.start) // ssize
        if surv:
            def f( x, theta):
                return exp(-self.H( x, theta, tN=x1))
            solns = CreateSolnsf( f, self.x, self.twalk.Output[self.start::tau,:-1])
            ax, quan = PlotSolnsMCq( self.x, solns,\
                q=[0.10], fill=True, color='blue', med_col='red', ax=ax, **kwargs)
            ax.set_ylabel(r"$S(t)$")
        else:
            solns = CreateSolnsf(self.q_SV, self.x, self.twalk.Output[self.start::tau,:-1])
            ax, quan = PlotSolnsMCq( self.x, solns,\
                q=[0.10], fill=False, color='orange', med_col='orange', ax=ax, **kwargs)
            solns = CreateSolnsf(self.h, self.x, self.twalk.Output[self.start::tau,:-1])
            ax, quan = PlotSolnsMCq( self.x, solns,\
                q=[0.10], fill=True, color='blue', med_col='red', ax=ax, **kwargs)
            ax.set_ylabel(r"$h(t), q(t)$")
        ax.set_xlabel(r"$t$")
        return ax, quan, solns        

def SimStudy( su_i, true_pars, N, p_censor):
    """Simulation study usign su_i:
        su_i: a Surv instance.
        true_pars: true parameters.
        N: sample size
        p_censor: probability of censorship.
    """
    


def LeukReg( leukcov ):
    """Define a regresion for the Leuk data:
        time	status	sex	age	tpi
    """
    leukcov = leukcov[:500,:]
    surv_times = leukcov[:,0]
    censor = leukcov[:,1]
    sex = leukcov[:,2]
    age = leukcov[:,3]
    tpi = leukcov[:,4]
    
    ### Regresio on la
    n = surv_times.size
    ### Number of parameters, 4 regresion on la cnst sex age tpi
    ### plus one par for kappa and one par for h0
    q = 4+1+1 
    X = zeros((n + 1 + 1, q))
    X[0:n,0] = ones(n)
    X[0:n,1] = sex
    X[0:n,2] = age
    X[0:n,3] = tpi
    X[n,4] = 1
    X[n+1,5] = 1
    is_reg_par = [True,False,False]
    leuk_logistic_cov = SurvLogistic( \
            surv_times = surv_times,\
            censor = censor,\
            Covariates = ( is_reg_par, X))
    ### Default exp link funciton
    #leuk_logistic_cov.actual_link_fuction = lambda z: exp(z) #default
    ### Priors for regresion pàrameters
    leuk_logistic_cov.prior = [norm for i in range(4)]
    ### Priors for kappa and h0
    leuk_logistic_cov.prior += [gamma( 2, scale=1/1), gamma( 2, scale=1/1)]
    leuk_logistic_cov.positive = array([4,5])
    leuk_logistic_cov.parname[4] = r"$\kappa$"
    leuk_logistic_cov.parname[5] = r"$h_0$"
    return leuk_logistic_cov 



if __name__ == "__main__":

    from numpy import loadtxt
    run_lung = False # Run logistic hazard with lung data, NOT IN THE PAPER
    run_leuk_cov = False # Run logistic hazard with lung data, with coviariate NOT WORKING

    run_leuk = True # Run logistic hazard with lukemia data, Example 1
    if not(run_leuk):
        ax_pred_log_h = None
        ax_pred_log_S = None
    run_leuk_har = False # Run harmonic hazard with leukemia data, NOT IN TEH PAPER
    
    run_ht = True # Run Hazard Treatment hazard with Rotterdam data
    
    fig_fmt = ".eps"
    
    if run_lung:
        lung = loadtxt( "lung.csv", delimiter=" ", skiprows=1)
    
        lung_logistic = SurvLogistic( surv_times = lung[:,1], censor = lung[:,2].astype(int))
        lung_logistic.h0prior = gamma( 2, scale=1/4) #Change the default prior for h0
        lung_logistic.RunMCMC( T=500000, burn_in=10000)
        lung_logistic.PlotAll()
        ax, quan, solns = lung_logistic.PlotPredRisk()
    
        #lung_weibull = SurvWeibull( surv_times = lung[:,0], censor = lung[:,2].astype(int))
        #lung_weibull.RunMCMC( T=50000, start=1000)
        #lung_weibull.PlotAll()
    if run_leuk_cov:
        leukcov = loadtxt( "LeukSurvCov.csv", delimiter=",", skiprows=1)
        leuk_logistic_cov = LeukReg(leukcov)
        leuk_logistic_cov.RunMCMC( T=50000, start=1000 )
        leuk_logistic_cov.PlotAll()


    if run_leuk:
        """
         Furthe rinformation on the Leukemia data:
    
        https://rdrr.io/cran/spBayesSurv/man/LeukSurv.html
    
        """
        leuk = loadtxt( "LeukSurv.csv", delimiter=",", skiprows=1)
        # leukcov = loadtxt( "LeukSurvCov.csv", delimiter=",", skiprows=1)
        leuk_logistic = SurvLogistic( surv_times = leuk[:,0], censor = leuk[:,1].astype(int))
        ### Default, NOT using ODE, for using ODE:
        #leuk_logistic.ODE = LogisticODE
        #leuk_logistic.LogLikelihood = leuk_logistic.LogLikelihoodODE
        
        #leuk_logistic.RunMCMC( T=500_000, burn_in=20_000)
        #leuk_logistic.twalk.SavetwalkOutput("leuk_logistic_Output.csv", burn_in=20_000)
        leuk_logistic.twalk.LoadtwalkOutput("leuk_logistic_Output.csv")
        leuk_logistic.AnaMCMC(burn_in=0) ##The burn_in has been removed when saved
        
        fig, ax = leuk_logistic.PlotAll(bins=20, f_nam_base="Figs/leuk_logistic_", ext=fig_fmt)
        #fig.savefig("leuk_logistic_posts.pdf")
        ax_pred_log_h, quan, solns = leuk_logistic.PlotPredRisk( x1=14, ssize=500)
        ax_pred_log_h.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        ax_pred_log_h.get_figure().savefig("Figs/leuk_logistic_h" + fig_fmt)
        ax_pred_log_S, quan, solns = leuk_logistic.PlotPredRisk( x1=28, ssize=500, surv=True, mean_col="red")
        ax_pred_log_S.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        ax_pred_log_S.get_figure().savefig("Figs/leuk_logistic_S" + fig_fmt)
    if run_leuk_har:
        leuk = loadtxt( "LeukSurv.csv", delimiter=",", skiprows=1)
        leuk_har = SurvHarmonic( surv_times = leuk[:,0], censor = leuk[:,1].astype(int))
        leuk_har.prior[0] = gamma( 2, scale=2) #tau
        leuk_har.prior[1] = gamma( 2, scale=2) #w2
        leuk_har.prior[2] = gamma( 2, scale=2) # hb
        leuk_har.prior[3] = gamma( 2, scale=2) # h0
        leuk_har.prior[4] = norm(  loc=0, scale=100) # r0

        #leuk_har.RunMCMC( T=500_000, burn_in=20_000)
        #leuk_har.twalk.SavetwalkOutput("leuk_har_Output.csv", burn_in=20_000)
        leuk_har.twalk.LoadtwalkOutput("leuk_har_Output.csv")
        leuk_har.AnaMCMC(burn_in=0) ##The burn_in has been removed when the Output was saved

        fig, ax = leuk_har.PlotAll(bins=20, f_nam_base="Figs/leuk_har_", ext=fig_fmt)
        #fig.savefig("leuk_har_posts.pdf")
        ax, quan, solns = leuk_har.PlotPredRisk( x1=14, ssize=500, ax=ax_pred_log_h)
        ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        ax.get_figure().savefig("Figs/leuk_h" + fig_fmt)
        ax, quan, solns = leuk_har.PlotPredRisk( x1=14, ssize=500, surv=True, mean_col="black", ax=ax_pred_log_S)
        ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        ax.get_figure().savefig("Figs/leuk_S" + fig_fmt)
        
    if run_ht:
        ###   Roterdam data, I just removed the text colum "size" ond ordereded by dtime
        rott = loadtxt( "Rotterdam2.csv", delimiter=",", skiprows=1)
        ###           Rotterdam data                 dtime                    death
        rott_ht = SurvHazardTreatment( surv_times = rott[:, 12]/365.24, censor = rott[:,13].astype(int),\
                    h0=0.01, q0=1e-6)
        
        #rott_ht.RunMCMC( T=200_000, burn_in=2000)
        #rott_ht.twalk.SavetwalkOutput("rott_ht_Output.csv", burn_in=2000)
        rott_ht.twalk.LoadtwalkOutput("rott_ht_Output.csv")
        rott_ht.AnaMCMC(burn_in=0) ##The burn_in has been removed when the Output was saved
        
        fig, ax = rott_ht.PlotAll(bins=20, f_nam_base="Figs/rott_ht_", ext=fig_fmt)
        #fig.savefig("rott_ht_posts.pdf")
        ax, quan, solns = rott_ht.PlotPredRisk( x1=None, ssize=500)
        ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        ax.get_figure().savefig("Figs/rott_ht_h" + fig_fmt)
        ax, quan, solns = rott_ht.PlotPredRisk( x1=None, mult=3, ssize=500, surv=True, mean_col="red")
        ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        ax.get_figure().savefig("Figs/rott_ht_S" + fig_fmt)
        
        ###Stability analysis
        # ['$\\lambda$', '$\\kappa$', '$\\alpha$', '$\\beta$']
        #        0             1            2           3
        # a = al*kappa/la
        a = rott_ht.twalk.Output[:,2]*rott_ht.twalk.Output[:,1]/rott_ht.twalk.Output[:,0]
        # b = al*kappa/beta
        b = rott_ht.twalk.Output[:,2]*rott_ht.twalk.Output[:,1]/rott_ht.twalk.Output[:,3]
        D = 1 - a*b
        h_star = rott_ht.twalk.Output[:,1]*(1-a)/D
        q_star = rott_ht.twalk.Output[:,1]*(1-b)/D
        equilibrium = (h_star > 0) * (q_star > 0) * (D > 0)
        ### all simulated points are stable
        fig_star, ax_star = subplots()
        ax_star.hist( h_star, bins=20, color="blue", density=True)
        ax_star.hist( q_star, bins=20, color="orange", histtype='step', linewidth=1.5, density=True)
        ax_star.set_xlabel(r"$h^*, q^*$")
        ax_star.set_xlim((0.05,0.140))
        ax_star.set_ylabel("Density")
        fig_star.tight_layout()
        fig_star.savefig("Figs/rott_ht_h_star" + fig_fmt)
        
        
        





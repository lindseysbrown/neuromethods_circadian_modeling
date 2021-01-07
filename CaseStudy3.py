# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:34:29 2016

@author: abel

Model from Kronauer et al. 1999 in JBR, "Quantifying human circadian response
to light."

We will define this model to take cs-usable light inputs. This probest he 
underlying oscillator, not so much process L.
"""

import casadi as cs
import numpy as np
import gillespy
import matplotlib.pyplot as plt
from local_imports import LimitCycle as ctb
from scipy.integrate import odeint


#Setup different light functions to be used with Kronauer model
def I_CR(t):
    """
    Test light input function. 9500lux from t=10 to 11.5
    """
    return 150*cs.heaviside(t-cs.floor(t/24)*24-8)-150*cs.heaviside(t-cs.floor(t/24)*24-24)

def I_null(t):
    return 0

def I_dur_smooth(duration, start_t = 15):
    """ 
    Test light input function. 9500lux from t=10 to 10.5. Smooth, works best
    for longer pulses (dur>0.1h).
    """
    def np_heavisidesmooth(t):
        return 0.5 + 0.5*np.tanh(300*t)
    
    def I_fun(t):
        return (1*9500)*np_heavisidesmooth(t-start_t) - (1*9500)*np_heavisidesmooth(t-start_t-duration) 
        
    return I_fun

def I_dur(duration, start_t = 15):
    """ 
    Test light input function. 9500lux, discontinuities at start_t, start_T+dur
    """
    def I(t):
        return (1*9500/2)*(np.sign(t-start_t)+1) - (1*9500/2)*(np.sign(t-start_t-duration)+1)
    return I


# Constants and Equation Setup
EqCount = 3
ParamCount = 9

param_P = [0.13, 0.55, 1./3, 24.2]
param_L = [0.16, 0.013, 19.875, 0.6, 9500] #alpha0, beta, G, p, I0 original parameters

param = param_P+param_L

y0in = [ -0.17,  -1.22, .5]

y0DLMOn = [ 4.62084204e-01,  -8.96101555e-01,   0]

#y0in = [ -0.55,  -1.11, 0.19]
period = param_P[-1]

def kronauer(I, paramset):
    """
    Function for the L-P process model of human circadian rhythms. Takes 
    function I(t), light intensity, as its input. I(t) must be able to be 
    evaluated by casadi.
    
    Some LC information cannot be computed here because n does not use a limit 
    cycle. Instead, use kronauer_LC for limit cycle calcs.
    """

    
    
    #===================================================================
    # set up the ode system
    #===================================================================
    
    # light input
        
    [mu, k, q, taux, alpha0, beta, G, p, I0] = paramset
    
    def alpha(t):
        return alpha0*(I(t)/I0)**p
    
        # output drive onto the pacemaker for the prompt response model
    def Bhat(t, n):
        return G*alpha(t)*(1-n)
    
    # circadian modulation of photic sensitivity
    def B(t, x, xc, n):
        return (1-0.4*x)*(1-0.4*xc)*Bhat(t, n)
    
    
    def func(y, t):
        x, xc, n = y
        ode = [[]]*EqCount
        ode[0] = (cs.pi/12)*(xc +mu*(x/3. + (4/3.)*x**3 - (256/105.)*x**7)+B(t, x, xc, n))
        ode[1] = (cs.pi/12)*(q*B(t, x, xc, n)*xc - ((24/(0.99729*taux))**2 + k*B(t, x, xc, n))*x)
        ode[2] = 60*(alpha(t)*(1-n)-beta*n)   
        return ode
        
    return func

if __name__ == "__main__":


    ts = np.linspace(0, 240, 24000)
    sol = odeint(kronauer(lf.I_dur(2, start_t = 10), param), y0DLMOn, ts)
    solb = odeint(kronauer(lf.I_dur(6, start_t=10), param), y0DLMOn, ts)
    
    plt.figure()
    plt.plot(ts, sol)
    plt.plot(ts, solb)
    plt.title('Effect of Different Duration Light Pulses')




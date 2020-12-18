"""
Created on Tue Jan 13 13:01:35 2014

@author: John H. Abel
"""

# common imports
from __future__ import division

# python packages
import numpy as np
import casadi as cs

modelversion = 'gonze_model'

# constants and equations setup, trying a new method
EqCount = 4
ParamCount  = 20

param = [  0.7,    1,    4, 0.35,    1,  0.7, 0.35,   
             1,  0.7, 0.35,    1, 0.35,    1,    1,
           0.4,    1,  0.75,    0 # second to last shoudl be 0.75
           ]
y0in = np.array([ 0.05069219,  0.10174506,  2.28099242,  0.01522458])
period = 30.27

def ODEmodel(ko=None, bmalko=None):
    
    # For two oscillators
    X1 = cs.SX.sym('X1')
    Y1 = cs.SX.sym('Y1')
    Z1 = cs.SX.sym('Z1')
    A1 = cs.SX.sym('A1')
    
    state_set = cs.vertcat([X1, Y1, Z1, A1])

    # Parameter Assignments
    v1  = cs.SX.sym('v1')
    K1  = cs.SX.sym('K1')
    n   = cs.SX.sym('n')
    v2  = cs.SX.sym('v2')
    K2  = cs.SX.sym('K2')
    k3  = cs.SX.sym('k3')
    v4  = cs.SX.sym('v4')
    K4  = cs.SX.sym('K4')
    k5  = cs.SX.sym('k5')
    v6  = cs.SX.sym('v6')
    K6  = cs.SX.sym('K6')
    k7  = cs.SX.sym('k7')
    v8  = cs.SX.sym('v8')
    K8  = cs.SX.sym('K8')
    vc  = cs.SX.sym('vc')
    Kc  = cs.SX.sym('Kc')
    K   = cs.SX.sym('K')
    L   = cs.SX.sym('L')

    param_set = cs.vertcat([v1,K1,n,v2,K2,k3,v4,K4,k5,v6,K6,k7,v8,K8,vc,Kc,K,L])

    # Time
    t = cs.SX.sym('t')

    # oscillators
    ode = [[]]*EqCount
    
    ode[0] = v1*K1**n/(K1**n + Z1**n) \
             - v2*(X1)/(K2+X1) \
             +vc*K*((A1))/(Kc +K*(A1))
             
    ode[1] = k3*(X1) - v4*Y1/(K4+Y1)
    
    ode[2] = k5*Y1 - v6*Z1/(K6+Z1)
    ode[3] = k7*(X1) - v8*A1/(K8+A1)

    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t,x=state_set,p=param_set), 
            cs.daeOut(ode=ode))

    fn.setOption("name","gonze_model")

    return fn

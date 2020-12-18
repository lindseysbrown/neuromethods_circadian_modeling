"""
Created on 9 Aug 2018

@author: John H. Abel

60 cell version of model from Gonze 2005.
Constructed using gillespy.
"""

# common imports
from __future__ import division
from collections import OrderedDict

# python packages
import numpy as np
import casadi as cs
import gillespy as gsp

modelversion = 'gonze_model_manycell'

param = [  0.7,    1,    4, 0.35,    1,  0.7, 0.35,   
             1,  0.7, 0.35,    1, 0.35,    1,    1,
           0.4,    1,  0.75,    0
           ]

period = 35.111795693955706

class GonzeModelManyCells(gsp.Model):
    """ Stochastic version of the ODE model. Contains 60 cells."""
    def __init__(self, parameter_values=param, initial_values=[], 
                 bmalko='None', AVPcells=53, VIPcells=27):
        """
        """
        assert AVPcells+VIPcells==80, \
                 "Total cells involved in signaling !=80."
        kav = 2.5 # 2.5 AVP:1VIP
        sv = 1000 #system volume
        gsp.Model.__init__(self, name="gonze120", volume=sv)
        self.timespan(np.linspace(0,7*period,7*4*24+1))

        # set up kos, bmalkos
        avpbmalko = '1*'
        avpbmalko_ic=1.
        vipbmalko = '1*'
        vipbmalko_ic=1.
        bmalko_v1 = '0.05*'

        # set up kos
        if bmalko=='AVP':
            avpbmalko = bmalko_v1
            avpbmalko_ic=0.05
        elif bmalko=='VIP':
            vipbmalko = bmalko_v1
            vipbmalko_ic=0.05
        elif bmalko=='AVPVIP':
            vipbmalko=bmalko_v1
            avpbmalko=bmalko_v1
            avpbmalko_ic=0.05
            vipbmalko_ic=0.05
    
        # For identical v2
        pnames = ['v1','K1','n','v2','K2','k3','v4','K4',
                    'k5','v6','K6','k7','v8','K8','vc','Kc','K','L']
        v1 = gsp.Parameter(name=pnames[0], expression=parameter_values[0])
        K1 = gsp.Parameter(name=pnames[1], expression=parameter_values[1])
        n =  gsp.Parameter(name=pnames[2], expression=parameter_values[2])
        v2 = gsp.Parameter(name=pnames[3], expression=parameter_values[3])
        K2 = gsp.Parameter(name=pnames[4], expression=parameter_values[4])
        k3 = gsp.Parameter(name=pnames[5], expression=parameter_values[5])
        v4 = gsp.Parameter(name=pnames[6], expression=parameter_values[6])
        K4 = gsp.Parameter(name=pnames[7], expression=parameter_values[7])
        k5 = gsp.Parameter(name=pnames[8], expression=parameter_values[8])
        v6 = gsp.Parameter(name=pnames[9], expression=parameter_values[9])
        K6 = gsp.Parameter(name=pnames[10], expression=parameter_values[10])
        k7 = gsp.Parameter(name=pnames[11], expression=parameter_values[11])
        v8 = gsp.Parameter(name=pnames[12], expression=parameter_values[12])
        K8 = gsp.Parameter(name=pnames[13], expression=parameter_values[13])
        vc = gsp.Parameter(name=pnames[14], expression=parameter_values[14])
        Kc = gsp.Parameter(name=pnames[15], expression=parameter_values[15])
        K =  gsp.Parameter(name=pnames[16], expression=parameter_values[16])
        L =  gsp.Parameter(name=pnames[17], expression=parameter_values[17])
        self.add_parameter([v1,K1,n,v2,K2,k3,v4,K4,k5,v6,K6,k7,
                           v8,K8,vc,Kc,K,L])

        # add all states as a dictionary
        NAVcells = 40

        state_dict = OrderedDict()
        avpcoupling = '0'
        for cellidx in range(AVPcells):
            co = cellidx*4
            # first compartment: AVP
            state_dict['X1'+str(cellidx)] = gsp.Species(name="X1"+str(cellidx),
                        initial_value=int(initial_values[0+co]*sv))
            state_dict['Y1'+str(cellidx)] = gsp.Species(name="Y1"+str(cellidx),
                        initial_value=int(initial_values[1+co]*sv))
            state_dict['Z1'+str(cellidx)] = gsp.Species(name="Z1"+str(cellidx),
                        initial_value=int(initial_values[2+co]*sv))
            state_dict['A1'+str(cellidx)] = gsp.Species(name="A1"+str(cellidx),
                    initial_value=int(avpbmalko_ic*initial_values[3+co]*sv))
            avpcoupling+='+A1'+str(cellidx)

        vipcoupling = '0'
        for cellidx in range(VIPcells):
            co = cellidx*4 + AVPcells*4
            # second compartment: VIP
            state_dict['X2'+str(cellidx)] = gsp.Species(name="X2"+str(cellidx),
                        initial_value=int(initial_values[0+co]*sv))
            state_dict['Y2'+str(cellidx)] = gsp.Species(name="Y2"+str(cellidx),
                        initial_value=int(initial_values[1+co]*sv))
            state_dict['Z2'+str(cellidx)] = gsp.Species(name="Z2"+str(cellidx),
                        initial_value=int(initial_values[2+co]*sv))
            state_dict['V2'+str(cellidx)] = gsp.Species(name="V2"+str(cellidx),
                    initial_value=int(vipbmalko_ic*initial_values[3+co]*sv))
            vipcoupling+="+V2"+str(cellidx)

        for cellidx in range(NAVcells):
            co = cellidx*3 + AVPcells*4 + VIPcells*4
            # third compartment: Nothing
            state_dict['X3'+str(cellidx)] = gsp.Species(name="X3"+str(cellidx),
                        initial_value=int(initial_values[0+co]*sv))
            state_dict['Y3'+str(cellidx)] = gsp.Species(name="Y3"+str(cellidx),
                        initial_value=int(initial_values[1+co]*sv))
            state_dict['Z3'+str(cellidx)] = gsp.Species(name="Z3"+str(cellidx),
                        initial_value=int(initial_values[2+co]*sv))

        sd = state_dict
        self.add_species(sd.values())

        kav = str(kav)
        ka = kav+'/('+kav+'+1.)'
        kv = '1/('+kav+'+1.)'
        coupling_str = '(('+ka+'*('+avpcoupling+')+'+\
                        kv+'*('+vipcoupling+'))/(40*vol))'
        # generate all rxns that are common to AVP cells
        for cellindex in range(AVPcells):
            ci = str(cellindex)
            rxn1 = gsp.Reaction(name = 'X1'+ci+'_production',
                    reactants = {},
                    products = {sd['X1'+ci]:1},
                    propensity_function = avpbmalko+'vol*v1*K1*K1*K1*K1/(K1*K1*K1*K1 +'+\
                                    '(Z1'+ci+'/vol)*(Z1'+ci+'/vol)*(Z1'+ci+\
                                    '/vol)*(Z1'+ci+'/vol))')

            rxn2 = gsp.Reaction(name = 'X1'+ci+'_degradation',
                        reactants = {sd['X1'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v2*(X1'+ci+'/vol)/(K2+X1'+\
                                                ci+'/vol)')

            rxn3 = gsp.Reaction(name='X1'+ci+'_coupling',
                         reactants={},
                         products={sd['X1'+ci]:1},
                         propensity_function='vol*vc*K*('+coupling_str+')/(Kc +K*'+coupling_str+')')

            rxn4 = gsp.Reaction(name='Y1'+ci+'_production',
                         reactants={sd['X1'+ci]:1},
                         products={sd['Y1'+ci]:1, sd['X1'+ci]:1},
                         rate=k3)

            rxn5 = gsp.Reaction(name = 'Y1'+ci+'_degradation',
                        reactants = {sd['Y1'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v4*(Y1'+ci+'/vol)/(K4+Y1'+\
                                                ci+'/vol)')

            rxn6 = gsp.Reaction(name='Z1'+ci+'_production',
                         reactants={sd['Y1'+ci]:1},
                         products={sd['Z1'+ci]:1, sd['Y1'+ci]:1},
                         rate=k5)

            rxn7 = gsp.Reaction(name = 'Z1'+ci+'_degradation',
                        reactants = {sd['Z1'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v6*(Z1'+ci+'/vol)/(K6+Z1'+\
                                                ci+'/vol)')

            rxn8 = gsp.Reaction(name='A1'+ci+'_production',
                         reactants={sd['X1'+ci]:1},
                         products={sd['X1'+ci]:1, sd['A1'+ci]:1},
                         rate=k7)

            rxn9 = gsp.Reaction(name = 'A1'+ci+'_degradation',
                        reactants = {sd['A1'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v8*(A1'+ci+'/vol)/(K8+A1'+\
                                                ci+'/vol)')

            self.add_reaction([rxn1,rxn2,rxn3,rxn4,rxn5,rxn6,rxn7,rxn8,rxn9])


        # generate all rxns that are common to VIP cells
        for cellindex in range(VIPcells):
            ci = str(cellindex)
            rxn1 = gsp.Reaction(name = 'X2'+ci+'_production',
                    reactants = {},
                    products = {sd['X2'+ci]:1},
                    propensity_function = vipbmalko+'vol*v1*K1*K1*K1*K1/(K1*K1*K1*K1 +'+\
                                    '(Z2'+ci+'/vol)*(Z2'+ci+'/vol)*(Z2'+ci+\
                                    '/vol)*(Z2'+ci+'/vol))')

            rxn2 = gsp.Reaction(name = 'X2'+ci+'_degradation',
                        reactants = {sd['X2'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v2*(X2'+ci+'/vol)/(K2+X2'+\
                                                ci+'/vol)')

            rxn3 = gsp.Reaction(name='X2'+ci+'_coupling',
                         reactants={},
                         products={sd['X2'+ci]:1},
                         propensity_function='vol*vc*K*('+coupling_str+')/(Kc +K*'+coupling_str+')')

            rxn4 = gsp.Reaction(name='Y2'+ci+'_production',
                         reactants={sd['X2'+ci]:1},
                         products={sd['Y2'+ci]:1, sd['X2'+ci]:1},
                         rate=k3)

            rxn5 = gsp.Reaction(name = 'Y2'+ci+'_degradation',
                        reactants = {sd['Y2'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v4*(Y2'+ci+'/vol)/(K4+Y2'+\
                                                ci+'/vol)')

            rxn6 = gsp.Reaction(name='Z2'+ci+'_production',
                         reactants={sd['Y2'+ci]:1},
                         products={sd['Z2'+ci]:1, sd['Y2'+ci]:1},
                         rate=k5)

            rxn7 = gsp.Reaction(name = 'Z2'+ci+'_degradation',
                        reactants = {sd['Z2'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v6*(Z2'+ci+'/vol)/(K6+Z2'+\
                                                ci+'/vol)')

            rxn8 = gsp.Reaction(name='V2'+ci+'_production',
                         reactants={sd['X2'+ci]:1},
                         products={sd['X2'+ci]:1, sd['V2'+ci]:1},
                         rate=k7)

            rxn9 = gsp.Reaction(name = 'V2'+ci+'_degradation',
                        reactants = {sd['V2'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v8*(V2'+ci+'/vol)/(K8+V2'+\
                                                ci+'/vol)')

            self.add_reaction([rxn1,rxn2,rxn3,rxn4,rxn5,rxn6,rxn7,rxn8,rxn9])


        # generate all rxns that are common to NAV cells
        for cellindex in range(NAVcells):
            ci = str(cellindex)
            rxn1 = gsp.Reaction(name = 'X3'+ci+'_production',
                    reactants = {},
                    products = {sd['X3'+ci]:1},
                    propensity_function = 'vol*v1*K1*K1*K1*K1/(K1*K1*K1*K1 +'+\
                                    '(Z3'+ci+'/vol)*(Z3'+ci+'/vol)*(Z3'+ci+\
                                    '/vol)*(Z3'+ci+'/vol))')

            rxn2 = gsp.Reaction(name = 'X3'+ci+'_degradation',
                        reactants = {sd['X3'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v2*(X3'+ci+'/vol)/(K2+X3'+\
                                                ci+'/vol)')

            rxn3 = gsp.Reaction(name='X3'+ci+'_coupling',
                         reactants={},
                         products={sd['X3'+ci]:1},
                         propensity_function='vol*vc*K*('+coupling_str+')/(Kc +K*'+coupling_str+')')

            rxn4 = gsp.Reaction(name='Y3'+ci+'_production',
                         reactants={sd['X3'+ci]:1},
                         products={sd['Y3'+ci]:1, sd['X3'+ci]:1},
                         rate=k3)

            rxn5 = gsp.Reaction(name = 'Y3'+ci+'_degradation',
                        reactants = {sd['Y3'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v4*(Y3'+ci+'/vol)/(K4+Y3'+\
                                                ci+'/vol)')

            rxn6 = gsp.Reaction(name='Z3'+ci+'_production',
                         reactants={sd['Y3'+ci]:1},
                         products={sd['Z3'+ci]:1, sd['Y3'+ci]:1},
                         rate=k5)

            rxn7 = gsp.Reaction(name = 'Z3'+ci+'_degradation',
                        reactants = {sd['Z3'+ci]:1},
                        products = {},
                        propensity_function = 'vol*v6*(Z3'+ci+'/vol)/(K6+Z3'+\
                                                ci+'/vol)')


            self.add_reaction([rxn1,rxn2,rxn3,rxn4,rxn5,rxn6,rxn7])

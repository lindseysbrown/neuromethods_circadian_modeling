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

y0in = np.array(
      [0.03649647, 0.10663466, 2.60469301, 0.01192638, 0.15020802,
       0.19639209, 1.699136  , 0.04282985, 0.17498466, 0.22961959,
       1.65447964, 0.05062847, 0.27392133, 0.44823915, 1.69886225,
       0.09175747, 0.09678593, 0.5411219 , 3.46699745, 0.04373477,
       0.10162063, 0.14110837, 1.85774339, 0.02871033, 0.06361829,
       0.10819374, 2.11877411, 0.0185076 , 0.2658215 , 0.634519  ,
       2.10819409, 0.10519723, 0.03096468, 0.13758529, 2.97860106,
       0.01136174, 0.05646195, 0.1040406 , 2.20016264, 0.01667297,
       0.1366907 , 0.17977646, 1.73219853, 0.03876298, 0.19068233,
       0.69110311, 2.75665085, 0.08382894, 0.05829896, 0.37983743,
       3.60568227, 0.0258205 , 0.09417055, 0.53273561, 3.48273887,
       0.04253476, 0.03570413, 0.10808946, 2.63461023, 0.0117711 ,
       0.28179642, 0.56138241, 1.88543179, 0.10383703, 0.11665168,
       0.59565689, 3.33402542, 0.05275335, 0.11974919, 0.16030203,
       1.7844304 , 0.03382388, 0.12658435, 0.61753486, 3.26144801,
       0.05718728, 0.03193911, 0.17491349, 3.20998031, 0.01257403,
       0.04458828, 0.10106302, 2.38934048, 0.01373802, 0.03436042,
       0.20631866, 3.33939272, 0.01403256, 0.28076279, 0.49189016,
       1.75574768, 0.0974223 , 0.17125861, 0.67995127, 2.91303202,
       0.07621747, 0.09305787, 0.13268258, 1.90077683, 0.02635128,
       0.03093859, 0.13870795, 2.98752288, 0.01138396, 0.23254463,
       0.32751211, 1.62028786, 0.07131719, 0.07344282, 0.11536166,
       2.02984124, 0.02107731, 0.03316637, 0.19232516, 3.28694656,
       0.01334854, 0.16865091, 0.22075213, 1.66411195, 0.04858779,
       0.10453018, 0.14406946, 1.84449061, 0.02951992, 0.05530311,
       0.36288311, 3.6013551 , 0.0243991 , 0.03799234, 0.24170443,
       3.44350626, 0.01595947, 0.03255529, 0.11877973, 2.7951997 ,
       0.01127483, 0.04080215, 0.26524999, 3.49495685, 0.01737934,
       0.03095935, 0.13779727, 2.98029873, 0.01136584, 0.06170946,
       0.39814127, 3.60625163, 0.02743211, 0.18540888, 0.24484723,
       1.64112094, 0.05406722, 0.0372021 , 0.10556097, 2.5800791 ,
       0.01207008, 0.25520479, 0.6565869 , 2.21238126, 0.1035541 ,
       0.17709525, 0.23263695, 1.65152703, 0.05028992, 0.33242437,
       3.58392194, 0.06929097, 0.43548697, 3.59463344, 0.04627113,
       0.10100334, 2.35641368, 0.12577388, 0.16705929, 1.76430733,
       0.28085039, 0.57109287, 1.90849285, 0.0996584 , 0.54998659,
       3.44913621, 0.03121411, 0.16046703, 3.1335892 , 0.17179048,
       0.22511317, 1.65919141, 0.05515041, 0.36199537, 3.60102535,
       0.21525365, 0.29388882, 1.6198416 , 0.0493198 , 0.10139819,
       2.30291576, 0.06848032, 0.43169606, 3.59658506, 0.18030285,
       0.68627538, 2.84042064, 0.06126784, 0.10670654, 2.14364334,
       0.03377758, 0.11327042, 2.7216808 , 0.21964986, 0.30198587,
       1.61895939, 0.10876944, 0.57584794, 3.38915257, 0.14186617,
       0.18602192, 1.71872694, 0.28201589, 0.55856572, 1.87901187])

period = 35.111795693955706

class GonzeModelManyCells(gsp.Model):
    """ Stochastic version of the ODE model. Contains 60 cells."""
    def __init__(self, parameter_values=param, initial_values=y0in, 
                 bmalko='None', kav=5):
        """
        """
        sv = 1000 #system volume
        gsp.Model.__init__(self, name="gonze60", volume=sv)
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
        AVPcells = 20
        VIPcells = 20
        NAVcells = 20

        state_dict = OrderedDict()
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
        coupling_str = '('+ka+'*(A10+A11+A12+A13+A14+A15+A16+A17+A18+A19+A110+A111+A112+A113+A114+A115+A116+A117+A118+A119)/(20*vol) +'+\
        kv+'*(V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V210+V211+V212+V213+V214+V215+V216+V217+V218+V219)/(20*vol))'
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
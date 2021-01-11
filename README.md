# neuromethods_circadian_modeling
Python code for the models used in the case studies of "Mathematical Modeling of Circadian Rhythms" in Circadian Clocks

In the accompanying chapter, we used three case studies to illustrate the use of mathematical modeling as a research technique to explain results as well as test and produce hypotheses.  In this repository, we include code for each of the three models for the reader interested in exploring the results of the models further.  All of these models are developed in Python 2.7.3.  Other package dependencies can be found in requirements.txt.

* CaseStudy1.py: This is the code for the model of the two molecular level feedback loops from Brown & Doyle 2020.  For more details and the original code, see [this repository](https://github.com/lindseysbrown/brown_circadian_dual-feedback_loop).
* CaseStudy2.py: This is the code for the final network level model used in Shan et al. 2020, which is based on the Gonze model (contained in local_models).  For more details and the original code, see [this repository](https://github.com/johnabel/shan_abel_AVP_VIP_modeling).
* CaseStudy3.py: This is the code for the Kronauer model (1999) that informed the simulations of the light protocols in the third case study.

This repository also contains locally used imports with basic tools for analysis and plotting (local_imports), which were developed by authors for other papers:

Abel JH, Chakrabarty A, Klerman EB, Doyle III FJ. Pharmaceutical-based entrainment of circadian phase via nonlinear model predictive control. Automatica. 2019; 100:336-348.

Hirota T, Lee JW, St. John PC, Sawa M, Iwaisako K, Noguchi T, et al. Identification of small molecule activators of cryptochrome. Science. 2012, 337(6098):1094-1097.

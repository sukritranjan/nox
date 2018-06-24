import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.integrate
import scipy.optimize
from scipy import interpolate as interp
from matplotlib.pyplot import cm


######################################
###Physical constants
######################################
R=8.3e-3 #units: kJ mol**-1 K**-1

#Gibbs free energy of formations under standard conditions taken from CRC 98
G_form_o_no3= -111.3 #kJ mol**-1
G_form_o_no2= -32.3 #kJ mol**-1
G_form_o_h= 0.#kJ mol**-1

G_form_o_h2o=-237.1 #kJ mol**-1

G_form_o_h2= 0. #kJ mol**-1
G_form_o_n2= 0. #kJ mol**-1
G_form_o_o2= 0. #kJ mol**-1

#Henry's Law constants taken from Sander 2015 (1 m**-3 Pa**-1 = 100 L**-1 bar**-1 -------> 1 mol m**-3 Pa**-1 = 100 M bar**-1)
H_h2= 7.8e-4 #M bar**-1 
H_o2= 1.3e-3 #M bar**-1
H_n2= 6.4e-4 #M bar**-1


######################################
###Tunable parameters
######################################
T=298. #K

pH=7.

pH2=1.e-3 #bar
pO2=2.e-6 #bar
pN2=1.1 #bar

######################################
###Intermediate parameters
######################################

conc_H=10.**(-pH)
conc_H2=pH2*H_h2
conc_O2=pO2*H_o2
conc_N2=pN2*H_n2

print '[O2], [N2],[H2]', conc_O2, conc_N2, conc_H2
print '\n\n'

#######################################
####Reaction: 4NO3- + 4H+ ----> 5O2 + 2N2 + 2H2O
#######################################
#G_o_nitrate_o2= (5.*G_form_o_o2 + 2.*G_form_o_n2 + 2.*G_form_o_h2o) - (4.*G_form_o_no3 + 4.*G_form_o_h) #kJ/mol
#print 'G_o_nitrate_o2 ', G_o_nitrate_o2

#Q_nitrate_o2=np.exp(-G_o_nitrate_o2/(R*T)) #dimensionless
#print 'Q_nitrate_o2',  Q_nitrate_o2

#conc_nitrate_o2=((conc_N2**2. * conc_O2**5.)/(conc_H**4. * Q_nitrate_o2))**0.25 #M; above this value, the reaction is spontaneous.
#print 'conc_nitrate_o2', conc_nitrate_o2

#print '\n\n'


#######################################
####Reaction: 4NO2- + 4H+ ----> 3O2 + 2N2 + 2H2O
#######################################
#G_o_nitrite_o2= (3.*G_form_o_o2 + 2.*G_form_o_n2 + 2.*G_form_o_h2o) - (4.*G_form_o_no2 + 4.*G_form_o_h) #kJ/mol
#print 'G_o_nitrite_o2 ', G_o_nitrite_o2

#Q_nitrite_o2=np.exp(-G_o_nitrite_o2/(R*T)) #dimensionless
#print 'Q_nitrite_o2',  Q_nitrite_o2

#conc_nitrite_o2=((conc_N2**2. * conc_O2**3.)/(conc_H**4. * Q_nitrite_o2))**0.25 #M; above this value, the reaction is spontaneous.
#print 'conc_nitrite_o2', conc_nitrite_o2

#print '\n\n'

######################################
###Reaction: 2NO3- + 2H+ + 5H2 ----> N2 + 6H2O
######################################
G_o_nitrate_h2=(6.*G_form_o_h2o + G_form_o_n2) - (2.*G_form_o_no3 + 2.*G_form_o_h + 5.*G_form_o_h2)
print 'G_o_nitrate_h2', G_o_nitrate_h2

Q_nitrate_h2=np.exp(-G_o_nitrate_h2/(R*T)) #dimensionless.
print 'Q_nitrate_h2', Q_nitrate_h2

conc_nitrate_h2=(conc_N2/(Q_nitrate_h2 * conc_H**2. * conc_H2**5.))**0.5
print 'conc_nitrate_h2', conc_nitrate_h2

print '\n\n'


######################################
###Reaction: 2NO2- + 2H+ + 3H2 ----> N2 + 4H2O
######################################
G_o_nitrite_h2=(4.*G_form_o_h2o + G_form_o_n2) - (2.*G_form_o_no2 + 2.*G_form_o_h + 3.*G_form_o_h2)
print 'G_o_nitrite_h2', G_o_nitrite_h2

Q_nitrite_h2=np.exp(-G_o_nitrite_h2/(R*T)) #dimensionless.
print 'Q_nitrite_h2', Q_nitrite_h2

conc_nitrite_h2=(conc_N2/(Q_nitrite_h2 * conc_H**2. * conc_H2**3.))**0.5
print 'conc_nitrite_h2', conc_nitrite_h2

print '\n\n'

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

#Gibbs free energy of formations under standard conditions
G_form_o_no3_aq= -111.3 #kJ mol**-1; aq; CRC 98
G_form_o_no2_aq= -32.3 #kJ mol**-1; aq; CRC 98
G_form_o_h_aq= 0.#kJ mol**-1; aq; CRC 98
#G_form_o_fe_aq=-78.9 #kJ mol**-1; l; CRC 98

G_form_o_h2_g= 0 #kJ mol**-1; aq; CRC 98
G_form_o_n2_g= 0 #kJ mol**-1; aq; CRC 98
G_form_o_o2_g= 0 #kJ mol**-1; aq; CRC 98
#G_form_o_n2o_g= 103.7 #kJ mol**-1; aq; CRC 98


G_form_o_h2_aq= 17.72 #kJ mol**-1; aq; Amend & Shock 2001
G_form_o_n2_aq= 18.18 #kJ mol**-1; aq; Amend & Shock 2001
G_form_o_o2_aq= 16.54 #kJ mol**-1; aq; Amend & Shock 2001

G_form_o_h2o_l= -237.1 #kJ mol**-1; l; CRC 98
#G_form_o_fe3o4_s= -1015.4 #kJ mol**-1; l; CRC 98


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

conc_Fe=1.e-6

######################################
###Intermediate parameters
######################################

conc_H=10.**(-pH)
conc_H2=pH2*H_h2
conc_O2=pO2*H_o2
conc_N2=pN2*H_n2

#print '[O2], [N2],[H2]', conc_O2, conc_N2, conc_H2
#print '\n\n'

#######################################
####Reaction: 4NO3-(aq) + 4H+(aq) ----> 5O2(aq) + 2N2(aq) + 2H2O(l)
#######################################
#print '4NO3-(aq) + 4H+(aq) ----> 5O2(aq) + 2N2(aq) + 2H2O(l)'

#G_o_nitrate_o2= (5.*G_form_o_o2_aq + 2.*G_form_o_n2_aq + 2.*G_form_o_h2o_l) - (4.*G_form_o_no3_aq + 4.*G_form_o_h_aq) #kJ/mol
#print 'G_o_nitrate_o2 ', G_o_nitrate_o2

#Q_nitrate_o2=np.exp(-G_o_nitrate_o2/(R*T)) #dimensionless
#print 'Q_nitrate_o2',  Q_nitrate_o2

#conc_nitrate_o2=((conc_N2**2. * conc_O2**5.)/(conc_H**4. * Q_nitrate_o2))**0.25 #M; above this value, the reaction is spontaneous.
#print 'conc_nitrate_o2', conc_nitrate_o2

#print '\n\n'


######################################
###Reaction: 4NO3-(aq) + 4H+(aq) ----> 5O2(g) + 2N2(g) + 2H2O(l)
######################################
print '4NO3-(aq) + 4H+(aq) ----> 5O2(g) + 2N2(g) + 2H2O(l)'

G_o_nitrate_o2= (5.*G_form_o_o2_g + 2.*G_form_o_n2_g + 2.*G_form_o_h2o_l) - (4.*G_form_o_no3_aq + 4.*G_form_o_h_aq) #kJ/mol
print 'G_o_nitrate_o2 ', G_o_nitrate_o2

Q_nitrate_o2=np.exp(-G_o_nitrate_o2/(R*T)) #dimensionless
print 'Q_nitrate_o2',  Q_nitrate_o2

conc_nitrate_o2=((pN2**2. * pO2**5.)/(conc_H**4. * Q_nitrate_o2))**0.25 #M; above this value, the reaction is spontaneous.
print 'conc_nitrate_o2', conc_nitrate_o2

print '\n\n'


#######################################
####Reaction: 4NO2-(aq) + 4H+(aq) ----> 3O2(aq) + 2N2(aq) + 2H2O(l)
#######################################
#print '4NO2-(aq) + 4H+(aq) ----> 3O2(aq) + 2N2(aq) + 2H2O(l)'

#G_o_nitrite_o2= (3.*G_form_o_o2_aq + 2.*G_form_o_n2_aq + 2.*G_form_o_h2o_l) - (4.*G_form_o_no2_aq + 4.*G_form_o_h_aq) #kJ/mol
#print 'G_o_nitrite_o2 ', G_o_nitrite_o2

#Q_nitrite_o2=np.exp(-G_o_nitrite_o2/(R*T)) #dimensionless
#print 'Q_nitrite_o2',  Q_nitrite_o2

#conc_nitrite_o2=((conc_N2**2. * conc_O2**3.)/(conc_H**4. * Q_nitrite_o2))**0.25 #M; above this value, the reaction is spontaneous.
#print 'conc_nitrite_o2', conc_nitrite_o2

#print '\n\n'

######################################
###Reaction: 4NO2-(aq) + 4H+(aq) ----> 3O2(g) + 2N2(g) + 2H2O(l)
######################################
print '4NO2-(aq) + 4H+(aq) ----> 3O2(g) + 2N2(g) + 2H2O(l)'

G_o_nitrite_o2= (3.*G_form_o_o2_g + 2.*G_form_o_n2_g + 2.*G_form_o_h2o_l) - (4.*G_form_o_no2_aq + 4.*G_form_o_h_aq) #kJ/mol
print 'G_o_nitrite_o2 ', G_o_nitrite_o2

Q_nitrite_o2=np.exp(-G_o_nitrite_o2/(R*T)) #dimensionless
print 'Q_nitrite_o2',  Q_nitrite_o2

conc_nitrite_o2=((pN2**2. * pO2**3.)/(conc_H**4. * Q_nitrite_o2))**0.25 #M; above this value, the reaction is spontaneous.
print 'conc_nitrite_o2', conc_nitrite_o2

print '\n\n'

#######################################
####Reaction: 2NO3-(aq) + 2H+(aq) + 5H2(aq) ----> N2(aq) + 6H2O(l)
#######################################
#print '2NO3-(aq) + 2H+(aq) + 5H2(aq) ----> N2(aq) + 6H2O(l)'

#G_o_nitrate_h2=(6.*G_form_o_h2o_l + G_form_o_n2_aq) - (2.*G_form_o_no3_aq + 2.*G_form_o_h_aq + 5.*G_form_o_h2_aq)
#print 'G_o_nitrate_h2', G_o_nitrate_h2

#Q_nitrate_h2=np.exp(-G_o_nitrate_h2/(R*T)) #dimensionless.
#print 'Q_nitrate_h2', Q_nitrate_h2

#conc_nitrate_h2=(conc_N2/(Q_nitrate_h2 * conc_H**2. * conc_H2**5.))**0.5
#print 'conc_nitrate_h2', conc_nitrate_h2

#print '\n\n'


######################################
###Reaction: 2NO3-(aq) + 2H+(aq) + 5H2(g) ----> N2(g) + 6H2O(l)
######################################
print '2NO3-(aq) + 2H+(aq) + 5H2(g) ----> N2(g) + 6H2O(l)'

G_o_nitrate_h2=(6.*G_form_o_h2o_l + G_form_o_n2_g) - (2.*G_form_o_no3_aq + 2.*G_form_o_h_aq + 5.*G_form_o_h2_g)
print 'G_o_nitrate_h2', G_o_nitrate_h2

Q_nitrate_h2=np.exp(-G_o_nitrate_h2/(R*T)) #dimensionless.
print 'Q_nitrate_h2', Q_nitrate_h2

conc_nitrate_h2=(pN2/(Q_nitrate_h2 * conc_H**2. * pH2**5.))**0.5
print 'conc_nitrate_h2', conc_nitrate_h2

print '\n\n'



#######################################
####Reaction: 2NO2-(aq) + 2H+(aq) + 3H2(aq) ----> N2(aq) + 4H2O(l)
#######################################
#print '2NO2-(aq) + 2H+(aq) + 3H2(aq) ----> N2(aq) + 4H2O(l)' 

#G_o_nitrite_h2=(4.*G_form_o_h2o_l + G_form_o_n2_aq) - (2.*G_form_o_no2_aq + 2.*G_form_o_h_aq + 3.*G_form_o_h2_aq)
#print 'G_o_nitrite_h2', G_o_nitrite_h2

#Q_nitrite_h2=np.exp(-G_o_nitrite_h2/(R*T)) #dimensionless.
#print 'Q_nitrite_h2', Q_nitrite_h2

#conc_nitrite_h2=(conc_N2/(Q_nitrite_h2 * conc_H**2. * conc_H2**3.))**0.5
#print 'conc_nitrite_h2', conc_nitrite_h2

#print '\n\n'

######################################
###Reaction: 2NO2-(aq) + 2H+(aq) + 3H2(g) ----> N2(g) + 4H2O(l)
######################################
print '2NO2-(aq) + 2H+(aq) + 3H2(g) ----> N2(g) + 4H2O(l)' 

G_o_nitrite_h2=(4.*G_form_o_h2o_l + G_form_o_n2_g) - (2.*G_form_o_no2_aq + 2.*G_form_o_h_aq + 3.*G_form_o_h2_g)
print 'G_o_nitrite_h2', G_o_nitrite_h2

Q_nitrite_h2=np.exp(-G_o_nitrite_h2/(R*T)) #dimensionless.
print 'Q_nitrite_h2', Q_nitrite_h2

conc_nitrite_h2=(pN2/(Q_nitrite_h2 * conc_H**2. * pH2**3.))**0.5
print 'conc_nitrite_h2', conc_nitrite_h2

print '\n\n'

#######################################
####Reaction: 2NO2-(aq) + 2H+(aq) + 3H2(g) ----> N2(g) + 4H2O(l)
#######################################
#print '2NO3-(aq) + 12Fe(2+, aq) + 11H2O(g) ----> N2O(g) + 4Fe3O4 + 22H+'

#G_o_nitrate_fe=(22.*G_form_o_h_aq + 4*G_form_o_fe3o4_s + G_form_o_n2o_g) - (2.*G_form_o_no3_aq + 12.*G_form_o_fe_aq + 11.*G_form_o_h2o_l)
#print 'G_o_nitrate_fe', G_o_nitrate_fe

#Q_nitrate_fe=np.exp(-G_o_nitrate_fe/(R*T)) #dimensionless.
#print 'Q_nitrate_fe', Q_nitrate_fe

#conc_nitrate_fe=(pN2/(Q_nitrite_h2 * conc_H**2. * pH2**3.))**0.5
#print 'conc_nitrate_fe', conc_nitrate_fe

#print '\n\n'

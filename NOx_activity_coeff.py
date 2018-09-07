#For calculating activity coefficients as a function of ionic strength, using three possible regimes: EDH, Davies, and TJ

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

A=.5085 #in M^-1/2
B=3.281 # in M^-1/2 nm^-1


### Effective radii for various ions
a_NO3=0.3 #nm
a_NO2=0.3 #nm
a_NH4=0.25 #nm
a_FeII=.6  #nm
a_H=.9  #nm


I=np.linspace(0,1000,1001) #ionic strength in mM (to 1 M)
I=I/1000.  #in M

I=np.linspace(0,100,1001)  #to 0.1 M - EDH
Ib=np.linspace(0,500,1001)  #to 0.5 M - DAVIES
Ic=np.linspace(0,1000,1001)  #to 1 M - TJ

I=I/1000.     #convert to M
Ib=Ib/1000.
Ic=Ic/1000.

### Extended Debye Huckel
logy_NO3=(-1.0*A*I**0.5)/(1.+B*a_NO3*I**0.5)
logy_NH4=(-1.0*A*I**0.5)/(1.+B*a_NH4*I**0.5)
logy_FeII=(-1.0*A*4.0*I**0.5)/(1.+B*a_FeII*I**0.5)  #factor of 4 because of z_i**2.0
logy_H=(-1.0*A*I**0.5)/(1.+B*a_H*I**0.5)


y_NO3=10.**logy_NO3
y_NH4=10.**logy_NH4
y_FeII=10.**logy_FeII
y_H=10.**logy_H

### Davies
logy_NO3b=(-1.0*A*Ib**0.5)/(1.+Ib**0.5) + 0.3*A*Ib
logy_NH4b=(-1.0*A*Ib**0.5)/(1.+Ib**0.5) + 0.3*A*Ib
logy_FeIIb=(-1.0*A*4.*Ib**0.5)/(1.+Ib**0.5) + 0.3*A*4.*Ib
logy_Hb=(-1.0*A*Ib**0.5)/(1.+Ib**0.5) + 0.3*A*Ib

y_NO3b=10.**logy_NO3b
y_NH4b=10.**logy_NH4b
y_FeIIb=10.**logy_FeIIb
y_Hb=10.**logy_Hb

### Truesdell-Jones
logy_NO3c=(-1.0*A*Ic**0.5)/(1.+B*a_NO3*Ic**0.5) + .1*Ic
logy_NH4c=(-1.0*A*Ic**0.5)/(1.+B*a_NH4*Ic**0.5) + .1*Ic
logy_FeIIc=(-1.0*A*4.*Ic**0.5)/(1.+B*a_FeII*Ic**0.5) + .1*Ic
logy_Hc=(-1.0*A*Ic**0.5)/(1.+B*a_H*Ic**0.5) + .1*Ic


y_NO3c=10.**logy_NO3c
y_NH4c=10.**logy_NH4c
y_FeIIc=10.**logy_FeIIc
y_Hc=10.**logy_Hc






plt.figure()
plt.plot(I,y_NO3,color='b',label='$NO_3^-, NO_2^-$',linewidth=4,ls='-')
plt.plot(Ib,y_NO3b,color='b',linewidth=4,ls='--')
plt.plot(Ic,y_NO3c,color='b',linewidth=4,ls='-.')
plt.plot(I,y_NH4,color='g',label='$NH_4^+$',linewidth=4,ls='-')
plt.plot(Ib,y_NH4b,color='g',linewidth=4,ls='--')
plt.plot(Ic,y_NH4c,color='g',linewidth=4,ls='-.')
plt.plot(I,y_FeII,color='r',label='$Fe^{2+}$',linewidth=4,ls='-')
plt.plot(Ib,y_FeIIb,color='r',linewidth=4,ls='--')
plt.plot(Ic,y_FeIIc,color='r',linewidth=4,ls='-.')
plt.plot(I,y_H,color='y',label='$H^{+}$',linewidth=4,ls='-')
plt.plot(Ib,y_Hb,color='y',linewidth=4,ls='--')
plt.plot(Ic,y_Hc,color='y',linewidth=4,ls='-.')
plt.xlabel('Ionic strength (M)',fontsize=16)
plt.ylabel('Activity coefficient',fontsize=16)
plt.tick_params(which='both',labelsize=14,pad=5)
plt.legend(loc='best')
plt.show()
plt.close()


#At I=1M:
y_NO3=0.697811859323
y_NH4=0.661672018092
y_FeII=0.259913321181
y_H=0.936181957894






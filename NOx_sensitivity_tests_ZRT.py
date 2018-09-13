################################################################################################
####          CONTAINS THE CODE USED FOR CALCULATIONS AND FIGURES FOR SENSITIVITY TESTS     ####
####        write-up of results in the document "Todd_update_NOx_sensitivity_tests.docx"    ####
################################################################################################




import numpy as np
import matplotlib
import matplotlib.pyplot as plt

########################   SALINITY DEPENDENCE OF HENRY'S LAW CONSTANT #######################
###### PARAMETERS #########
# (from Burkholder 2015)

### NH3
H0_NH3=60.0  #in M/bar
h0_NH3=-4.81E-2  #in M^-1

### N2
H0_N2=6.4E-4
h0_N2=-1.0E-3
ht_N2=-6.05E-4

### N2O
H0_N2O=2.4E-2
h0_N2O=-8.5E-3
ht_N2O=-4.79E-4

### NO
H0_NO=1.9E-3
h0_NO=6.0E-3

###### ION PARAMETERS
hi_Na=.1143
hi_Cl=.0318


c=np.linspace(0,0.5,100)   #concentration of Na/Cl in M - for total ionic strength of max 1 M
#print(c)

lgHH0_NH3=(hi_Na+h0_NH3+hi_Cl+h0_NH3)*c
lgHH0_N2=(hi_Na+h0_N2+hi_Cl+h0_N2)*c
lgHH0_N2O=(hi_Na+h0_N2O+hi_Cl+h0_N2O)*c
lgHH0_NO=(hi_Na+h0_NO+hi_Cl+h0_NO)*c

H_NH3=H0_NH3/(10.**lgHH0_NH3)
H_N2=H0_N2/(10.**lgHH0_N2)
H_N2O=H0_N2O/(10.**lgHH0_N2O)
H_NO=H0_NO/(10.**lgHH0_NO)

I=2.0*c

#Plot for salinity dependence
plt.figure()
plt.plot(I,H_NH3,ls='-',color='red',label='N(-3)',linewidth=3)
plt.plot(I,H_N2,ls='-',color='orange',label='N(0)',linewidth=3)
plt.plot(I,H_N2O,ls='-',color='yellow',label='N(+1)',linewidth=3)
plt.plot(I,H_NO,ls='-',color='green',label='N(+2)',linewidth=3)
plt.xlabel('Ionic strength (M)',fontsize=16)
plt.ylabel('Henry constant (M/bar)',fontsize=16)
plt.legend(loc='best')
plt.tick_params(which='both',labelsize=14,pad=5)
plt.yscale('log')
plt.show()
plt.close()



########################   SALINITY DEPENDENCE OF HENRY'S LAW CONSTANT #######################
###### PARAMETERS #########
# from Burkholder 2015

## NH3
A_NH3=-9.84
B_NH3=4160
C_NH3=0

## N2
A_N2=-177.1
B_N2=8640
C_N2=24.71

## N2O
A_N2O=-148.1
B_N2O=8610
C_N2O=20.266

##NO
A_NO=-157.1
B_NO=7950
C_NO=21.298

T=np.linspace(273,323,500)  #temperature range to consider in K

lnH_NH3=A_NH3+B_NH3/T+C_NH3*np.log(T)    #H in M/atm
lnH_N2=A_N2+B_N2/T+C_N2*np.log(T)
lnH_N2O=A_N2O+B_N2O/T+C_N2O*np.log(T)
lnH_NO=A_NO+B_NO/T+C_NO*np.log(T)

H_NH3=2.7182818**lnH_NH3
H_N2=2.7182818**lnH_N2
H_N2O=2.7182818**lnH_N2O
H_NO=2.7182818**lnH_NO

H_NH3=H_NH3/1.01325   #convert to M/bar
H_N2=H_N2/1.01325
H_N2O=H_N2O/1.01325
H_NO=H_NO/1.01325

#plot of temp dependence of H

aa=np.array([298,298])
bb=np.array([1.0E-10,1.0E3])

plt.figure()
plt.plot(T,H_NH3,ls='-',color='red',linewidth=3,label='N(-3)')
plt.plot(T,H_N2,ls='-',color='orange',linewidth=3,label='N(0)')
plt.plot(T,H_N2O,ls='-',color='yellow',linewidth=3,label='N(+1)')
plt.plot(T,H_NO,ls='-',color='green',linewidth=3,label='N(+2)')
plt.plot(aa,bb,ls='--',color='k')
plt.xlabel('Temperature (K)',fontsize=16)
plt.ylabel('H$_x$ (M/bar)',fontsize=16)
plt.legend(loc='best')
plt.tick_params(which='major',labelsize=14,pad=5)
plt.yscale('log')
plt.xlim(273,323)
plt.ylim(1.0E-6,1.0E3)
plt.show()
plt.close()




fig=plt.figure()
ax1=fig.add_subplot(221)
ax1.plot(T,H_NH3,ls='-',color='red',linewidth=3)
ax1.plot(aa,bb,ls='--',color='k')
ax1.set_ylabel('H$_{NH3}$ (M/bar)',fontsize=16)
ax1.tick_params(which='major',labelsize=14,pad=5)
ax1.set_xlim(273,323)
ax1.set_ylim(10,250)

ax2=fig.add_subplot(222)
ax2.plot(T,H_N2,ls='-',color='orange',linewidth=3)
ax2.plot(aa,bb,ls='--',color='k')
ax2.set_ylabel('H$_{N2}$ (M/bar)',fontsize=16)
ax2.tick_params(which='major',labelsize=14,pad=5)
ax2.set_xlim(273,323)
ax2.set_ylim(.0004,.0011)

ax3=fig.add_subplot(223,sharex=ax1)
ax3.plot(T,H_N2O,ls='-',color='yellow',linewidth=3)
ax3.plot(aa,bb,ls='--',color='k')
ax3.set_xlabel('Temperature (K)',fontsize=16)
ax3.set_ylabel('H$_{N2O}$ (M/bar)',fontsize=16)
ax3.tick_params(which='major',labelsize=14,pad=5)
ax3.set_xlim(273,323)
ax3.set_ylim(.01,.06)

ax4=fig.add_subplot(224,sharex=ax2)
ax4.plot(T,H_NO,ls='-',color='green',linewidth=3)
ax4.plot(aa,bb,ls='--',color='k')
ax4.set_xlabel('Temperature (K)',fontsize=16)
ax4.set_ylabel('H$_{NO}$ (M/bar)',fontsize=16)
ax4.tick_params(which='major',labelsize=14,pad=5)
ax4.set_xlim(273,323)
ax4.set_ylim(.00005,.00025)

plt.subplots_adjust(wspace=.5)
plt.show()
plt.close()


########################   TEMPERATURE DEPENDENCE OF K's #######################
###### PARAMETERS #########

logK4=222.3
logK8=160.8
logK12=162.1
logK14=101.6
logK15=40.4

#delta H^o_rxn in kJ/mol (note that some species are wrong state, but this was the best I could do so far)
dH04=-1301.6
dH08=-984.2
dH012=-1019.8
dH014=-633.0
dH015=-265.6

K4=10.**logK4
K8=10.**logK8
K12=10.**logK12
K14=10.**logK14
K15=10.**logK15

T0=298 #STP temp in K
T=np.linspace(273,323,100)  #range of temp to consider, in K
R=8.314E-3  #in kJ/mol/K

lnK4=np.log(K4)-(dH04/R*(1.0/T-1.0/T0))
lnK8=np.log(K8)-(dH08/R*(1.0/T-1.0/T0))
lnK12=np.log(K12)-(dH012/R*(1.0/T-1.0/T0))
lnK14=np.log(K14)-(dH014/R*(1.0/T-1.0/T0))
lnK15=np.log(K15)-(dH015/R*(1.0/T-1.0/T0))

K4=2.7182818**lnK4
K8=2.7182818**lnK8
K12=2.7182818**lnK12
K14=2.7182818**lnK14
K15=2.7182818**lnK15

logK4=np.log10(K4)
logK8=np.log10(K8)
logK12=np.log10(K12)
logK14=np.log10(K14)
logK15=np.log10(K15)


### plot of K dependence on temperature
plt.figure()
plt.plot(T,logK4,color='lime',label='Reaction 4',linewidth=3)
plt.plot(T,logK8,color='turquoise',label='Reaction 8',linewidth=3)
plt.plot(T,logK12,color='dodgerblue',label='Reaction 12',linewidth=3)
plt.plot(T,logK14,color='darkviolet',label='Reaction 14',linewidth=3)
plt.plot(T,logK15,color='fuchsia',label='Reaction 15',linewidth=3)
plt.plot(aa,bb,ls='--',color='k')
plt.xlabel('Temperature (K)',fontsize=16)
plt.ylabel('log(K)',fontsize=16)
plt.xlim(273,323)
plt.ylim(0,250)
#plt.yscale('log')
plt.tick_params(which='both',labelsize=14,pad=5)
plt.legend(loc='best')
plt.show()
plt.close()

#individually subplotted


fig=plt.figure()

ax1=fig.add_subplot(511)
ax1.plot(T,logK4,color='lime',linewidth=3)
ax1.plot(aa,bb,ls='--',color='k')
ax1.set_ylabel('log$K_4$',fontsize=16)
ax1.tick_params(which='both',labelsize=12,pad=5)
ax1.set_xlim(273,323)
ax1.set_ylim(200,245)
ax1.set_yticks([200,210,220,230,240])

ax2=fig.add_subplot(512,sharex=ax1)
ax2.plot(T,logK8,color='turquoise',linewidth=3)
ax2.plot(aa,bb,ls='--',color='k')
ax2.set_ylabel('log$K_8$',fontsize=16)
ax2.tick_params(which='both',labelsize=12,pad=5)
ax2.set_xlim(273,323)
ax2.set_ylim(145,185)
ax2.set_yticks([145,155,165,175,185])


ax3=fig.add_subplot(513,sharex=ax1)
ax3.plot(T,logK12,color='dodgerblue',linewidth=3)
ax3.plot(aa,bb,ls='--',color='k')
ax3.set_ylabel('log$K_{12}$',fontsize=16)
ax3.tick_params(which='both',labelsize=12,pad=5)
ax3.set_xlim(273,323)
ax3.set_ylim(145,185)
ax3.set_yticks([145,155,165,175,185])

ax4=fig.add_subplot(514,sharex=ax1)
ax4.plot(T,logK14,color='darkviolet',linewidth=3)
ax4.plot(aa,bb,ls='--',color='k')
ax4.set_ylabel('log$K_{14}$',fontsize=16)
ax4.tick_params(which='both',labelsize=12,pad=5)
ax4.set_xlim(273,323)
ax4.set_ylim(90,115)
ax4.set_yticks([90,95,100,105,110,115])

ax5=fig.add_subplot(515,sharex=ax1)
ax5.plot(T,logK15,color='fuchsia',linewidth=3)
ax5.plot(aa,bb,ls='--',color='k')
ax5.set_ylabel('log$K_{15}$',fontsize=16)
ax5.tick_params(which='both',labelsize=12,pad=5)
ax5.set_xlim(273,323)
ax5.set_ylim(35,45)
ax5.set_xlabel('Temperature (K)',fontsize=16)
ax5.set_yticks([35,37,39,41,43,45])

plt.show()
plt.close()




########################   TEMPERATURE DEPENDENCE OF K's #######################
###### PARAMETERS #########
#Note: couldn't do this for HNO2, because couldn't find delta H of formation for NO2-

pKa_HNO3=-1.3
pKa_NH4=9.24

Ka_HNO3=10.**(-1.0*pKa_HNO3)
Ka_NH4=10.**(-1.0*pKa_NH4)



dHrxn_HNO3=-32.5
dHrxn_NH4=86.9



lnKa_HNO3=np.log(Ka_HNO3)-(dHrxn_HNO3/R*(1.0/T-1.0/T0))
lnKa_NH4=np.log(Ka_NH4)-(dHrxn_NH4/R*(1.0/T-1.0/T0))

Ka_HNO3=2.7182818**lnKa_HNO3
Ka_NH4=2.7182818**lnKa_NH4

pKa_HNO3=-1.0*np.log10(Ka_HNO3)
pKa_NH4=-1.0*np.log10(Ka_NH4)

print(pKa_HNO3)
print(pKa_NH4)

plt.figure()
plt.plot(T,pKa_HNO3,color='navy',label='HNO3-',linewidth=3)
plt.plot(T,pKa_NH4,color='teal',label='NH4+',linewidth=3)
plt.xlabel('Temperature (K)',fontsize=16)
plt.ylabel('pKa',fontsize=16)
plt.xlim(273,323)
plt.tick_params(which='both',labelsize=14,pad=5)
plt.legend(loc='best')
plt.show()
plt.close()




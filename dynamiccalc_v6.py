"""
Reactions encoded

R0: NO2- + hv + H2O --> NO + OH +OH- (net reaction; see  Mack+1999, Carpenter+2015)
R1: NO3- + hv --> NO2- + (1/2)O2 (net reaction; see Mack+1999, Carpenter+2015)
R2: NO2- + Fe(2+) --> nitrogenous gas + Fe(III) (Jones+2015, Buchwald+2016, Grabb+2017)
R3: NO3- + Fe(2+) --> nitrogenous gas + Fe(III) (Scaling of R2, based on Samarkin+2010, Ottley+1997)
R4: NO2- + 6 Fe(2+) + 7H+ --> 6 Fe(III)+ 2H2O + NH3 (Summers & Chang 1993)
R5: NO2- + NH4- --> N2 + 2H2O (Duc+2003)
R6: NOx --(Vents)---> nitrogenous gas, NH3 (see Laneuville+2018, Wong+2017)
"""


import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.integrate
import scipy.optimize
from scipy import interpolate as interp
from matplotlib.pyplot import cm
from scipy.optimize import fsolve


M2cgs=(6.02e23)*1.e-3 #mol/L * particles/mol * L/cm**3 = molecules cm**-3
cgs2M=M2cgs**-1 #molecules cm**-3 to mol/L
R=1.987e-3# kcal K**-1 mol**-1

######################################
###Planet parameters
######################################

loss_fraction_k0=0.2 #Zafiriou+1979a report 20-100% of NO formed from photolysis in modern seawater is not recombined to form nitrite. So this parameter can take values 0.2-1. 
photic_depth_no3=(5.e2)*(2./3.) #cm. photic depth of no3 photolysis, scaled from equatorial value by 2/3 to global mean value. (Zafiriou+1979b, Cronin+2014)
photic_depth_no2=(10.e2)*(2./3.) #cm. photic depth of no2 photolysis, scaled from equatorial value by 2/3 to global mean value. (Zafiriou+1979a, Cronin+2014)
photic_fraction=1. #Fraction of ocean area which dominates NOx photochemical reprocessing. ~0.01 in modern ocean due to biology, ~1. in prebiotic ocean due to lack of biology

ocean_depth_mean_modern=3.8e5 #mean depth of modern ocean, in cm. CRC Handbook, pp.14-11
lake_depth=1.e1 #depth of fiducial lake, 10 cm (~approx that of Don Juan Pond).
lake_catchement_ratio=100. #Ratio of lake/pond catchement area to surface area (drainage ratio). From Davies+2008, can reasonably vary from 10-500.

nox_input_low=2.4e5 #NOx production; molecules/cm2/s. From Wong+2017; value is for 0.1-bar case, is minimum. 
nox_input_high=6.5e8 #NOx production; molecules/cm2/s. From Wong+2017; value is for 1-bar case, is maximum. 
nox_input_max=6.5e9 #Paul's calculations suggest an OOM enhancement possible from what Wong+2017 calculate.
frac_no3m=0.8 #Fraction of NOx that is delivered as NO3- (vs. NO2-). Via Summers+2007. 

conc_nh3=6.e-7#Concentration of NH3 in natural waters; M; from solubility considerations and upper limits on pNH3 from Henry's law.
pH_nh3=6.5 #pH at which the reaction is taking place, low pH should be better for fixed [NH3].


######################################
###Form functions to calculate reaction rates
######################################

######
#Rate Constants
######

###Photochemical reduction of NOx. Factor of 0.5 accounts for attenuation following Zafiriou.

k0_min=0.5*0.6e-6 #s**-1; Scaled to 0C from median value for modern seawater from Zafiriou+1979a. 
k0_max=0.5*2.4e-6 #s**-1; Scaled to 50C from median value for modern seawater from Zafiriou+1979a.
#NOTE that this process only works in environments where there are enough OH scavengers to outcompete nitrite, like the ocean. In environments with ow [OH scavengers], nitrite will react with OH and be regenerated; in the ocean, Br- prevents recombination, bicarbonate and carbonate should too (Zafiriou+1974). 

k1_min=0.5*1.1e-8 #s**-1;  Scaled to 0C from median value for modern seawater from Zafiriou+1979b.
k1_max=0.5*4.6e-8 #s**-1;  Scaled to 50C from median value for modern seawater from Zafiriou+1979b.

###Reduction of NOx to gas by Fe(II)
k2_min=2.e-6 #M**-1 s**-1; minimum rate; derived from Grabb+2017, for pH~6.5, 0C, [Fe(2+)]>[NO2-]
k2_max=9.e-2 #M**-1 s**-1; maximum rate; derived from Buchwald+2016, for pH=8, 50C, [Fe(2+)]>[NO2-]

##########k2_min=2.e-1*0.08 #M**-2 s**-1; FOR SECOND ORDER
##########k2_max=7.e1*9 #M**-2 s**-1; FOR SECOND ORDER

k3_min=0.# M**-1 s**-1; Picardal+2012
k3_max=9.e-4  #M**-1 s**-1; Petersen+1979, Ottley+1997



###Reduction of NOx to NH3 by Fe(II)
k4=4.2e-5 #M**-1.8 s**-1. From Summers & Chang 1993. Value for T=40C, pH=7.6 (maximum reaction rate measured, assumes pH=7.9, T=25C kinetics).


###Anammox of NO2- with NH4-
k5=6e5###2.6e16*np.exp(-15.7/(R*323.)) #M**-2 s**-1. From Duc+2003 for 25C=298K; not stated to be valid for reactant concentrations <0.05M, but Laneuville+2018 use them so for the time being we use them as well. 

###Destruction at hydrothermal vents
k6_min=8.e-17 #s**-1; minimum rate; from values in Wong+2017
k6_max=1.e-14 #s**-1; maximum rate; from value in Laneuville+2018


######
###Reactions
######

def reac0_no2m_photo(k, conc_NO2m, net_loss):
	"""
	Input concentrations must be in M
	Output reaction rate is units: M/s.
	
	net_loss: Zafiriou+1979 show that 20-100% of photolyzed NO2- will NOT reform in Earth's ocean, and will instead be lost to NO (g). This would not be the case in lakes. 
	
	Note: modern Earth ocean constant used; if primitive 300-370 nm flux higher, this would be higher. 
	"""
	
	
	rate=net_loss*k*conc_NO2m
	return rate


def reac1_no3m_photo(k, conc_NO3m):
	"""
	Input concentrations must be in M
	Output reaction rate is units: M/s.
	
	Reaction converts nitrate to nitrite
	
	Note: modern Earth ocean constant used; if primitive 300-370 nm flux higher, this would be higher. 
	"""
	rate=k*conc_NO3m
	return rate


def reac2_no2m_fe(k, conc_NO2m, conc_Fe2p):
	"""
	Input concentrations must be in M
	Output reaction rate is for NO2- removal, which is twice that of N2O production. Units: M/s.
	"""
	k=np.ones(np.shape(conc_NO2m))*k
	k[conc_NO2m>conc_Fe2p]*=0.1

	rate=k*(conc_NO2m * conc_Fe2p)
	
	return rate
#pdb.set_trace()

def reac3_no3m_fe(k, conc_NO3m, conc_Fe2p):
	"""
	Input concentrations must be in M
	Output reaction rate is for NO3- removal, which is twice that of N2O production. Units: M/s.
	"""
	
	rate=k*(conc_NO3m * conc_Fe2p)
	return rate


def reac4_no2m_fe_nh3(k, conc_NO2m, conc_Fe2p):
	"""
	Input concentrations must be in M
	Output reaction rate is for NH3 production, which is the same as for NO2- removal. Units: M/s
	"""
	
	rate=k*(conc_NO2m)*(conc_Fe2p**1.8)
	return rate


def reac5_hno2_nh3(k, conc_NO2m, conc_nh3, pH):
	"""
	Input concentrations must be in M
	Output reaction rate is for HNO2 removal, which is assumed to be equal to the rate of NO2- removal due to instantaneous acidification reaction. Units: M/s.
	
	Note: reaction kinematics not known to be valid at low concentrations.
	"""
	
	conc_HNO2=conc_NO2m*10.**(3.25-pH)
	rate=k*(conc_HNO2**2. * conc_nh3)
	return rate

def reac6_vents(k, conc_NOx):
	"""
	Input concentrations must be in M
	Output reaction rate is units: M/s.
	"""
	rate=k*conc_NOx
	return rate

######
###Column-integrated reaction rates. Note conversion to from M s**-1 to cm**-2 s**-1 in all cases.
######
#Photochemical reactions
def reac0_no2m_photo_colint(k, conc_no2m, net_loss, photic_depth_no2, photic_fraction):
	return reac0_no2m_photo(k, conc_no2m, net_loss)*M2cgs*photic_depth_no2*photic_fraction #Note conversion to units of molecules cm**-2 s**-1. 

def reac1_no3m_photo_colint(k, conc_no3m, photic_depth_no3, photic_fraction):
	return reac1_no3m_photo(k, conc_no3m)*M2cgs*photic_depth_no3*photic_fraction #Note conversion to units of molecules cm**-2 s**-1.

#Thermal reactions
def reac2_no2m_fe_colint(k, conc_no2m, conc_fe, total_depth):
	return reac2_no2m_fe(k, conc_no2m, conc_fe)*M2cgs*total_depth #Note conversion to units of molecules cm**-2 s**-1. 

def reac3_no3m_fe_colint(k, conc_no3m, conc_fe, total_depth):
	return reac3_no3m_fe(k, conc_no3m, conc_fe)*M2cgs*total_depth #Note conversion to units of molecules cm**-2 s**-1. 

def reac4_no2m_fe_colint(k, conc_no2m, conc_fe, total_depth):
	return reac4_no2m_fe_nh3(k, conc_no2m, conc_fe)*M2cgs*total_depth #

def reac5_hno2_nh3_colint(k, conc_no2m, conc_nh3, pH, total_depth):
	return reac5_hno2_nh3(k, conc_no2m, conc_nh3, pH)*M2cgs*total_depth #Note conversion to units of molecules cm**-2 s**-1. 

def reac6_vents_colint(k, conc_NOx, total_depth):
	return reac6_vents(k, conc_NOx)*M2cgs*total_depth #Note conversion to units of molecules cm**-2 s**-1.

######################################
###Plot Ocean Case
######################################
###set up calculation
conc_fe2_range=np.array([10.e-6, 600.e-6]) #M
conc_no2m_range=10.**(np.arange(-9., -0., step=1.)) #M
conc_no3m_range=10.**(np.arange(-9., -0., step=1.)) #M

###NOx supply rate
supply_rate_no2_low=conc_no2m_range*0.+nox_input_low
supply_rate_no2_high=conc_no2m_range*0.+nox_input_high

supply_rate_no3_low=conc_no3m_range*0.+nox_input_low
supply_rate_no3_high=conc_no3m_range*0.+nox_input_high


###initialize variables to hold loss rates. Dimensions: (1 dimension for each reaction) x (lower/upper limit) x (number of [NO2-] or [NO3-] concentrations to consider) x (number of [Fe2+] to consider). Most of this array will have dimension zero; they are there just because numpy doesn't do ragged arrays very well. 
loss_rates_no2m=np.zeros((5, 2, len(conc_no2m_range), len(conc_fe2_range))) #reactions 0, 2, 4, 5, 6
loss_rates_no3m=np.zeros((3, 2, len(conc_no3m_range), len(conc_fe2_range))) #reactions 1,3, 6

##Reactions 0, 1 (photochem)
#reduction of nitrite to no by UV
loss_rates_no2m[0, 0, :, 0]=reac0_no2m_photo_colint(k0_min, conc_no2m_range, loss_fraction_k0, photic_depth_no2, photic_fraction) #min
loss_rates_no2m[0, 1, :, 0]=reac0_no2m_photo_colint(k0_max, conc_no2m_range, loss_fraction_k0, photic_depth_no2, photic_fraction) #max

#reduction of nitrate to nitrite by UV
loss_rates_no3m[0, 0,  :, 0]=reac1_no3m_photo_colint(k1_min, conc_no3m_range, photic_depth_no3, photic_fraction) #min
loss_rates_no3m[0, 1,  :, 0]=reac1_no3m_photo_colint(k1_max, conc_no3m_range, photic_depth_no3, photic_fraction) #max

##Reactions 2,3 (NOx reduction by Fe to nitrogenous gas)
for fe_ind in range(0, len(conc_fe2_range)):
	conc_fe=conc_fe2_range[fe_ind]
	
	loss_rates_no2m[1, 0, :, fe_ind]=reac2_no2m_fe_colint(k2_min, conc_no2m_range, conc_fe, ocean_depth_mean_modern) #min
	loss_rates_no2m[1, 1, :, fe_ind]=reac2_no2m_fe_colint(k2_max, conc_no2m_range, conc_fe, ocean_depth_mean_modern) #max
	
	loss_rates_no3m[1, 0, :, fe_ind]=reac3_no3m_fe_colint(k3_min, conc_no3m_range, conc_fe, ocean_depth_mean_modern) #min
	loss_rates_no3m[1, 1, :, fe_ind]=reac3_no3m_fe_colint(k3_max, conc_no3m_range, conc_fe, ocean_depth_mean_modern) #max

##Reaction 4: nitrite reduction to nh3 by fe
loss_rates_no2m[2, 0, :, 0]=reac4_no2m_fe_colint(k4, conc_no2m_range, 600.e-6, ocean_depth_mean_modern) #reduction of no2- to nh3 by fe++; upper bound

##Reaction 5: nitrite reduction to nh3 by fe
loss_rates_no2m[3, 0, :, 0]=reac5_hno2_nh3_colint(k5, conc_no2m_range,  conc_nh3,pH_nh3, ocean_depth_mean_modern) #anammox of nitrite and nh3

##Reaction 6: Vents
loss_rates_no2m[4, 0, :, 0]=reac6_vents_colint(k6_min, conc_no2m_range, ocean_depth_mean_modern) #Reduction by vents; min
loss_rates_no2m[4, 1, :, 0]=reac6_vents_colint(k6_max, conc_no2m_range, ocean_depth_mean_modern) #Reduction by vents; max

loss_rates_no3m[2, 0, :, 0]=reac6_vents_colint(k6_min, conc_no3m_range, ocean_depth_mean_modern) #Reduction by vents; min
loss_rates_no3m[2, 1, :, 0]=reac6_vents_colint(k6_max, conc_no3m_range, ocean_depth_mean_modern) #Reduction by vents; max


###Make plot
markersizeval=4.
plt.rcParams.update({'font.size': 14})

fig, (ax0, ax1)=plt.subplots(2, figsize=(8.5, 10.0), sharex=False, sharey=False)

colors=np.array(['red','blue']) 
hatch_list=np.array(['//','\\'])
label_list=np.array([r'NO$_X^-$ + 10$\mu$M Fe$^{2+}$-->N$_2$O,N$_2$', r'NO$_X^-$- + 600$\mu$M Fe$^{2+}$-->N$_2$O,N$_2$'])

##Nitrite Plot
ax0.set_title(r'NO$_2^-$')

#Supply
ax0.plot(conc_no2m_range, supply_rate_no2_low, color='black', linestyle='--', linewidth=2, label=r'$\phi_{NO_{X}}$ (Low)')
ax0.plot(conc_no2m_range, supply_rate_no2_high, color='black', linestyle='-', linewidth=2, label=r'$\phi_{NO_{X}}$ (High)')


#Reduction by Fe to gas
for fe_ind in range(0, len(conc_fe2_range)):
	ax0.fill_between(conc_no2m_range, loss_rates_no2m[1, 0, :, fe_ind], loss_rates_no2m[1, 1, :, fe_ind], facecolor=colors[fe_ind], alpha=0.2, edgecolor='black', hatch=hatch_list[fe_ind], label=label_list[fe_ind])

#Photochemistry
ax0.fill_between(conc_no2m_range, loss_rates_no2m[0, 0, :, 0], loss_rates_no2m[0, 1, :, 0], facecolor='darkgrey', alpha=0.9, edgecolor='black', label='Photoreduction')


#Other reduction
ax0.plot(conc_no2m_range, loss_rates_no2m[2, 0, :, 0], color='brown', linestyle='--', label=r'NO$_2^-$ + Fe$^{2+}$-->NH$_3$')
ax0.plot(conc_no2m_range, loss_rates_no2m[3, 0, :, 0], color='orange', linestyle='--', label='Anammox')

#Vents
ax0.fill_between(conc_no2m_range, loss_rates_no2m[4, 0, :, 0], loss_rates_no2m[4, 1, :, 0], facecolor='darkgreen', alpha=0.8, edgecolor='black', label='Vents')

###Nitrate Plot
ax1.set_title(r'NO$_3^-$')

#Supply
ax1.plot(conc_no3m_range, supply_rate_no3_low, color='black', linewidth=2, linestyle='--')
ax1.plot(conc_no3m_range, supply_rate_no3_high, color='black', linewidth=2, linestyle='-')

#Reduction by Fe to gas
for fe_ind in range(0, len(conc_fe2_range)):
	ax1.fill_between(conc_no3m_range, loss_rates_no3m[1, 0, :, fe_ind], loss_rates_no3m[1, 1, :, fe_ind], facecolor=colors[fe_ind], alpha=0.2, edgecolor='black', hatch=hatch_list[fe_ind], label=label_list[fe_ind])

#Photochemistry
ax1.fill_between(conc_no3m_range, loss_rates_no3m[0, 0, :, 0], loss_rates_no3m[0, 1, :, 0], facecolor='darkgrey', alpha=0.9, edgecolor='black')

#Vents
ax1.fill_between(conc_no3m_range, loss_rates_no3m[2, 0, :, 0], loss_rates_no3m[2, 1, :, 0], facecolor='darkgreen', alpha=0.8, label='Vents')


ax0.set_yscale('log')
ax0.set_ylabel(r'NO$_2^-$ Prod/Loss (cm$^{-2}$ s$^{-1}$)')
ax0.set_ylim([1.e4, 1.e12])
ax0.set_xscale('log')
ax0.set_xlim([min(conc_no2m_range), max(conc_no2m_range)])
ax0.set_xlabel(r'[NO$_2^-$] (M)')

ax1.set_yscale('log')
ax1.set_ylabel(r'NO$_3^-$ Prod/Loss (cm$^{-2}$ s$^{-1}$)')
ax1.set_ylim([1.e4, 1.e12])
ax1.set_xscale('log')
ax1.set_xlim([min(conc_no3m_range), max(conc_no3m_range)])
ax1.set_xlabel(r'[NO$_3^-$] (M)')

ax0.legend(bbox_to_anchor=[-0.13, 1.11, 1.15, .15],loc='lower left', ncol=3, mode='expand', borderaxespad=0., fontsize=13)
plt.tight_layout(rect=(0,0,1,0.88))
plt.subplots_adjust(wspace=0., hspace=0.3)

plt.savefig('./Plots/important_rxns_ocean_focused_v6.pdf', orientation='portrait',papertype='letter', format='pdf')

########################################
#####Plot Lake Case
########################################
###set up calculation
conc_fe2_range=np.array([10.e-6, 600.e-6]) #M
conc_no2m_range=10.**(np.arange(-9., -0., step=1.)) #M
conc_no3m_range=10.**(np.arange(-9., -0., step=1.)) #M

###NOx supply rate
supply_rate_no2_low=conc_no2m_range*0.+nox_input_low
supply_rate_no2_high=conc_no2m_range*0.+nox_input_high
supply_rate_no2_low_catchmentarea=supply_rate_no2_low*lake_catchement_ratio
supply_rate_no2_high_catchmentarea=supply_rate_no2_high*lake_catchement_ratio

supply_rate_no3_low=conc_no3m_range*0.+nox_input_low
supply_rate_no3_high=conc_no3m_range*0.+nox_input_high
supply_rate_no3_low_catchmentarea=supply_rate_no3_low*lake_catchement_ratio
supply_rate_no3_high_catchmentarea=supply_rate_no3_high*lake_catchement_ratio

###initialize variables to hold loss rates. Dimensions: (1 dimension for each reaction) x (lower/upper limit) x (number of [NO2-] or [NO3-] concentrations to consider) x (number of [Fe2+] to consider). Most of this array will have dimension zero; they are there just because numpy doesn't do ragged arrays very well. 
loss_rates_no2m=np.zeros((5, 2, len(conc_no2m_range), len(conc_fe2_range))) #reactions 0, 2, 4, 5, 6
loss_rates_no3m=np.zeros((3, 2, len(conc_no3m_range), len(conc_fe2_range))) #reactions 1,3, 6

##Reactions 0, 1 (photochem)
#reduction of nitrite to no by UV
loss_rates_no2m[0, 0, :, 0]=reac0_no2m_photo_colint(k0_min, conc_no2m_range, loss_fraction_k0, np.minimum(lake_depth, photic_depth_no2), 1.) #min
loss_rates_no2m[0, 1, :, 0]=reac0_no2m_photo_colint(k0_max, conc_no2m_range, loss_fraction_k0, np.minimum(lake_depth, photic_depth_no2), 1.) #max

#reduction of nitrate to nitrite by UV
loss_rates_no3m[0, 0,  :, 0]=reac1_no3m_photo_colint(k1_min, conc_no3m_range, np.minimum(lake_depth, photic_depth_no3), 1.) #min
loss_rates_no3m[0, 1,  :, 0]=reac1_no3m_photo_colint(k1_max, conc_no3m_range, np.minimum(lake_depth, photic_depth_no3), 1.) #max

##Reactions 2,3 (NOx reduction by Fe to nitrogenous gas)
for fe_ind in range(0, len(conc_fe2_range)):
	conc_fe=conc_fe2_range[fe_ind]
	
	loss_rates_no2m[1, 0, :, fe_ind]=reac2_no2m_fe_colint(k2_min, conc_no2m_range, conc_fe, lake_depth) #min
	loss_rates_no2m[1, 1, :, fe_ind]=reac2_no2m_fe_colint(k2_max, conc_no2m_range, conc_fe, lake_depth) #max
	
	loss_rates_no3m[1, 0, :, fe_ind]=reac3_no3m_fe_colint(k3_min, conc_no3m_range, conc_fe, lake_depth) #min
	loss_rates_no3m[1, 1, :, fe_ind]=reac3_no3m_fe_colint(k3_max, conc_no3m_range, conc_fe, lake_depth) #max

##Reaction 4: nitrite reduction to nh3 by fe
loss_rates_no2m[2, 0, :, 0]=reac4_no2m_fe_colint(k4, conc_no2m_range, 600.e-6, lake_depth) #reduction of no2- to nh3 by fe++; upper bound

##Reaction 5: nitrite reduction to nh3 by fe
loss_rates_no2m[3, 0, :, 0]=reac5_hno2_nh3_colint(k5, conc_no2m_range,  conc_nh3,pH_nh3, lake_depth) #anammox of nitrite and nh3

###Make plot
markersizeval=4.
plt.rcParams.update({'font.size': 14})
fig, (ax0, ax1)=plt.subplots(2, figsize=(8.5, 10.0), sharex=False, sharey=False)

colors=np.array(['red','blue']) #'darkgreen', 'indigo', 'orange', 'brown', 
hatch_list=np.array(['//','\\'])
label_list=np.array([r'NO$_X^-$ + 10$\mu$M Fe$^{2+}$-->N$_2$O,N$_2$', r'NO$_X^-$ + 600$\mu$M Fe$^{2+}$-->N$_2$O,N$_2$'])

##Nitrite Plot
ax0.set_title(r'NO$_2^-$')

#Supply
ax0.plot(conc_no2m_range, supply_rate_no2_low, color='black', linestyle='--', linewidth=2, label=r'$\phi_{NO_{X}}$ (Low)')
ax0.plot(conc_no2m_range, supply_rate_no2_high, color='black', linestyle='-', linewidth=2, label=r'$\phi_{NO_{X}}$ (High)')
ax0.plot(conc_no2m_range, supply_rate_no2_low_catchmentarea, color='hotpink', linestyle='--', linewidth=2, label=r'$\phi_{NO_{X}}$ (Low, DR=100)')
ax0.plot(conc_no2m_range, supply_rate_no2_high_catchmentarea, color='hotpink', linestyle='-', linewidth=2, label=r'$\phi_{NO_{X}}$ (High, DR=100)')



#Reduction by Fe to gas
for fe_ind in range(0, len(conc_fe2_range)):
	ax0.fill_between(conc_no2m_range, loss_rates_no2m[1, 0, :, fe_ind], loss_rates_no2m[1, 1, :, fe_ind], facecolor=colors[fe_ind], alpha=0.2, edgecolor='black', hatch=hatch_list[fe_ind], label=label_list[fe_ind])

#Photochemistry
ax0.fill_between(conc_no2m_range, loss_rates_no2m[0, 0, :, 0], loss_rates_no2m[0, 1, :, 0], facecolor='darkgrey', alpha=0.9, edgecolor='black', label='Photoreduction')

#Other reduction
ax0.plot(conc_no2m_range, loss_rates_no2m[2, 0, :, 0], color='brown', linestyle='--', label=r'NO$_2^-$ + Fe$^{2+}$-->NH$_3$')
ax0.plot(conc_no2m_range, loss_rates_no2m[3, 0, :, 0], color='orange', linestyle='--', label='Anammox')

###Nitrate Plot
ax1.set_title(r'NO$_3^-$')

#Supply
ax1.plot(conc_no3m_range, supply_rate_no3_low, color='black', linestyle='--', linewidth=2)
ax1.plot(conc_no3m_range, supply_rate_no3_high, color='black', linestyle='-', linewidth=2)
ax1.plot(conc_no2m_range, supply_rate_no3_low_catchmentarea, color='hotpink', linestyle='--', linewidth=2)
ax1.plot(conc_no2m_range, supply_rate_no3_high_catchmentarea, color='hotpink', linestyle='-', linewidth=2)

#Reduction by Fe to gas
for fe_ind in range(0, len(conc_fe2_range)):
	ax1.fill_between(conc_no3m_range, loss_rates_no3m[1, 0, :, fe_ind], loss_rates_no3m[1, 1, :, fe_ind], facecolor=colors[fe_ind], alpha=0.2, edgecolor='black', hatch=hatch_list[fe_ind], label=label_list[fe_ind])

#Photochemistry
ax1.fill_between(conc_no3m_range, loss_rates_no3m[0, 0, :, 0], loss_rates_no3m[0, 1, :, 0], facecolor='darkgrey', alpha=0.9, edgecolor='black')

ax0.set_yscale('log')
ax0.set_ylabel(r'NO$_2^-$ Prod/Loss (cm$^{-2}$ s$^{-1}$)')
ax0.set_ylim([1.e4, 1.e12])
ax0.set_xscale('log')
ax0.set_xlim([min(conc_no2m_range), max(conc_no2m_range)])
ax0.set_xlabel(r'[NO$_2^-$] (M)')

ax1.set_yscale('log')
ax1.set_ylabel(r'NO$_3^-$ Prod/Loss (cm$^{-2}$ s$^{-1}$)')
ax1.set_ylim([1.e4, 1.e12])
ax1.set_xscale('log')
ax1.set_xlim([min(conc_no3m_range), max(conc_no3m_range)])
ax1.set_xlabel(r'[NO$_3^-$] (M)')

ax0.legend(bbox_to_anchor=[-0.13, 1.11, 1.18, .15],loc='lower left', ncol=3, mode='expand', borderaxespad=0., fontsize=13)
plt.tight_layout(rect=(0,0,1,0.88))
plt.subplots_adjust(wspace=0., hspace=0.3)

plt.savefig('./Plots/important_rxns_pond_focused_v6.pdf', orientation='portrait',papertype='letter', format='pdf')

######################################
###Set up system
######################################

def equations(p, S_nox, conc_fe, case):
	
	conc_no2m, conc_no3m=p#10.**(p) #concentration of NO2-, #concentration of NO3-, M
	
	conc_no2m=np.array([conc_no2m])
	conc_no3m=np.array([conc_no3m])

	result=np.ones(np.shape(p)) #initialize vector to hold residual (the output)

	S_no2m=S_nox*(1.-frac_no3m) #supply of NO2-, cm**-2 s**-1
	S_no3m=S_nox*frac_no3m #supply of NO3-, cm**-2 s**-1
	
	
	if case==0: #Ocean, minimum NOx consumption (max NOx accumulation) case
		result[0]=S_no2m - (reac0_no2m_photo_colint(k0_min, conc_no2m, loss_fraction_k0, photic_depth_no2, photic_fraction) + reac2_no2m_fe_colint(k2_min, conc_no2m, conc_fe, ocean_depth_mean_modern)) #balance equation for NO2-
		
		result[1]=S_no3m + reac0_no2m_photo_colint(k0_min, conc_no2m, loss_fraction_k0, photic_depth_no2, photic_fraction) - (reac1_no3m_photo_colint(k1_min, conc_no3m, photic_depth_no3, photic_fraction) + reac3_no3m_fe_colint(k3_min, conc_no3m, conc_fe, ocean_depth_mean_modern))
	if case==1: #Ocean, maximum NOx consumption (min NOx accumulation) case
		result[0]=S_no2m - (reac0_no2m_photo_colint(k0_max, conc_no2m, loss_fraction_k0, photic_depth_no2, photic_fraction) + reac2_no2m_fe_colint(k2_max, conc_no2m, conc_fe, ocean_depth_mean_modern)) #balance equation for NO2-
		
		result[1]=S_no3m + reac0_no2m_photo_colint(k0_max, conc_no2m, loss_fraction_k0, photic_depth_no2, photic_fraction) - (reac1_no3m_photo_colint(k1_max, conc_no3m, photic_depth_no3, photic_fraction) + reac3_no3m_fe_colint(k3_max, conc_no3m, conc_fe, ocean_depth_mean_modern))
	if case==2: #Lake, minimum NOx consumption (max NOx accumulation) case
		result[0]=S_no2m*lake_catchement_ratio - (reac0_no2m_photo_colint(k0_min, conc_no2m, loss_fraction_k0, np.minimum(lake_depth, photic_depth_no2), 1.) + reac2_no2m_fe_colint(k2_min, conc_no2m, conc_fe, lake_depth)) #balance equation for NO2-
		
		result[1]=S_no3m*lake_catchement_ratio + reac0_no2m_photo_colint(k0_min, conc_no2m, loss_fraction_k0, np.minimum(lake_depth, photic_depth_no2), 1.) - (reac1_no3m_photo_colint(k1_min, conc_no3m, np.minimum(lake_depth, photic_depth_no3), 1.) + reac3_no3m_fe_colint(k3_min, conc_no3m, conc_fe, lake_depth))
	if case==3: #Lake, maximum NOx consumption (min NOx accumulation) case
		result[0]=S_no2m*lake_catchement_ratio - (reac0_no2m_photo_colint(k0_max, conc_no2m, loss_fraction_k0, np.minimum(lake_depth, photic_depth_no2), 1.) + reac2_no2m_fe_colint(k2_max, conc_no2m, conc_fe, lake_depth)) #balance equation for NO2-
		
		result[1]=S_no3m*lake_catchement_ratio + reac0_no2m_photo_colint(k0_max, conc_no2m, loss_fraction_k0, np.minimum(lake_depth, photic_depth_no2), 1.) - (reac1_no3m_photo_colint(k1_max, conc_no3m, np.minimum(lake_depth, photic_depth_no3), 1.) + reac3_no3m_fe_colint(k3_max, conc_no3m, conc_fe, lake_depth)) #
	
	#pdb.set_trace()
	return result

######################################
###Solve system
######################################

initial_guess=np.array([1.e-7, 1.e-7]) #initial guess for NO2- (M), NO3- (M) concentrations. Just needs to be the correct scale.

nox_input_list=10.**np.linspace(np.log10(nox_input_low), np.log10(nox_input_max), num=15) #cm**-2 s**-1; flux of NOx from atmosphere.
fe_input_list=np.array([10.e-6, 600.e-6]) #M; [Fe2+ concentration]

no2m_conc_list=np.zeros((len(nox_input_list),len(fe_input_list), 4)) #initialize variable to hold [NO2-]
no3m_conc_list=np.zeros((len(nox_input_list),len(fe_input_list), 4)) #initialize variable to hold [NO3-]
###resid_list=np.zeros(len(nox_input_list),4,2) #initialize variable to hold residuals

for ind in range(0, len(nox_input_list)):
	nox_input=nox_input_list[ind]
	
	for ind2 in range(0, len(fe_input_list)):
		fe_input=fe_input_list[ind2]
	
		#Case 0: ocean, minimum reduction (maximum accumulation) case
		no2m_conc_list[ind, ind2, 0], no3m_conc_list[ind, ind2, 0]=fsolve(equations, initial_guess,args=(np.array([nox_input]),np.array([fe_input]), 0), xtol=1.0E-3, maxfev=1000*(len(initial_guess)+1))

		#Case 1: ocean, maximum reduction (minimum accumulation) case
		no2m_conc_list[ind, ind2, 1], no3m_conc_list[ind, ind2, 1]=fsolve(equations, initial_guess,args=(np.array([nox_input]),np.array([fe_input]), 1), xtol=1.0E-3, maxfev=1000*(len(initial_guess)+1))
		
		#Case 2: lake, minimum reduction (maximum accumulation) case
		no2m_conc_list[ind, ind2, 2], no3m_conc_list[ind, ind2, 2]=fsolve(equations, initial_guess,args=(np.array([nox_input]),np.array([fe_input]), 2), xtol=1.0E-3, maxfev=1000*(len(initial_guess)+1))

		#Case 3: lake, minimum reduction (maximum accumulation) case
		no2m_conc_list[ind, ind2, 3], no3m_conc_list[ind, ind2, 3]=fsolve(equations, initial_guess,args=(np.array([nox_input]), np.array([fe_input]), 3), xtol=1.0E-3, maxfev=1000*(len(initial_guess)+1))
		
		###if ind==(len(nox_input_list)-1):
			###print equations(np.array([no2m_conc_list[ind, ind2, 3], no3m_conc_list[ind, ind2, 3]]), np.array([nox_input]), np.array([fe_input]), 3)/nox_input
			###pdb.set_trace()

#######################################
####Plot
#######################################
###Make plot
markersizeval=4.
plt.rcParams.update({'font.size': 14})
fig, ax=plt.subplots(2,2, figsize=(8, 8), sharex=True, sharey=True)

#Demarcate range of fluxes from Wong+2017
for ind1 in range(0,2):
	for ind2 in range(0,2):
		ax[ind1, ind2].tick_params(labelsize=16)
		ax[ind1, ind2].axhline(1.e-6, color='black', linestyle='--', linewidth=2)
		ax[ind1, ind2].axvspan(nox_input_low, nox_input_high, color='lightgrey', alpha=0.3) #Range of phi_nox from Wong+2017


#Ocean, NO2-
ax[0,0].fill_between(nox_input_list, no2m_conc_list[:, 0, 0], no2m_conc_list[:, 0, 1], facecolor='red', alpha=0.6, edgecolor='black', label=r'[Fe$^{2+}$]=10 $\mu$M')
ax[0,0].fill_between(nox_input_list, no2m_conc_list[:, 1, 0], no2m_conc_list[:, 1, 1], facecolor='blue', alpha=0.6, edgecolor='black', label=r'[Fe$^{2+}$]=600 $\mu$M')

#Ocean, NO3-
ax[1,0].fill_between(nox_input_list, no3m_conc_list[:, 0, 0], no3m_conc_list[:, 0, 1], facecolor='red', alpha=0.6, edgecolor='black', label=r'[Fe$^{2+}$]=10 $\mu$M')
ax[1,0].fill_between(nox_input_list, no3m_conc_list[:, 1, 0], no3m_conc_list[:, 1, 1], facecolor='blue', alpha=0.6, edgecolor='black', label=r'[Fe$^{2+}$]=600 $\mu$M')

#Lake, NO2-
ax[0,1].fill_between(nox_input_list, no2m_conc_list[:, 0, 2], no2m_conc_list[:, 0, 3], facecolor='red', alpha=0.6, edgecolor='black', label=r'[Fe$^{2+}$]=10 $\mu$M')
ax[0,1].fill_between(nox_input_list, no2m_conc_list[:, 1, 2], no2m_conc_list[:, 1, 3], facecolor='blue', alpha=0.6, edgecolor='black', label=r'[Fe$^{2+}$]=600 $\mu$M')

#Ocean, NO3-
ax[1,1].fill_between(nox_input_list, no3m_conc_list[:, 0, 2], no3m_conc_list[:, 0, 3], facecolor='red', alpha=0.6, edgecolor='black', label=r'[Fe$^{2+}$]=10 $\mu$M')
ax[1,1].fill_between(nox_input_list, no3m_conc_list[:, 1, 2], no3m_conc_list[:, 1, 3], facecolor='blue', alpha=0.6, edgecolor='black', label=r'[Fe$^{2+}$]=600 $\mu$M')

ax[1,1].set_yscale('log')
ax[1,1].set_xscale('log')

ax[0,0].set_title('Ocean')
ax[0,1].set_title('Pond')

ax[0,0].set_ylabel(r'[NO$_2^-$] (M)')
ax[1,0].set_ylabel(r'[NO$_3^-$] (M)')

ax[1,0].set_xlabel(r'$\phi_{NO_{x}}$ (cm$^{-2}$ s$^{-1})$')
ax[1,1].set_xlabel(r'$\phi_{NO_{x}}$ (cm$^{-2}$ s$^{-1})$')

ax[1,1].set_ylim([1.e-17, 1.e-1])
ax[1,1].set_xlim([min(nox_input_list), max(nox_input_list)])

plt.tight_layout(rect=(0,0,1,0.95))
plt.subplots_adjust(wspace=0.1, hspace=0.1)




ax[0,0].legend(bbox_to_anchor=[0, 1.10, 2, .15],loc='lower left', ncol=2, mode='expand', borderaxespad=0., fontsize=16)


plt.savefig('./Plots/conc_system_v6.pdf', orientation='portrait',papertype='letter', format='pdf')


plt.show()

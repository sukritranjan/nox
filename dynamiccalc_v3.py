"""
Reactions encoded

R1: NO2- + 6 Fe(2+) + 7H+ --> 6 Fe(III)+ 2H2O + NH3 (Summers & Chang 1993)
R2: 2NO2- + 6Fe(2+) + 5H2O --> NO2(g) + 2Fe3O4 + 10H+ (Sorensen & Thorling 1991, plus our analysis)
R3: 
R4:
R5: NO3- + hv --> NO2- + (1/2)O2 (net reaction; see Zafiriou+1979, Mack+1999)
"""


import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.integrate
import scipy.optimize
from scipy import interpolate as interp
from matplotlib.pyplot import cm


M2cgs=(6.02e23)*1.e-3 #mol/L * particles/mol * L/cm**3 = molecules cm**-3
cgs2M=M2cgs**-1 #molecules cm**-3 to mol/L

######################################
###Planet parameters
######################################

loss_fraction_k4=0.2 #Zafiriou+1979a report 20-100% of NO formed from photolysis in modern seawater is not recombined to form nitrite. So this parameter can take values 0.2-1. 
photic_depth_no3=5.e2 #cm. photic depth of no3 photolysis in modern ocean in Central South Pacific (Zafiriou+1979b)
photic_depth_no2=10.e2 #cm. photic depth of no2 photolysis in modern ocean in Central South Pacific (Zafiriou+1979a)
photic_fraction=0.01 #Fraction of ocean area which dominates NOx photochemical reprocessing. Zafiriou+1979ab took this as the Central South Pacific, where dynamics puts lots of nitrate/nitrite close to the surface and where there is a lot of Sunlight.

ocean_depth_mean_modern=3.8e5 #mean depth of modern ocean, in cm. CRC Handbook, pp.14-11
lake_depth=1.e1 #depth of fiducial lake, 10 cm (~approx that of Don Juan Lake).
lake_catchement_ratio=100. #Ratio of lake/pond catchement area to surface area. From Davies+2008, can reasonably vary from 10-500.

nox_input_low=2.4e5 #NOx production; molecules/cm2/s. From Wong+2017; value is for 0.1-bar case, is minimum. 
nox_input_high=6.5e8 #NOx production; molecules/cm2/s. From Wong+2017; value is for 1-bar case, is maximum. 

conc_nh3=6.e-7#Concentration of NH3 in natural waters; M; from solubility considerations and upper limits on pNH3 from Henry's law.
pH_nh3=6.5 #pH at which the reaction is taking place, low pH should be better for fixed [NH3].


######################################
###Form functions to calculate reaction rates
######################################

######
###Constants
######

###Thermal loss reactions for NOx
k1=4.2e-5 #M**-1.8 s**-1. From Summers & Chang 1993. Value for T=40C, pH=7.6 (maximum reaction rate measured, assumes pH=7.9, T=25C kinetics).

k2_min=3.e-5 #M**-1 s**-1; derived from Grabb+2017, for pH~6.5, room temperature.
k2_max=1.e-2 #M**-1 s**-`; derived from Buchwald+2016, for pH=8, room temperature. 

k3_min=1.e-2*k2_min  #M**-1 s**-1; via scaling of k2_min by 0.01 (Samarkin+2010), which is consistent with Ottley+1997 at 10C, pH=8.
k3_max=1.e-2*k2_max  #M**-1 s**-1; via scaling of k2_max by 0.01 (Samarkin+2010), which is consistent with Ottley+1997 at 10C, pH=8.

k6=7.9e4 #M**-2 s**-1. From Duc+2003 for 25C; not stated to be valid for reactant concentrations <0.05M, but Laneuville+2018 use them so for the time being we use them as well. 

###Photochemical reactions for NOx
k4=1.2e-6 #s**-1; median value for modern seawater from Zafiriou+1979a. Note that this process only works in environments where there are enough anions to outcompete nitrite for reactions with OH-, like the ocean. In lake environments with few other ions, nitrite will interact wtih OH- and be regenerated; in the ocean, Br- prevents recombination (Zafiriou+XXXX). 

k5=2.3e-8 #s**-1; median value for modern seawater from Zafiriou+1979b.

###Destruction at hydrothermal vents
k_wong=8.e-17 #s**-1; from values in Wong+2017
k_laneuville=1.e-14 #s**-1; from value in Laneuville+2018


######
###Reactions
######

def reac1_no2m_fe(k, conc_NO2m, conc_Fe2p):
	"""
	Input concentrations must be in M
	Output reaction rate is for NH3 production, which is the same as for NO2- removal. Units: M/s
	"""
	
	rate=k*(conc_NO2m)*(conc_Fe2p**1.8)
	return rate

def reac2_no2m_fe(k, conc_NO2m, conc_Fe2p):
	"""
	Input concentrations must be in M
	Output reaction rate is for NO2- removal, which is twice that of N2O production. Units: M/s.
	"""
	
	rate=k*(conc_NO2m * conc_Fe2p)
	
	return rate

def reac3_no3m_fe(k, conc_NO3m, conc_Fe2p):
	"""
	Input concentrations must be in M
	Output reaction rate is for NO3- removal, which is twice that of N2O production. Units: M/s.
	"""
	rate=k*(conc_NO3m * conc_Fe2p)
	return rate

def reac4_no2m_photo(k, conc_NO2m, net_loss):
	"""
	Input concentrations must be in M
	Output reaction rate is units: M/s.
	
	net_loss: Zafiriou+1979 show that 20-100% of photolyzed NO2- will NOT reform in Earth's ocean, and will instead be lost to NO (g). This would not be the case in lakes. 
	
	Note: modern Earth ocean constant used; if primitive 300-370 nm flux higher, this would be higher. 
	"""	
	
	rate=net_loss*k*conc_NO2m
	return rate


def reac5_no3m_photo(k, conc_NO3m):
	"""
	Input concentrations must be in M
	Output reaction rate is units: M/s.
	
	Reaction converts nitrate to nitrite
	
	Note: modern Earth ocean constant used; if primitive 300-370 nm flux higher, this would be higher. 
	"""
	rate=k*conc_NO3m
	return rate


def reac6_hno2_nh3(k, conc_NO2m, conc_nh3, pH):
	"""
	Input concentrations must be in M
	Output reaction rate is for HNO2 removal, which is assumed to be equal to the rate of NO2- removal due to instantaneous acidification reaction. Units: M/s.
	
	Note: reaction kinematics not known to be valid at low concentrations.
	"""
	
	conc_HNO2=conc_NO2m*10.**(3.25-pH)
	
	rate=k*(conc_HNO2**2. * conc_nh3)
	
	return rate

def reac_vents(k, conc_NOx):
	"""
	Input concentrations must be in M
	Output reaction rate is units: M/s.
	"""
	rate=k*conc_NOx
	return rate




######
###Column-integrated reactions
######

def reac1_no2m_fe_colint(k, conc_no2m, conc_fe, total_depth):
	return reac1_no2m_fe(k, conc_no2m, conc_fe)*M2cgs*total_depth #Note conversion to units of molecules cm**-2 s**-1. 

def reac2_no2m_fe_colint(k, conc_no2m, conc_fe, total_depth):
	return reac2_no2m_fe(k, conc_no2m, conc_fe)*M2cgs*total_depth #Note conversion to units of molecules cm**-2 s**-1. 

def reac3_no3m_fe_colint(k, conc_no3m, conc_fe, total_depth):
	return reac3_no3m_fe(k, conc_no3m, conc_fe)*M2cgs*total_depth #Note conversion to units of molecules cm**-2 s**-1. 

def reac4_no2m_photo_colint(k, conc_no2m, net_loss, photic_depth_no2, photic_fraction):
	return reac4_no2m_photo(k, conc_no2m, net_loss)*M2cgs*photic_depth_no2*photic_fraction #Note conversion to units of molecules cm**-2 s**-1. 

def reac5_no3m_photo_colint(k, conc_no3m, photic_depth_no3, photic_fraction):
	return reac5_no3m_photo(k, conc_no3m)*M2cgs*photic_depth_no3*photic_fraction #Note conversion to units of molecules cm**-2 s**-1. 

def reac6_hno2_nh3_colint(k, conc_no2m, conc_nh3, pH, total_depth):
	return reac6_hno2_nh3(k, conc_no2m, conc_nh3, pH)*M2cgs*total_depth #Note conversion to units of molecules cm**-2 s**-1. 

def reac_vents_colint(k, conc_NOx, total_depth):
	return reac_vents(k, conc_NOx)*M2cgs*total_depth #Note conversion to units of molecules cm**-2 s**-1.

######################################
###Plot Ocean Case
######################################

conc_fe2_range=10.**(np.arange(-6., -3., step=1.))
conc_no2m_range=10.**(np.arange(-12., 0., step=1.))
conc_no3m_range=10.**(np.arange(-12., 0., step=1.))


loss_rates_no2m=np.zeros((7, len(conc_no2m_range), len(conc_fe2_range)))
loss_rates_no3m=np.zeros((5, len(conc_no3m_range), len(conc_fe2_range)))


###Do calculations: Fixed
loss_rates_no2m[0, :, 0]=reac_vents_colint(k_laneuville, conc_no2m_range, ocean_depth_mean_modern) #Reduction by vents; Lanueville
loss_rates_no2m[1, :, 0]=reac_vents_colint(k_wong, conc_no2m_range, ocean_depth_mean_modern) #Reduction by vents; Wong
loss_rates_no2m[2, :, 0]=reac4_no2m_photo_colint(k4, conc_no2m_range, loss_fraction_k4, photic_depth_no2, photic_fraction) #reduction of nitrite to no by UV
loss_rates_no2m[3, :, 0]=reac6_hno2_nh3_colint(k6, conc_no2m_range,  conc_nh3,pH_nh3, ocean_depth_mean_modern) #anamox of nitrite and nh3
loss_rates_no2m[4, :, 0]=reac1_no2m_fe_colint(k1, conc_no2m_range, 1.e-4, ocean_depth_mean_modern) #reduction of no2- to nh3 by fe++

loss_rates_no3m[0, :, 0]=reac_vents_colint(k_laneuville, conc_no3m_range, ocean_depth_mean_modern) #Reduction by vents; Lanueville
loss_rates_no3m[1, :, 0]=reac_vents_colint(k_wong, conc_no3m_range, ocean_depth_mean_modern) #Reduction by vents; Wong
loss_rates_no3m[2, :, 0]=reac5_no3m_photo_colint(k5, conc_no3m_range, photic_depth_no3, photic_fraction) #reduction of nitrate to nitrite by UV


###Do calculations: variable Fe
for fe_ind in range(0, len(conc_fe2_range)):
	conc_fe=conc_fe2_range[fe_ind]
	
	loss_rates_no2m[5, :, fe_ind]=reac2_no2m_fe_colint(k2_min, conc_no2m_range, conc_fe, ocean_depth_mean_modern) #reduction of no2- to gaseous n (esp n2o) by fe++, pH=6.5 (min)
	loss_rates_no2m[6, :, fe_ind]=reac2_no2m_fe_colint(k2_max, conc_no2m_range, conc_fe, ocean_depth_mean_modern) #reduction of no2- to gaseous n (esp n2o) by fe++, pH=8 (max)		
	
	loss_rates_no3m[3, :, fe_ind]=reac3_no3m_fe_colint(k3_min, conc_no3m_range, conc_fe, ocean_depth_mean_modern) #reduction of no3- to no2- by fe++, pH=6.5 (min)
	loss_rates_no3m[4, :, fe_ind]=reac3_no3m_fe_colint(k3_max, conc_no3m_range, conc_fe, ocean_depth_mean_modern) #reduction of no3- to no2- by fe++, pH=8 (min)


###Make plot
markersizeval=4.
fig, (ax0, ax1)=plt.subplots(2, figsize=(8.5, 10.0), sharex=False, sharey=False)

colors=np.array(['darkgreen', 'indigo', 'orange', 'brown', 'red','green','blue'])
hatch_list=np.array(['//','\\','-'])
label_list=np.array(['NO3- + 1uM Fe(II)-->N2O', 'NO3- + 10uM Fe(II)-->N2O', 'NO3- + 100uM Fe(II)-->N2O'])

ax0.set_title('NO2-')
ax0.plot(conc_no2m_range, conc_no2m_range*0.+nox_input_high, color='black', linestyle='-', label='NOx Prod. (1 bar CO2)')
ax0.plot(conc_no2m_range, conc_no2m_range*0.+nox_input_low, color='black', linestyle='--', label='NOx Prod. (0.1 bar CO2)')
ax0.fill_between(conc_no2m_range, loss_rates_no2m[0, :, 0], loss_rates_no2m[1, :, 0], facecolor=colors[0], alpha=0.6, edgecolor='black', label='Vents')
ax0.plot(conc_no2m_range, loss_rates_no2m[2, :, 0], color=colors[1], linestyle='-', label='Photoreduction')
ax0.plot(conc_no2m_range, loss_rates_no2m[3, :, 0], color=colors[2], linestyle='--', label='Anamox')
ax0.plot(conc_no2m_range, loss_rates_no2m[4, :, 0], color=colors[3], linestyle='--', label='NO2- + Fe(II)-->NH3')

ax1.set_title('NO3-')
ax1.plot(conc_no3m_range, conc_no3m_range*0.+nox_input_high, color='black', linestyle='-')
ax1.plot(conc_no3m_range, conc_no3m_range*0.+nox_input_low, color='black', linestyle='--')

ax1.fill_between(conc_no3m_range, loss_rates_no3m[0, :, 0], loss_rates_no3m[1, :, 0], facecolor=colors[0], alpha=0.6)
ax1.plot(conc_no3m_range, loss_rates_no3m[2, :, 0], color=colors[1], linestyle='-')

for fe_ind in range(0, len(conc_fe2_range)):
	
	ax0.fill_between(conc_no2m_range, loss_rates_no2m[5, :, fe_ind], loss_rates_no2m[6, :, fe_ind], facecolor=colors[4+fe_ind], alpha=0.2, edgecolor='black', hatch=hatch_list[fe_ind], label=label_list[fe_ind])
	ax1.fill_between(conc_no3m_range, loss_rates_no3m[3, :, fe_ind], loss_rates_no3m[4, :, fe_ind], facecolor=colors[4+fe_ind], alpha=0.2, edgecolor='black', hatch=hatch_list[fe_ind])

ax0.set_yscale('log')
ax0.set_ylabel('NO2- Prod/Loss (cm**-2 s**-1)')
ax0.set_ylim([1.e2, 1.e10])
ax0.set_xscale('log')
ax0.set_xlabel('[NO2-] (M)')

ax1.set_yscale('log')
ax1.set_ylabel('NO3- Prod/Loss (cm**-2 s**-1)')
ax1.set_ylim([1.e2, 1.e10])
ax1.set_xscale('log')
ax1.set_xlabel('[NO3-] (M)')

ax0.legend(bbox_to_anchor=[-0.09, 1.08, 1.09, .152],loc='lower left', ncol=3, mode='expand', borderaxespad=0., fontsize=12)
plt.tight_layout(rect=(0,0,1,0.9))
plt.subplots_adjust(wspace=0., hspace=0.2)

plt.savefig('./Plots/important_rxns_ocean_focused_v3.pdf', orientation='portrait',papertype='letter', format='pdf')

#######################################
####Plot Lake Case
#######################################

conc_fe2_range=10.**(np.arange(-6., -3., step=1.))
conc_no2m_range=10.**(np.arange(-12., 0., step=1.))
conc_no3m_range=10.**(np.arange(-12., 0., step=1.))


loss_rates_no2m=np.zeros((5, len(conc_no2m_range), len(conc_fe2_range)))
loss_rates_no3m=np.zeros((3, len(conc_no3m_range), len(conc_fe2_range)))


###Do calculations: Fixed
loss_rates_no2m[0, :, 0]=reac4_no2m_photo_colint(k4, conc_no2m_range, loss_fraction_k4, np.minimum(lake_depth, photic_depth_no2), 1.) #reduction of nitrite to no by UV
loss_rates_no2m[1, :, 0]=reac6_hno2_nh3_colint(k6, conc_no2m_range,  conc_nh3,pH_nh3, lake_depth) #anamox of nitrite and nh3
loss_rates_no2m[2, :, 0]=reac1_no2m_fe_colint(k1, conc_no2m_range, 1.e-4, lake_depth) #reduction of no2- to nh3 by fe++

loss_rates_no3m[0, :, 0]=reac5_no3m_photo_colint(k5, conc_no3m_range, np.minimum(lake_depth, photic_depth_no3), 1.) #reduction of nitrate to nitrite by UV


###Do calculations: variable Fe
for fe_ind in range(0, len(conc_fe2_range)):
	conc_fe=conc_fe2_range[fe_ind]
	
	loss_rates_no2m[3, :, fe_ind]=reac2_no2m_fe_colint(k2_min, conc_no2m_range, conc_fe, lake_depth) #reduction of no2- to gaseous n (esp n2o) by fe++, pH=6.5 (min)
	loss_rates_no2m[4, :, fe_ind]=reac2_no2m_fe_colint(k2_max, conc_no2m_range, conc_fe, lake_depth) #reduction of no2- to gaseous n (esp n2o) by fe++, pH=8 (max)		
	
	loss_rates_no3m[1, :, fe_ind]=reac3_no3m_fe_colint(k3_min, conc_no3m_range, conc_fe, lake_depth) #reduction of no3- to no2- by fe++, pH=6.5 (min)
	loss_rates_no3m[2, :, fe_ind]=reac3_no3m_fe_colint(k3_max, conc_no3m_range, conc_fe, lake_depth) #reduction of no3- to no2- by fe++, pH=8 (min)

###Make plot
markersizeval=4.
fig, (ax0, ax1)=plt.subplots(2, figsize=(8.5, 10.0), sharex=False, sharey=False)

colors=np.array(['indigo', 'orange', 'brown', 'red','green','blue'])
hatch_list=np.array(['//','\\','-'])
label_list=np.array(['NO3- + 1uM Fe(II)-->N2O', 'NO3- + 10uM Fe(II)-->N2O', 'NO3- + 100uM Fe(II)-->N2O'])

ax0.set_title('NO2-')
ax0.plot(conc_no2m_range, conc_no2m_range*0.+nox_input_high, color='black', linestyle='-', label='NOx Prod. (1 bar CO2)')
ax0.plot(conc_no2m_range, conc_no2m_range*0.+nox_input_low, color='black', linestyle='--', label='NOx Prod. (0.1 bar CO2)')
ax0.plot(conc_no2m_range, conc_no2m_range*0.+nox_input_high*lake_catchement_ratio, color='hotpink', linestyle='-', label='NOx Prod. \n(1 bar CO2, CA/SA=100)')
ax0.plot(conc_no2m_range, conc_no2m_range*0.+nox_input_low*lake_catchement_ratio, color='hotpink', linestyle='--', label='NOx Prod. \n(0.1 bar CO2, CA/SA=100)')
###ax0.axvline(6.6e-6, color='black', linestyle=':', label='[HCO3-]')

ax0.plot(conc_no2m_range, loss_rates_no2m[0, :, 0], color=colors[0], linestyle='-', label='Photoreduction')
ax0.plot(conc_no2m_range, loss_rates_no2m[1, :, 0], color=colors[1], linestyle='--', label='Anamox')
ax0.plot(conc_no2m_range, loss_rates_no2m[2, :, 0], color=colors[2], linestyle='--', label='NO2- + Fe(II)-->NH3')

ax1.set_title('NO3-')
ax1.plot(conc_no3m_range, conc_no3m_range*0.+nox_input_high, color='black', linestyle='-')
ax1.plot(conc_no3m_range, conc_no3m_range*0.+nox_input_low, color='black', linestyle='--')
ax1.plot(conc_no3m_range, conc_no3m_range*0.+nox_input_high*lake_catchement_ratio, color='hotpink', linestyle='-', label='NOx Prod. (1 bar CO2, CA/SA=100)')
ax1.plot(conc_no3m_range, conc_no3m_range*0.+nox_input_low*lake_catchement_ratio, color='hotpink', linestyle='--', label='NOx Prod. (0.1 bar CO2, CA/SA=100)')

ax1.plot(conc_no3m_range, loss_rates_no3m[0, :, 0], color=colors[0], linestyle='-')

for fe_ind in range(0, len(conc_fe2_range)):
	
	ax0.fill_between(conc_no2m_range, loss_rates_no2m[3, :, fe_ind], loss_rates_no2m[4, :, fe_ind], facecolor=colors[3+fe_ind], alpha=0.2, edgecolor='black', hatch=hatch_list[fe_ind], label=label_list[fe_ind])
	ax1.fill_between(conc_no3m_range, loss_rates_no3m[1, :, fe_ind], loss_rates_no3m[2, :, fe_ind], facecolor=colors[3+fe_ind], alpha=0.2, edgecolor='black', hatch=hatch_list[fe_ind])

ax0.set_yscale('log')
ax0.set_ylabel('NO2- Prod/Loss (cm**-2 s**-1)')
ax0.set_ylim([1.e4, 1.e12])
ax0.set_xscale('log')
ax0.set_xlabel('[NO2-] (M)')

ax1.set_yscale('log')
ax1.set_ylabel('NO3- Prod/Loss (cm**-2 s**-1)')
ax1.set_ylim([1.e4, 1.e12])
ax1.set_xscale('log')
ax1.set_xlabel('[NO3-] (M)')

ax0.legend(bbox_to_anchor=[-0.09, 1.08, 1.09, .152],loc='lower left', ncol=3, mode='expand', borderaxespad=0., fontsize=12)
plt.tight_layout(rect=(0,0,1,0.85))
plt.subplots_adjust(wspace=0., hspace=0.2)

plt.savefig('./Plots/important_rxns_lake_focused_v3.pdf', orientation='portrait',papertype='letter', format='pdf')




plt.show()

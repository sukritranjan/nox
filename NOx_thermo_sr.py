#test as a function of parameter space, including pH and H2 amount
# FOR SYSTEM WITH ONLY MAXIMUM logK REACTION FOR EACH INITIAL SPECIES: RXNS 4,8,12,14,15
### USING AQUEOUS DELTA G VALUES WHERE EVERYWHERE
### COUPLED OCEAN/ATMOSPHERE MODEL

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import pdb
from scipy.optimize import fsolve

###############
###Physical constants/parameters
###############
Na=6.02E23  #Avagadro's number
amu=1.66E-27 #amu in kg
V=1.4E21  #Volume of oceans in L
Rearth=6.37E6 #radius of Earth in m
g=9.8  #surface gravity in m/s^2

###############
###Prescribed conditions
###############
pHarr=np.array([2.,4.,6.,8.,10.,12.]) #range of pH considered
H2arr=10.**np.array([0., -10., -20., -30., -40.]) #[H2] (M)

#pHarr=np.array([3., 7., 10.]) #range of pH considered
#H2arr=10.**np.array([0., -1., -5., -10., -20]) #[H2] (M)

totmole=4.E20  #total inventory of N atoms in moles NOTE: Should be 4e20, revised SR

mu=28.0*amu  #mean molecular mass of N2-dominated atmosphere, amu --> kg

###############
###Chemical parameters
###############
#Redox reactions eq. constants
logK4=222.3
logK8=160.8
logK12=162.1  #ocrrected for aqueous dG
logK14=101.6  #corrected for aqueous dG
logK15=40.4

#Henry's law constants in M/bar
H_HNO3=2.6E6
H_HNO2=50.
H_NO=1.9E-3
H_N2O=2.5E-2
H_N2=6.4E-4
H_NH3=60.

#Ka's for relevant acid/base partitioning pairs (pKa = - log(Ka))
logKaHNO3=1.3
logKaHNO2=-3.4
logKaNH4=-9.24

###############
###Derived quantities
###############
Harr=10.**(-pHarr) #[H+] (M)
OHarr=10.**(-14.)/Harr #[OH-] (M)

logH2arr=np.log10(H2arr)

K4=10.**logK4
K8=10.**logK8
K12=10.**logK12
K14=10.**logK14
K15=10.**logK15

Ka_HNO3=10.**logKaHNO3
Ka_HNO2=10.**logKaHNO2
Ka_NH4=10.**logKaNH4

###############
###Setup variables to hold outputs
###############
solutions=np.zeros([len(H2arr), len(pHarr), 6])

initial_guesses=np.zeros([len(H2arr), len(pHarr), 6])

###[H2]=1 M 
initial_guesses[0,0]=np.log10(np.array([6.5307866921472359e-129, 3.6725816691768732e-98, 1.3030886357178359e-80, 5.3696563920777559e-100, 8.5102619204198187e-39, 0.14620854989565243])) #pH=2
initial_guesses[0,1]=np.log10(np.array([6.5308313817487857e-125, 3.6724021004342038e-94, 1.3030914892184713e-78, 5.369659126654043e-96, 8.5103035556588403e-35, 0.14620848574996131])) #pH=4
initial_guesses[0,2]=np.log10(np.array([6.5308313817487857e-125, 3.6724021004342038e-94, 1.3030914892184713e-78, 5.369659126654043e-96, 8.5103035556588403e-35, 0.14620848574996131])) #pH=6; did not locate in NOx_thermo_analysis.py, so reseeded with pH=4
initial_guesses[0,3]=np.log10(np.array([6.1796436695361327e-117, 3.472754651357184e-86, 1.2321975305417502e-74, 4.8011356473040781e-88, 7.6093703940854099e-27, 0.13825290326306555])) #pH=8
initial_guesses[0,4]=np.log10(np.array([9.6690986286545364e-114, 5.4373337362337786e-83, 1.9292388066228457e-73, 1.1769878515994952e-85, 1.8654000345144486e-24, 0.021646415485613569])) #pH=10
initial_guesses[0,5]=np.log10(np.array([1.1330456614860721e-111, 6.3715839878125337e-81, 2.260723309987257e-73, 1.6161989638783328e-85, 2.5615027376518656e-24, 0.00025365732722732381])) #pH=12

###[H2]=1e-10 M #
initial_guesses[1,0]=np.log10(np.array([1.27623847e-88, 7.17681631e-68, 2.54643052e-55, 2.05051836e-59, 3.24985259e-08, 2.85714204e-01])) #pH=2
initial_guesses[1,1]=np.log10(np.array([1.27333539e-84, 7.16049110e-64, 2.54063812e-53, 2.04120028e-55, 3.23508442e-04, 2.85064285e-01])) #pH=4
initial_guesses[1,2]=np.log10(np.array([2.40756054e-81, 1.35387078e-60, 4.80371482e-52, 7.29716948e-53, 1.15652342e-01, 5.38985667e-02])) #pH=6
initial_guesses[1,3]=np.log10(np.array([2.67006320e-79, 1.50148688e-58, 5.32740035e-52, 8.97492174e-53, 1.42246989e-01, 5.97744151e-04])) #pH=8
initial_guesses[1,4]=np.log10(np.array([2.67282525e-77, 1.50304010e-56, 5.32740035e-52, 8.97492181e-53, 1.42541437e-01, 5.97744151e-06])) #pH=10
initial_guesses[1,5]=np.log10(np.array([2.67282525e-75, 1.50304010e-54, 5.32740035e-52, 8.97492181e-53, 1.42541437e-01, 5.97744151e-08])) #pH=12

###[H2]=1e-20 M #
initial_guesses[2,0]=np.log10(np.array([2.67301248e-60, 1.50314538e-49, 5.33336106e-42, 8.99501666e-43, 1.42561407e-01, 5.98412954e-13])) #pH=2
initial_guesses[2,1]=np.log10(np.array([2.67301147e-58, 1.50314470e-47, 5.33336011e-42, 8.99501667e-43, 1.42561278e-01, 5.98412973e-15])) #pH=4
initial_guesses[2,2]=np.log10(np.array([2.67301151e-56, 1.50314487e-45, 5.33336165e-42, 8.99501863e-43, 1.42561278e-01, 5.98412945e-17])) #pH=6
initial_guesses[2,3]=np.log10(np.array([2.67301110e-54, 1.50314472e-43, 5.33336171e-42, 8.99501883e-43, 1.42561278e-01, 5.98412958e-19])) #pH=8
initial_guesses[2,4]=np.log10(np.array([2.67301132e-52, 1.50314473e-41, 5.33336174e-42, 8.99501679e-43, 1.42561278e-01, 5.98412971e-21])) #pH=10
initial_guesses[2,5]=np.log10(np.array([2.67301132e-50, 1.50314473e-39, 5.33336180e-42, 8.99501718e-43, 1.42561278e-01, 5.98412984e-23])) #pH=12

###[H2]=1e-30 M #
initial_guesses[3,0]=np.log10(np.array([2.67301377e-35, 1.50314471e-34, 5.33335449e-32, 8.99503353e-33, 1.42561278e-01, 5.98413093e-28])) #pH=2
initial_guesses[3,1]=np.log10(np.array([2.67301429e-33, 1.50314599e-32, 5.33336541e-32, 8.99503069e-33, 1.42561278e-01, 5.98413112e-30])) #pH=4
initial_guesses[3,2]=np.log10(np.array([2.67300910e-31, 1.50314489e-30, 5.33337228e-32, 8.99505447e-33, 1.42561278e-01, 5.98413649e-32])) #pH=6
initial_guesses[3,3]=np.log10(np.array([2.67300921e-29, 1.50314495e-28, 5.33336637e-32, 8.99502515e-33, 1.42561278e-01, 5.98413054e-34])) #pH=8
initial_guesses[3,4]=np.log10(np.array([2.67301472e-27, 1.50314496e-26, 5.33336166e-32, 8.99498986e-33, 1.42561278e-01, 5.98412580e-36])) #pH=10
initial_guesses[3,5]=np.log10(np.array([2.67301486e-25, 1.50314497e-24, 5.33335386e-32, 8.99501006e-33, 1.42561278e-01, 5.98412637e-38])) #pH=12

###[H2]=1e-40 M #
initial_guesses[4,0]=np.log10(np.array([2.67300615e-10, 1.50314502e-19, 5.33336146e-22, 8.99501822e-23, 1.42561278e-01, 5.98412870e-43])) #pH=2
initial_guesses[4,1]=np.log10(np.array([2.67963855e-08, 1.50314463e-17, 5.33335899e-22, 8.99500769e-23, 1.42561264e-01, 5.98412658e-45])) #pH=4
initial_guesses[4,2]=np.log10(np.array([1.3294746125737547e-07, 7.4761859219562198e-17, 2.6526508656170916e-23, 2.2251545785956824e-25, 0.00035266323437769718, 2.9763232239217925e-48])) #pH=6
initial_guesses[4,3]=np.log10(np.array([2.67176111e-04, 1.50244173e-13, 5.33086441e-22, 8.98659713e-23, 1.42427966e-01, 5.98132824e-49])) #pH=8
initial_guesses[4,4]=np.log10(np.array([2.55089683e-02, 1.43447471e-11, 5.08970833e-22, 8.19192166e-23, 1.29833209e-01, 5.71074667e-51])) #pH=10
initial_guesses[4,5]=np.log10(np.array([2.82522480e-01, 1.58874066e-10, 5.63706458e-23, 1.00486107e-24, 1.59259747e-03, 6.32489049e-54])) #pH=12


###############
###Define core equation to solve
###############

def equations(p, logH2, logH, totmole):
	#REDOX RXNS 4,8,12,14,15
	#WITH LOGS
	
	#Extract variables to solve for
	logNO3,logNO2,logNO,logN2O,logN2,logNH4 = p    #log10(concentration in M)
	
	#Extract prescribed parameters
	
	
	#Compute derived quantities: concentrations of variables
	conc_NH4=10.**logNH4 #M
	conc_N2=10.**logN2 #M
	conc_N2O=10.**logN2O #M
	conc_NO=10.**logNO #M
	conc_NO2=10.**logNO2#M
	conc_NO3=10.**logNO3 #M
	
	conc_H=10.**logH
	
	#Derived quantities: dissociation reactions
	conc_NH3=conc_NH4*Ka_NH4/conc_H #M
	conc_HNO2=conc_H*conc_NO2/Ka_HNO2 #M
	conc_HNO3=conc_H*conc_NO3/Ka_HNO3 #M
	
	#Derived quantities: gas partial pressures. Need to convert to Pascals (SI unit of pressure)
	p_NH3=(conc_NH3/H_NH3)*1.e5 #bar --> Pa 
	p_N2=(conc_N2/H_N2)*1.e5 #bar --> Pa 
	p_N2O=(conc_N2O/H_N2O)*1.e5 #bar --> Pa 
	p_NO=(conc_NO/H_NO)*1.e5 #bar --> Pa 
	p_HNO2=(conc_HNO2/H_HNO2)*1.e5 #bar --> Pa 
	p_HNO3=(conc_HNO3/H_HNO3)*1.e5 #bar --> Pa 

	
	
	#Redox equations
	f4=logN2-2.*logNO3-2.*logH-5.*logH2-logK4
	f8=logN2-2.*logNO2-2.*logH-3.*logH2-logK8
	f12=2.*logNH4-2.*logNO-2.*logH-5.*logH2-logK12
	f14=2.*logNH4-logN2O-2.*logH-4.*logH2-logK14
	f15=2.*logNH4-logN2-2.*logH-3.*logH2-logK15
	
	#Conservation of N equation
	fX=V*(conc_NH4 + conc_NH3 + 2.*conc_N2 + 2.*conc_N2O + conc_NO + conc_NO2 + conc_NO3 + conc_HNO2 + conc_HNO3) + ((p_HNO3 + p_HNO2 + p_NO + 2.*p_N2O + 2.*p_N2 + p_NH3)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na)- totmole
	
	return np.array([f4,f8,f12,f14,f15,fX])

###############
###Run calculation
###############
for i in np.arange(0, len(H2arr)):
	H2=H2arr[i]
	logH2=logH2arr[i]
	
	for j in np.arange(0, len(pHarr)):
		pH=pHarr[j]
		H=Harr[j]
		initial_guess=initial_guesses[i,j]
		
		logH=np.log10(H)    
		
		solution=fsolve(equations, initial_guess,args=(logH2, logH,totmole), xtol=1.0E-6, maxfev=1000*(len(initial_guess)+1))
		
		logNO3,logNO2,logNO,logN2O,logN2,logNH4=solution
		
		NO3=10.**logNO3
		NO2=10.**logNO2
		NO=10.**logNO
		N2O=10.**logN2O
		N2=10.**logN2
		NH4=10.**logNH4
		
		solutions[i,j,:]=solution
		
		residual=equations(solution, logH2, logH, totmole)
		residual[-1]=residual[-1]/totmole #scale to figure out how well we are conserving N
		
		print 'pH = ', pH, ', [H2]=', H2
		#print solution
		print 10.**solution
		print  residual #check on solution quality...should be 0 to 1.0E-3 at least
		print ''
	pdb.set_trace()
	x=1+1



###############
###Extract total moles of N in each oxidation state
###############
N_oxidation=np.zeros(np.shape(solutions))

for i in np.arange(0, len(H2arr)):	
	for j in np.arange(0, len(pHarr)):
		pH=pHarr[j]
		
		#Extract solution
		solution=solutions[i,j,:]
		conc_NO3,conc_NO2,conc_NO,conc_N2O,conc_N2,conc_NH4 = 10**solution
		
		#Derive related quantities
		conc_H=10.**(-pH)
		
		#Dissociation reactions
		conc_NH3=conc_NH4*Ka_NH4/conc_H #M
		conc_HNO2=conc_H*conc_NO2/Ka_HNO2 #M
		conc_HNO3=conc_H*conc_NO3/Ka_HNO3 #M
		
		#Gas partial pressures
		p_NH3=(conc_NH3/H_NH3)*1.e5 #bar --> Pa 
		p_N2=(conc_N2/H_N2)*1.e5 #bar --> Pa 
		p_N2O=(conc_N2O/H_N2O)*1.e5 #bar --> Pa 
		p_NO=(conc_NO/H_NO)*1.e5 #bar --> Pa 
		p_HNO2=(conc_HNO2/H_HNO2)*1.e5 #bar --> Pa 
		p_HNO3=(conc_HNO3/H_HNO3)*1.e5 #bar --> Pa 
		
		#Put into matrix
		N_oxidation[i,j,0]=V*(conc_NO3 + conc_HNO3) + ((p_HNO3)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na) #NO3- (+5)
		N_oxidation[i,j,1]=V*(conc_NO2 + conc_HNO2) + ((p_HNO2)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na) #NO2- (+3)
		N_oxidation[i,j,2]=V*(conc_NO) + ((p_NO)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na) #NO (+2)
		N_oxidation[i,j,3]=2.*(V*(conc_N2O) + ((p_N2O)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na)) #N2O (+1)
		N_oxidation[i,j,4]=2.*(V*(conc_N2) + ((p_N2)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na)) #N2 (+0)
		N_oxidation[i,j,5]=V*(conc_NH3 + conc_NH4) + ((p_NH3)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na) #NH3 (-3)

N_oxidation_norm=N_oxidation/totmole #normalize by total number of moles. 

print np.sum(N_oxidation_norm, axis=2) # CHECKSUM: should be 1 uniformly
###############
###Plot
###############
fig, ax=plt.subplots(len(H2arr), figsize=(8.5, 10.0), sharex=True, sharey=True)

width=0.2
for i in np.arange(0, len(H2arr)):
	H2=H2arr[i]
	ax[i].set_title('[H2]=' + str(H2))
	pdb.set_trace
	ax[i].bar(pHarr, N_oxidation_norm[i,:,0], color='violet', label='N(+5)')
	ax[i].bar(pHarr, N_oxidation_norm[i,:,1], color='blue', label='N(+3)', bottom=N_oxidation_norm[i,:,0])
	ax[i].bar(pHarr, N_oxidation_norm[i,:,2], color='green', label='N(+2)', bottom=N_oxidation_norm[i,:,1]+N_oxidation_norm[i,:,0])
	ax[i].bar(pHarr, N_oxidation_norm[i,:,3], color='yellow', label='N(+1)', bottom=N_oxidation_norm[i,:,2]+N_oxidation_norm[i,:,1]+N_oxidation_norm[i,:,0])
	ax[i].bar(pHarr, N_oxidation_norm[i,:,4], color='orange', label='N(+0)', bottom=N_oxidation_norm[i,:,3]+N_oxidation_norm[i,:,2]+N_oxidation_norm[i,:,1]+N_oxidation_norm[i,:,0])
	ax[i].bar(pHarr, N_oxidation_norm[i,:,5], color='red', label='N(-3)', bottom=N_oxidation_norm[i,:,4]+N_oxidation_norm[i,:,3]+N_oxidation_norm[i,:,2]+N_oxidation_norm[i,:,1]+N_oxidation_norm[i,:,0])
	
	ax[i].set_ylabel('Mole Fraction',fontsize=16)

ax[0].legend(bbox_to_anchor=[-0.08, 1.25, 1.08, .152],loc='lower left', ncol=3, mode='expand', borderaxespad=0., fontsize=16)
plt.tight_layout(rect=(0,0.1,1,0.90))
ax[-1].set_ylim([0,1])
ax[-1].set_xlabel('pH',fontsize=16)
plt.savefig('./Plots/nox_thermo_sr.pdf', orientation='portrait',papertype='letter', format='pdf')
plt.show()
plt.close()






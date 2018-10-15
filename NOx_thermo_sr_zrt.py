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
H2arr=10.**np.array([0., -2., -5., -10., -15.,-20]) #[H2] (M)


#pHarr=np.array([3., 7., 10.]) #range of pH considered
#H2arr=10.**np.array([0., -1., -5., -10., -20]) #[H2] (M)

totmole=4.E20  #total inventory of N atoms in moles NOTE: Should be 4e20, revised SR

mu=28.0*amu  #mean molecular mass of N2-dominated atmosphere, amu --> kg

###############
###Chemical parameters
###############
#Redox reactions eq. constants

#ORIGINAL
logK4=222.3
logK8=160.8
logK12=162.1
logK14=101.6
logK15=40.4

#T=2 deg C
#logK4=222.91
#logK8=161.16
#logK12=162.60
#logK14=101.71
#logK15=40.40
#H_NH3=195.2
#H_N2=.00100319
#H_N2O=.0510684
#H_NO=0.00309323        #from site.colorado.edu source
#logKaHNO3=1.52
#logKaHNO2=-3.47
#logKaNH4=-10.0


#T=18 deg C
#logK4=222.72
#logK8=161.0
#logK12=162.30
#logK14=101.66
#logK15=40.36
#H_NH3=85.0
#H_N2=.00072118
#H_N2O=.02872565
#H_NO=0.00216659        #from site.colorado.edu source
#logKaHNO3=1.37
#logKaHNO2=-3.29
#logKaNH4=-9.46

#T=25 deg C
#logK4=222.63
#logK8=160.94
#logK12=162.11
#logK14=101.60
#logK15=40.31
#H_NH3=60.7
#H_N2=.00064601
#H_N2O=.02320822
#H_NO=0.00191709        #from site.colorado.edu source
#logKaHNO3=1.3
#logKaHNO2=-3.23
#logKaNH4=-9.24

#T=37 deg C
#logK4=222.50
#logK8=160.86
#logK12=161.73
#logK14=101.46
#logK15=40.19
#H_NH3=35.4
#H_N2=.000555781
#H_N2O=.01688052
#H_NO=0.00161919         #from site.colorado.edu source
#logKaHNO3=1.18
#logKaHNO2=-3.13
#logKaNH4=-8.9

#T=45 deg C
#logK4=222.40
#logK8=160.81
#logK12=161.43
#logK14=101.33
#logK15=40.08
#H_NH3=25.2
#H_N2=.00051928
#H_N2O=.01406652
#H_NO=0.00148442        #from site.colorado.edu source
#logKaHNO3=1.1
#logKaHNO2=-3.08
#logKaNH4=-8.67



#Henry's law constants in M/bar - ORIGINAL
H_HNO3=2.6E6     #ASSUME THAT H_HNO3 and H_HNO2 DON'T CHANGE WITH TEMPERATURE B/C CAN'T FIND THEIR PARAMETERS, AND THESE SPECIES ARE NEGLIGIBLE ANYWAY
H_HNO2=50.
H_NO=1.9E-3
H_N2O=2.5E-2
H_N2=6.4E-4
H_NH3=60.

#Ka's for relevant acid/base partitioning pairs (pKa = - log(Ka)) - ORIGINAL
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

###[H2]=1e-2 M #
initial_guesses[1,0]=np.log10(np.array([1.3072200584622479e-124, 7.5091639007977784e-95, 8.3568068735601919e-78, 2.1741122787475373e-95, 3.4095526912741566e-35, 0.29285739241273284]))  #pH=2
initial_guesses[1,1]=np.log10(np.array([1.3062475160775857e-120, 7.3562653368130816e-91, 8.2546766304400522e-76, 2.1542711132860827e-91, 3.4144023188511645e-31, 0.29285552541285043]))  #pH=4
initial_guesses[1,2]=np.log10(np.array([1.3073908272009773e-116, 7.3519988850885346e-87, 8.2490784156040341e-74, 2.1518443991456237e-87, 3.410443539148614e-27, 0.29268834707510483]))  #pH=6
initial_guesses[1,3]=np.log10(np.array([1.2368870352406052e-112, 6.9553780501676332e-83, 7.803769153244494e-72, 1.9258016161979853e-83, 3.0521951434059933e-23, 0.2768892303084361]))  #pH=8
initial_guesses[1,4]=np.log10(np.array([1.9331063823446646e-109, 1.0870656047816203e-79, 1.219707671145968e-70, 4.7044787309854566e-81, 7.4560963143784492e-21, 0.043276861213836573]))  #pH=10
initial_guesses[1,5]=np.log10(np.array([2.2644050350727376e-107, 1.2733685281905022e-77, 1.4287426878803031e-70, 6.4551780215982676e-81, 1.0230767702826503e-20, 0.00050693714182307927]))  #pH=12

###[H2=1e-5 M #
initial_guesses[2,0]=np.log10(np.array([1.3072097607610755e-108, 7.5089035765608094e-83, 8.3579609484013207e-68, 2.1743336586343061e-79, 3.409498973590979e-23, 0.29285739482226381]))
initial_guesses[2,1]=np.log10(np.array([1.3086669950007884e-104, 7.356358509915462e-79, 8.2535457129720702e-66, 2.1543065230519802e-75, 3.4144899587308422e-19, 0.29285552538429771]))
initial_guesses[2,2]=np.log10(np.array([1.307390827190539e-100, 7.3519988850396998e-75, 8.2490784155679987e-64, 2.1518443991266125e-71, 3.4104435391180788e-15, 0.2926883470737926]))
initial_guesses[2,3]=np.log10(np.array([1.2368856182321638e-96, 6.9553730818629673e-71, 7.8037693908066927e-62, 1.9258014945927433e-67, 3.0521948462369058e-11, 0.27688921789528687]))
initial_guesses[2,4]=np.log10(np.array([1.2368856182321638e-92, 6.9553730818629673e-67, 7.8037693908066927e-60, 1.9258014945927433e-63, 3.0521948462369058e-07, 0.29]))
initial_guesses[2,5]=np.log10(np.array([1.2368856182321638e-88, 6.9553730818629673e-63, 7.8037693908066927e-58, 1.9258014945927433e-59, 3.0521948462369058e-03, 0.29]))



###[H2]=1e-10 M #
initial_guesses[3,0]=np.log10(np.array([1.27623847e-88, 7.17681631e-68, 2.54643052e-55, 2.05051836e-59, 3.24985259e-08, 2.85714204e-01])) #pH=2
initial_guesses[3,1]=np.log10(np.array([1.27333539e-84, 7.16049110e-64, 2.54063812e-53, 2.04120028e-55, 3.23508442e-04, 2.85064285e-01])) #pH=4
initial_guesses[3,2]=np.log10(np.array([2.40756054e-81, 1.35387078e-60, 4.80371482e-52, 7.29716948e-53, 1.15652342e-01, 5.38985667e-02])) #pH=6
initial_guesses[3,3]=np.log10(np.array([2.67006320e-79, 1.50148688e-58, 5.32740035e-52, 8.97492174e-53, 1.42246989e-01, 5.97744151e-04])) #pH=8
initial_guesses[3,4]=np.log10(np.array([2.67282525e-77, 1.50304010e-56, 5.32740035e-52, 8.97492181e-53, 1.42541437e-01, 5.97744151e-06])) #pH=10
initial_guesses[3,5]=np.log10(np.array([2.67282525e-75, 1.50304010e-54, 5.32740035e-52, 8.97492181e-53, 1.42541437e-01, 5.97744151e-08])) #pH=12

###[H2]=1e-15 M #   #used 1e-20 guesses initially
initial_guesses[4,0]=np.log10(np.array([2.67301248e-60, 1.50314538e-49, 5.33336106e-42, 8.99501666e-43, 1.42561407e-01, 5.98412954e-13])) #pH=2
initial_guesses[4,1]=np.log10(np.array([2.67301147e-58, 1.50314470e-47, 5.33336011e-42, 8.99501667e-43, 1.42561278e-01, 5.98412973e-15])) #pH=4
initial_guesses[4,2]=np.log10(np.array([2.67301151e-56, 1.50314487e-45, 5.33336165e-42, 8.99501863e-43, 1.42561278e-01, 5.98412945e-17])) #pH=6
initial_guesses[4,3]=np.log10(np.array([2.67301110e-54, 1.50314472e-43, 5.33336171e-42, 8.99501883e-43, 1.42561278e-01, 5.98412958e-19])) #pH=8
initial_guesses[4,4]=np.log10(np.array([2.67301132e-52, 1.50314473e-41, 5.33336174e-42, 8.99501679e-43, 1.42561278e-01, 5.98412971e-21])) #pH=10
initial_guesses[4,5]=np.log10(np.array([2.67301132e-50, 1.50314473e-39, 5.33336180e-42, 8.99501718e-43, 1.42561278e-01, 5.98412984e-23])) #pH=12

###[H2]=1e-20 M #
initial_guesses[5,0]=np.log10(np.array([2.67301248e-60, 1.50314538e-49, 5.33336106e-42, 8.99501666e-43, 1.42561407e-01, 5.98412954e-13])) #pH=2
initial_guesses[5,1]=np.log10(np.array([2.67301147e-58, 1.50314470e-47, 5.33336011e-42, 8.99501667e-43, 1.42561278e-01, 5.98412973e-15])) #pH=4
initial_guesses[5,2]=np.log10(np.array([2.67301151e-56, 1.50314487e-45, 5.33336165e-42, 8.99501863e-43, 1.42561278e-01, 5.98412945e-17])) #pH=6
initial_guesses[5,3]=np.log10(np.array([2.67301110e-54, 1.50314472e-43, 5.33336171e-42, 8.99501883e-43, 1.42561278e-01, 5.98412958e-19])) #pH=8
initial_guesses[5,4]=np.log10(np.array([2.67301132e-52, 1.50314473e-41, 5.33336174e-42, 8.99501679e-43, 1.42561278e-01, 5.98412971e-21])) #pH=10
initial_guesses[5,5]=np.log10(np.array([2.67301132e-50, 1.50314473e-39, 5.33336180e-42, 8.99501718e-43, 1.42561278e-01, 5.98412984e-23])) #pH=12




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
    #pdb.set_trace()
	x=1+1



###############
###Extract total moles of N in each oxidation state
###############
N_oxidation=np.zeros(np.shape(solutions))

#ZRT here
N_gas=np.zeros(np.shape(solutions))
N_aq=np.zeros(np.shape(solutions))

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
    
        N_aq[i,j,0]=V*(conc_NO3 + conc_HNO3)
#print(i)
#       print(j)

        N_aq[i,j,1]=V*(conc_NO2 + conc_HNO2)

#print('HERE FUCKER')
#print(N_aq[:,:,0])


#print('NO3')

#print('NO2')
#print(N_aq[*,*,1])

        #Put into matrix with aqueous and gaseous species separately
#N_gas[i,j,0]=((p_HNO3)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na)  #NO3-
#N_gas[i,j,1]=((p_HNO2)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na) #NO2- (+3)
#N_gas[i,j,2]=((p_NO)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na) #NO (+2)
#N_gas[i,j,3]=2.*((p_N2O)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na) #N2O (+1)
#N_gas[i,j,4]=2.*((p_N2)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na) #N2 (+0)
#N_gas[i,j,5]=((p_NH3)/(mu*g))*(4*np.pi*Rearth**2.)*(1./Na) #NH3 (-3)



#N_aq[i,j,0]=V*(conc_NO3 + conc_HNO3)
# N_aq[i,j,1]=V*(conc_NO2 + conc_HNO2)
#N_aq[i,j,2]=V*(conc_NO)
#N_aq[i,j,3]=2.*(V*(conc_N2O))
# N_aq[i,j,4]=2.*(V*(conc_N2))
# N_aq[i,j,5]=V*(conc_NH3 + conc_NH4)

#print('HERE LOSER')
#print(N_aq)
#print('END LOSER')


N_oxidation_norm=N_oxidation/totmole #normalize by total number of moles.

#N_gas_norm=N_gas/totmole
N_aq_norm=N_aq/totmole

#print np.sum(N_gas_norm, axis=2)
#print np.sum(N_aq_norm, axis=2)

#print(N_aq_norm)

#print np.sum(N_oxidation_norm, axis=2) # CHECKSUM: should be 1 uniformly

pH=np.array([2,4,6,8,10,12])
NO3_1=np.array([1.28E-128,1.28E-124,1.28E-120,1.21E-116,1.89E-113,2.21E-111])
NO2_1=np.array([7.18E-98,7.18E-94,7.17E-90,6.79E-86,1.06E-82,1.24E-80])

NO3_n2=np.array([1.28E-120,1.28E-116,1.28E-112,1.21E-108,1.89E-105,2.21E-103])
NO2_n2=np.array([7.18E-92,7.18E-88,7.17E-84,6.79E-80,1.06E-76,1.24E-74])

NO3_n5=np.array([1.28E-108,1.28E-104,1.28E-100,1.21E-96,1.89E-93,2.21E-91])
NO2_n5=np.array([7.18E-83,7.18E-79,7.17E-75,6.79E-71,1.06E-67,1.24E-65])


NO3_n10=np.array([1.28E-88,9.45E-85,1.84E-82,1.85E-80,1.85E-78,1.85E-76])
NO2_n10=np.array([7.18E-68,5.31E-64,1.03E-61,1.04E-59,1.04E-57,1.04E-55])

NO3_n15=np.array([5.86E-74,5.86E-72,5.86E-70,5.86E-68,5.86E-66,5.86E-64])
NO2_n15=np.array([3.30E-58,3.30E-56,3.30E-54,3.30E-52,3.30E-50,3.30E-48])

NO3_n20=np.array([1.85E-61,1.85E-59,1.85E-57,1.85E-55,1.85E-53,1.85E-51])
NO2_n20=np.array([1.04E-50,1.04E-48,1.04E-46,1.04E-44,1.04E-42,1.04E-40])

fig=plt.figure()
ax1=fig.add_subplot(321)
ax1.plot(pH,NO3_1,marker='o',ls='none',color='violet',label='NO3-',markersize=15)
ax1.plot(pH,NO2_1,marker='o',ls='none',color='blue',label='NO2-',markersize=15)
ax1.legend(loc='best')
ax1.set_title('[H2]=1.0')
ax1.set_ylabel('Conc (M)',fontsize=16)
ax1.set_yscale('log')
ax1.set_xlim(1,13)

ax2=fig.add_subplot(323,sharex=ax1)
ax2.plot(pH,NO3_n2,marker='o',ls='none',color='violet',label='NO3-',markersize=15)
ax2.plot(pH,NO2_n2,marker='o',ls='none',color='blue',label='NO2-',markersize=15)
ax2.set_title('[H2]=0.01')
ax2.set_ylabel('Conc (M)',fontsize=16)
ax2.set_yscale('log')
ax2.set_xlim(1,13)

ax3=fig.add_subplot(325,sharex=ax1)
ax3.plot(pH,NO3_n5,marker='o',ls='none',color='violet',label='NO3-',markersize=15)
ax3.plot(pH,NO2_n5,marker='o',ls='none',color='blue',label='NO2-',markersize=15)
ax3.set_title('[H2]=1e-05')
ax3.set_ylabel('Conc (M)',fontsize=16)
ax3.set_yscale('log')
ax3.set_xlabel('pH',fontsize=16)
ax3.set_xlim(1,13)


ax4=fig.add_subplot(322,sharex=ax1)
ax4.plot(pH,NO3_n10,marker='o',ls='none',color='violet',label='NO3-',markersize=15)
ax4.plot(pH,NO2_n10,marker='o',ls='none',color='blue',label='NO2-',markersize=15)
ax4.set_title('[H2]=1e-10')
ax4.set_ylabel('Conc (M)',fontsize=16)
ax4.set_yscale('log')
ax4.set_xlim(1,13)

ax5=fig.add_subplot(324,sharex=ax1)
ax5.plot(pH,NO3_n15,marker='o',ls='none',color='violet',label='NO3-',markersize=15)
ax5.plot(pH,NO2_n15,marker='o',ls='none',color='blue',label='NO2-',markersize=15)
ax5.set_title('[H2]=1e-15')
ax5.set_ylabel('Conc (M)',fontsize=16)
ax5.set_yscale('log')
ax5.set_xlim(1,13)


ax6=fig.add_subplot(326,sharex=ax1)
ax6.plot(pH,NO3_n20,marker='o',ls='none',color='violet',label='NO3-',markersize=15)
ax6.plot(pH,NO2_n20,marker='o',ls='none',color='blue',label='NO2-',markersize=15)
ax6.set_title('[H2]=1e-20')
ax6.set_ylabel('Conc (M)',fontsize=16)
ax6.set_yscale('log')
ax6.set_xlabel('pH',fontsize=16)
ax6.set_xlim(1,13)


plt.subplots_adjust(hspace=.3,wspace=.3)


plt.show()
plt.close()




###############
###Plot - ORIGINAL
###############
fig, ax=plt.subplots(len(H2arr), figsize=(8.5, 10.0), sharex=True, sharey=True)

width=0.2
for i in np.arange(0, len(H2arr)):
    H2=H2arr[i]
    ax[i].set_title('[H2]=' + str(H2))
    #	pdb.set_trace
    ax[i].bar(pHarr, N_oxidation_norm[i,:,0], color='violet', label='N(+5)')
    ax[i].bar(pHarr, N_oxidation_norm[i,:,1], color='blue', label='N(+3)', bottom=N_oxidation_norm[i,:,0])
    ax[i].bar(pHarr, N_oxidation_norm[i,:,2], color='green', label='N(+2)', bottom=N_oxidation_norm[i,:,1]+N_oxidation_norm[i,:,0])
    ax[i].bar(pHarr, N_oxidation_norm[i,:,3], color='yellow', label='N(+1)', bottom=N_oxidation_norm[i,:,2]+N_oxidation_norm[i,:,1]+N_oxidation_norm[i,:,0])
    ax[i].bar(pHarr, N_oxidation_norm[i,:,4], color='orange', label='N(+0)', bottom=N_oxidation_norm[i,:,3]+N_oxidation_norm[i,:,2]+N_oxidation_norm[i,:,1]+N_oxidation_norm[i,:,0])
    ax[i].bar(pHarr, N_oxidation_norm[i,:,5], color='red', label='N(-3)', bottom=N_oxidation_norm[i,:,4]+N_oxidation_norm[i,:,3]+N_oxidation_norm[i,:,2]+N_oxidation_norm[i,:,1]+N_oxidation_norm[i,:,0])
        
    #ax[i].set_ylabel('Mole Fraction',fontsize=16)

ax[3].set_ylabel('Mole Fraction',fontsize=16)
ax[0].legend(bbox_to_anchor=[-0.08, 1.6, 1.08, .152],loc='lower left', ncol=3, mode='expand', borderaxespad=0., fontsize=16)
#ax[0].legend(bbox_to_anchor=[-0.08, 1.25, 1.08, .152],loc='lower left', ncol=3, mode='expand', borderaxespad=0., fontsize=16)
plt.tight_layout(rect=(0,0.1,1,0.90))
ax[-1].set_ylim([0,1])
ax[-1].set_xlabel('pH',fontsize=16)
#plt.savefig('./Plots/nox_thermo_sr.pdf', orientation='portrait',papertype='letter', format='pdf')
plt.show()
plt.close()





####ONLY CASE THAT IS NOTABLY TEMPERATURE DEPENDENT: pH 4, H2=1E-10

#print('N(-3)')
#print(N_oxidation_norm[3,:,5])
#print('N(0)')
#print(N_oxidation_norm[3,:,4])

#T=2 deg C
#N(-3)
#[  9.99952565e-01   7.40131254e-01   1.44219870e-02   1.53548866e-04 9.82498966e-06   8.38764657e-06]
#N(0)
#[  4.74351561e-05   2.59868791e-01   9.85578013e-01   9.99846451e-01 9.99990175e-01   9.99991612e-01]

#T=18 deg C
#N(-3)
#[  9.99952561e-01   7.39373711e-01   1.37773939e-02   1.46638535e-04 9.38279436e-06   8.01014180e-06]
#N(0)
#[  4.74396350e-05   2.60626289e-01   9.86222606e-01   9.99853361e-01 9.99990617e-01   9.99991990e-01]

#T=25 deg C
#N(-3)
#[  9.99952561e-01   7.07697732e-01   1.30117464e-02   1.38436272e-04 8.85793139e-06   7.56206318e-06]
#N(0)
#[  4.74396350e-05   2.92302265e-01   9.86988254e-01   9.99861564e-01 9.99991142e-01   9.99992438e-01]

#T=37 deg C
#N(-3)
#[  9.99952561e-01   6.62409705e-01   1.13423375e-02   1.20574029e-04 7.71494013e-06   6.58628491e-06]
#N(0)
#[  4.74396350e-05   3.37590294e-01   9.88657663e-01   9.99879426e-01 9.99992285e-01   9.99993414e-01]

#T=45 deg C
#N(-3)
#[  9.99952561e-01   6.19553258e-01   9.99993573e-03   1.06232375e-04 6.79724243e-06   5.80284117e-06]
#N(0)
#[  4.74396350e-05   3.80446742e-01   9.90000064e-01   9.99893768e-01 9.99993203e-01   9.99994197e-01]

T=np.array([2,18,25,37,45])
NH4=np.array([7.40131254e-01,7.39373711e-01,7.07697732e-01,6.62409705e-01,6.19553258e-01])
N2=np.array([2.59868791e-01,2.60626289e-01,2.92302265e-01,3.37590294e-01,3.80446742e-01])

plt.figure()
plt.plot(T,NH4,marker='o',ls='-',color='r',label='N(-3)',linewidth=2,markersize=10)
plt.plot(T,N2,marker='o',ls='-',color='orange',label='N(0)',linewidth=2,markersize=10)
plt.xlabel('Temperature ($^o C$)',fontsize=16)
plt.ylabel('Mole Fraction',fontsize=16)
plt.tick_params(which='both',labelsize=14,pad=5)
plt.legend(loc='best')
plt.xlim(0,47)
plt.show()
plt.close()





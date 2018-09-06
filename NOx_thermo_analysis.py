import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

V=1.4E21  #Volume of oceans in L
n=2.05E20  #total inventory of N atoms in moles,
totmole=2.05E20
logn=np.log10(n)  #20.56

#Ka's for relevant acid/base partitioning pairs
#PLEASE CHECK ME ON THIS!!!
logKaHNO3=1.3
logKaHNO2=-3.4
logKaNH4=-9.24

#Henry's law constants in M/bar
H_HNO3=2.6E6
H_HNO2=50.
H_NO=1.9E-3
H_N2O=2.5E-2
H_N2=6.4E-4
H_NH3=60.


#constants
mu=28.0  #assumed mean molecular mass of N2-dominated atmosphere
mH=1.67E-27 #mass of hydrogen in kg
g=9.8  #surface gravity in m/s^2
Rearth=6.37E6 #radius of Earth in m
Na=6.02E23  #Avagadro's number


#########     RESULTS
####  FOR SOLVING SIMPLIFIED SYSTEM, WITH HENRY'S LAW AND pKa SUBSTITUTIONS INCLUDED

#H2=1.0E0
#initial guess: (-10,-10,-10,-10,-10,-1.5) FOR pH 10
pH10_0=np.array([9.6690986286545364e-114, 5.4373337362337786e-83, 1.9292388066228457e-73, 1.1769878515994952e-85, 1.8654000345144486e-24, 0.021646415485613569])
#initial guess: (-10,-10,-10,-10,-10,-2) FOR pH 12
pH12_0=np.array([1.1330456614860721e-111, 6.3715839878125337e-81, 2.260723309987257e-73, 1.6161989638783328e-85, 2.5615027376518656e-24, 0.00025365732722732381])
#initial guess: (-10,-10,-10,-10,-10,-1) FOR pH 2 and 4
pH2_0=np.array([6.5307866921472359e-129, 3.6725816691768732e-98, 1.3030886357178359e-80, 5.3696563920777559e-100, 8.5102619204198187e-39, 0.14620854989565243])
pH4_0=np.array([6.5308313817487857e-125, 3.6724021004342038e-94, 1.3030914892184713e-78, 5.369659126654043e-96, 8.5103035556588403e-35, 0.14620848574996131])
#initial guess: (-10,-10,-10,-10,-10,-1.1) FOR pH 8
pH8_0=np.array([6.1796436695361327e-117, 3.472754651357184e-86, 1.2321975305417502e-74, 4.8011356473040781e-88, 7.6093703940854099e-27, 0.13825290326306555])

pH_0=np.array([2,4,8,10,12])


NO3_0=np.array([6.5307866921472359e-129,6.5308313817487857e-125,6.1796436695361327e-117,9.6690986286545364e-114,1.1330456614860721e-111])
NO2_0=np.array([3.6725816691768732e-98,3.6724021004342038e-94,3.472754651357184e-86,5.4373337362337786e-83,6.3715839878125337e-81])
NO_0=np.array([1.3030886357178359e-80,1.3030914892184713e-78,1.2321975305417502e-74,1.9292388066228457e-73,2.260723309987257e-73])
N2O_0=np.array([5.3696563920777559e-100,5.369659126654043e-96,4.8011356473040781e-88,1.1769878515994952e-85,1.6161989638783328e-85])
N2_0=np.array([8.5102619204198187e-39,8.5103035556588403e-35,7.6093703940854099e-27,1.8654000345144486e-24,2.5615027376518656e-24])
NH4_0=np.array([0.14620854989565243,0.14620848574996131,0.13825290326306555,0.021646415485613569,0.00025365732722732381])

H_0=10.0**(-pH_0)

HNO3_0=(H_0*NO3_0)/(10.**logKaHNO3)
HNO2_0=(H_0*NO2_0)/(10.**logKaHNO2)
NH3_0=(10.**logKaNH4)*NH4_0/H_0

PHNO3_0=HNO3_0/H_HNO3     #### THIS IS IN BARS!!!!
PHNO2_0=HNO2_0/H_HNO2
PNO_0=NO_0/H_NO
PN2O_0=N2O_0/H_N2O
PN2_0=N2_0/H_N2
PNH3_0=NH3_0/H_NH3

molesPHNO3_0=PHNO3_0*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPHNO2_0=PHNO2_0*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNO_0=PNO_0*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2O_0=2.0*PN2O_0*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2_0=2.0*PN2_0*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNH3_0=PNH3_0*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole

molesNO3_0=NO3_0*V/totmole
molesNO2_0=NO2_0*V/totmole
molesNO_0=NO_0*V/totmole
molesN2O_0=2.0*N2O_0*V/totmole
molesN2_0=2.0*N2_0*V/totmole
molesNH4_0=NH4_0*V/totmole
molesHNO3_0=HNO3_0*V/totmole
molesHNO2_0=HNO2_0*V/totmole
molesNH3_0=NH3_0*V/totmole

#print('HERE LOSER')
#print(molesPHNO3_0)
#print(molesPHNO2_0)
#print(molesPNO_0)
#print(molesPN2O_0)
#print(molesPN2_0)
#print(molesPNH3_0)
#print(molesNO3_0)
#print(molesNO2_0)
#print(molesNO_0)
#print(molesN2O_0)
#print(molesN2_0)
#print(molesNH4_0)
#print(molesHNO3_0)
#print(molesHNO2_0)
#print(molesNH3_0)
#print('END LOSER')

#H2=1.0E-1
#initial guess: (-10,-10,-10,-10,-4,-1.5) - I DON'T THINK THESE ARE GOOD!

#pH2_n1=np.array([1e-10, 1e-10, 1e-10, 1e-10, 0.0001, 0.031622776601683791])
#pH4_n1=np.array([1e-10, 1e-10, 1e-10, 1e-10, 0.0001, 0.031622776601683791])
#pH6_n1=np.array([1e-10, 1e-10, 1e-10, 1e-10, 0.0001, 0.031622776601683791])
#pH8_n1=np.array([1e-10, 1e-10, 1e-10, 1e-10, 0.0001, 0.031622776601683791])
#pH10_n1=np.array([1e-10, 1e-10, 1e-10, 1e-10, 0.0001, 0.031622776601683791])
#pH12_n1=np.array([1.1330115445846722e-107, 6.3713968814699985e-78, 7.1488248806842531e-71, 1.6161040448181669e-81, 2.5613522989411579e-21, 0.00025364987849447351])



#H2=1.0E-10
#(-10,-10,-10,-10,-2,-4) - for pH 4
pH4_n10=np.array([9.6158503267707134e-85, 5.4073900161760857e-64, 6.2147006885024857e-54, 1.2213508362793381e-56, 0.00018449108717403793, 0.069730088604870399])

#(-10,-10,-10,-10,-3,-6) - for pH 6,10,12
pH6_n10=np.array([1.3208344400920045e-82, 7.4275978878963919e-62, 1.6875359226639214e-53, 9.0054631385455993e-56, 0.00034809418496648616, 0.0018934464475166093])
pH10_n10=np.array([1.239458056863841e-78, 7.4766914114832709e-58, 2.6559524429894738e-53, 2.2236191616555032e-55, 0.00035265903919009031, 2.9763055210979087e-07])
pH12_n10=np.array([1.3294292008158695e-76, 7.4761509036144849e-56, 2.6526381346167462e-53, 2.2251337419661358e-55, 0.00035265993055390912, 2.9763092825034395e-09])
#(-10,-10,-10,-10,-2,-6) - for pH 8
#pH8_n10=np.array([3.2124456601730433e-87, 1.8064910060576626e-66, 0.0, 0.0, 0.00046885618292080142, 5.5219039908652594e-291])

pH_n10=np.array([4,6,10,12])

NO3_n10=np.array([9.6158503267707134e-85,1.3208344400920045e-82,1.239458056863841e-78,1.3294292008158695e-76])
NO2_n10=np.array([5.4073900161760857e-64,7.4275978878963919e-62,7.4766914114832709e-58,7.4761509036144849e-56])
NO_n10=np.array([6.2147006885024857e-54,1.6875359226639214e-53,2.6559524429894738e-53,2.6526381346167462e-53])
N2O_n10=np.array([1.2213508362793381e-56,9.0054631385455993e-56,2.2236191616555032e-55,2.2251337419661358e-55])
N2_n10=np.array([0.00018449108717403793,0.00034809418496648616,0.00035265903919009031,0.00035265993055390912])
NH4_n10=np.array([0.069730088604870399,0.0018934464475166093,2.9763055210979087e-07,2.9763092825034395e-09])

H_n10=10.0**(-pH_n10)

HNO3_n10=(H_n10*NO3_n10)/(10.**logKaHNO3)
HNO2_n10=(H_n10*NO2_n10)/(10.**logKaHNO2)
NH3_n10=(10.**logKaNH4)*NH4_n10/H_n10

PHNO3_n10=HNO3_n10/H_HNO3
PHNO2_n10=HNO2_n10/H_HNO2
PNO_n10=NO_n10/H_NO
PN2O_n10=N2O_n10/H_N2O
PN2_n10=N2_n10/H_N2
PNH3_n10=NH3_n10/H_NH3


molesPHNO3_n10=PHNO3_n10*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPHNO2_n10=PHNO2_n10*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNO_n10=PNO_n10*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2O_n10=2.0*PN2O_n10*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2_n10=2.0*PN2_n10*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNH3_n10=PNH3_n10*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole

molesNO3_n10=NO3_n10*V/totmole
molesNO2_n10=NO2_n10*V/totmole
molesNO_n10=NO_n10*V/totmole
molesN2O_n10=2.0*N2O_n10*V/totmole
molesN2_n10=2.0*N2_n10*V/totmole
molesNH4_n10=NH4_n10*V/totmole
molesHNO3_n10=HNO3_n10*V/totmole
molesHNO2_n10=HNO2_n10*V/totmole
molesNH3_n10=NH3_n10*V/totmole


#H2=1.0E-20
#(-10,-10,-10,-10,-3,-6) - for pH 2,4,6,10,12
pH2_n20=np.array([1.3294647444043087e-61, 7.4761890019528038e-51, 2.6526521100351451e-43, 2.2251565890329653e-45, 0.000352663539212519, 2.9763245102576758e-14])
pH4_n20=np.array([1.3294753250365865e-59, 7.4762068878544647e-49, 2.652651670478183e-43, 2.2251565045462071e-45, 0.00035266353994873819, 2.9763245133643812e-16])
pH6_n20=np.array([1.3294463822431016e-57, 7.4761891699158104e-47, 2.6526520733791967e-43, 2.2251564887859582e-45, 0.00035266354073968932, 2.9763245167020232e-18])
pH10_n20=np.array([1.2981042116177623e-53, 7.475790123918335e-43, 2.6532813304329661e-43, 2.225283215425085e-45, 0.0003526635756716176, 2.9763246641070302e-22])
pH12_n20=np.array([1.3288884163628069e-51, 7.4763078804211111e-41, 2.6527010583125465e-43, 2.2252275476714404e-45, 0.00035267473863127797, 2.9763717689564433e-24])
#(-10,-10,-10,-10,-3,-7) - for pH 8
pH8_n20=np.array([1.3294976153511838e-55, 7.4761891272323787e-45, 2.6526516048492495e-43, 2.2251564696882396e-45, 0.00035266353678450158, 2.9763245000119863e-20])

pH_n20=np.array([2,4,6,8,10,12])

NO3_n20=np.array([1.3294647444043087e-61,1.3294753250365865e-59,1.3294463822431016e-57,1.3294976153511838e-55,1.2981042116177623e-53,1.3288884163628069e-51])
NO2_n20=np.array([7.4761890019528038e-51,7.4762068878544647e-49,7.4761891699158104e-47,7.4761891272323787e-45,7.475790123918335e-43,7.4763078804211111e-41])
NO_n20=np.array([2.6526521100351451e-43,2.652651670478183e-43,2.6526520733791967e-43,2.6526516048492495e-43,2.6532813304329661e-43,2.6527010583125465e-43])
N2O_n20=np.array([2.2251565890329653e-45,2.2251565045462071e-45,2.2251564887859582e-45,2.2251564696882396e-45,2.225283215425085e-45,2.2252275476714404e-45])
N2_n20=np.array([0.000352663539212519,0.00035266353994873819,0.00035266354073968932,0.00035266353678450158,0.0003526635756716176,0.00035267473863127797])
NH4_n20=np.array([2.9763245102576758e-14,2.9763245133643812e-16,2.9763245167020232e-18,2.9763245000119863e-20,2.9763246641070302e-22,2.9763717689564433e-24])

H_n20=10.0**(-pH_n20)

HNO3_n20=(H_n20*NO3_n20)/(10.**logKaHNO3)
HNO2_n20=(H_n20*NO2_n20)/(10.**logKaHNO2)
NH3_n20=(10.**logKaNH4)*NH4_n20/H_n20

PHNO3_n20=HNO3_n20/H_HNO3
PHNO2_n20=HNO2_n20/H_HNO2
PNO_n20=NO_n20/H_NO
PN2O_n20=N2O_n20/H_N2O
PN2_n20=N2_n20/H_N2
PNH3_n20=NH3_n20/H_NH3


molesPHNO3_n20=PHNO3_n20*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPHNO2_n20=PHNO2_n20*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNO_n20=PNO_n20*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2O_n20=2.0*PN2O_n20*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2_n20=2.0*PN2_n20*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNH3_n20=PNH3_n20*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole

molesNO3_n20=NO3_n20*V/totmole
molesNO2_n20=NO2_n20*V/totmole
molesNO_n20=NO_n20*V/totmole
molesN2O_n20=2.0*N2O_n20*V/totmole
molesN2_n20=2.0*N2_n20*V/totmole
molesNH4_n20=NH4_n20*V/totmole
molesHNO3_n20=HNO3_n20*V/totmole
molesHNO2_n20=HNO2_n20*V/totmole
molesNH3_n20=NH3_n20*V/totmole

#H2=1.0E-30

#initial guess: (-10,-10,-10,-10,-3,-6) - for pH 2,4,6,10,12
pH2_n30=np.array([1.3294516203184857e-36, 7.4761893045559268e-36, 2.6526520819457222e-33, 2.2251565298655816e-35, 0.00035266354503584574, 2.9763245348308576e-29])
pH4_n30=np.array([1.3294753360462922e-34, 7.4761945337850452e-34, 2.6526513181472204e-33, 2.2251565510101247e-35, 0.00035266354578974171, 2.9763245380121154e-31])
pH6_n30=np.array([1.32948622259765e-32, 7.476189231356527e-32, 2.6526521585618523e-33, 2.2251564948295548e-35, 0.00035266354658729248, 2.9763245413776112e-33])
pH10_n30=np.array([1.261532271425828e-28, 7.4756417044328983e-28, 2.6530142467829784e-33, 2.2253532424580545e-35, 0.00035266361667486153, 2.9763248371316403e-37])
pH12_n30=np.array([1.3294666365413141e-26, 7.4761792111564105e-26, 2.6526486605752634e-33, 2.2251504848544915e-35, 0.00035266260122734157, 2.976320552164732e-39])
#initial guess: (-10,-10,-10,-10,-3,-7) - for pH 8
pH8_n30=np.array([1.3294507486386407e-30, 7.4761891342010118e-30, 2.6526520025809216e-33, 2.2251565196200309e-35, 0.00035266353737288216, 2.9763245024948331e-35])

pH_n30=np.array([2,4,6,8,10,12])

NO3_n30=np.array([1.3294516203184857e-36,1.3294753360462922e-34,1.32948622259765e-32,1.3294507486386407e-30,1.261532271425828e-28,1.3294666365413141e-26])
NO2_n30=np.array([7.4761893045559268e-36,7.4761945337850452e-34,7.476189231356527e-32,7.4761891342010118e-30,7.4756417044328983e-28,7.4761792111564105e-26])
NO_n30=np.array([2.6526520819457222e-33,2.6526513181472204e-33,2.6526521585618523e-33,2.6526520025809216e-33,2.6530142467829784e-33,2.6526486605752634e-33])
N2O_n30=np.array([2.2251565298655816e-35,2.2251565510101247e-35,2.2251564948295548e-35,2.2251565196200309e-35,2.2253532424580545e-35,2.2251504848544915e-35])
N2_n30=np.array([0.00035266354503584574,0.00035266354578974171,0.00035266354658729248,0.00035266353737288216,0.00035266361667486153,0.00035266260122734157])
NH4_n30=np.array([2.9763245348308576e-29,2.9763245380121154e-31,2.9763245413776112e-33,2.9763245024948331e-35,2.9763248371316403e-37,2.976320552164732e-39])

H_n30=10.0**(-pH_n30)

HNO3_n30=(H_n30*NO3_n30)/(10.**logKaHNO3)
HNO2_n30=(H_n30*NO2_n30)/(10.**logKaHNO2)
NH3_n30=(10.**logKaNH4)*NH4_n30/H_n30

PHNO3_n30=HNO3_n30/H_HNO3
PHNO2_n30=HNO2_n30/H_HNO2
PNO_n30=NO_n30/H_NO
PN2O_n30=N2O_n30/H_N2O
PN2_n30=N2_n30/H_N2
PNH3_n30=NH3_n30/H_NH3


molesPHNO3_n30=PHNO3_n30*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPHNO2_n30=PHNO2_n30*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNO_n30=PNO_n30*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2O_n30=2.0*PN2O_n30*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2_n30=2.0*PN2_n30*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNH3_n30=PNH3_n30*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole

molesNO3_n30=NO3_n30*V/totmole
molesNO2_n30=NO2_n30*V/totmole
molesNO_n30=NO_n30*V/totmole
molesN2O_n30=2.0*N2O_n30*V/totmole
molesN2_n30=2.0*N2_n30*V/totmole
molesNH4_n30=NH4_n30*V/totmole
molesHNO3_n30=HNO3_n30*V/totmole
molesHNO2_n30=HNO2_n30*V/totmole
molesNH3_n30=NH3_n30*V/totmole


#H2=1.0E-40
#initial guess: (-9,-10,-10,-10,-3,-6)
pH2_n40=np.array([1.3294753699349472e-11, 7.4761893070107633e-21, 2.6526520665218787e-23, 2.2251565936036519e-25, 0.00035266355373852023, 2.9763245715542233e-44])
pH4_n40=np.array([1.3294751405486326e-09, 7.4761892816303296e-19, 2.6526520573672662e-23, 2.2251565784553781e-25, 0.00035266355133986923, 2.9763245614324383e-46])
pH6_n40=np.array([1.3294746125737547e-07, 7.4761859219562198e-17, 2.6526508656170916e-23, 2.2251545785956824e-25, 0.00035266323437769718, 2.9763232239217925e-48])
pH8_n40=np.array([1.3294143872444935e-05, 7.4758491137933619e-15, 2.6525313635751129e-23, 2.2249540930756729e-25, 0.00035263145957122521, 2.9761891381771432e-50])
pH10_n40=np.array([0.0013234196378532244, 7.4421447906477358e-13, 2.6405726160450686e-23, 2.2049372292170454e-25, 0.0003494590004449706, 2.9627712054537926e-52])
pH12_n40=np.array([2.3268409323100092e-112, 0.0, 2.6860960775203803e-23, 2.2816188274985738e-25, 0.0003526624902403055, 3.0138493796049148e-54])

pH_n40=np.array([2,4,6,8,10,12])

NO3_n40=np.array([1.3294753699349472e-11,1.3294751405486326e-09,1.3294746125737547e-07,1.3294143872444935e-05,0.0013234196378532244,2.3268409323100092e-112])
NO2_n40=np.array([7.4761893070107633e-21,7.4761892816303296e-19,7.4761859219562198e-17,7.4758491137933619e-15,7.4421447906477358e-13,0.0])
NO_n40=np.array([2.6526520665218787e-23,2.6526520573672662e-23,2.6526508656170916e-23,2.6525313635751129e-23,2.6405726160450686e-23,2.6860960775203803e-23])
N2O_n40=np.array([2.2251565936036519e-25,2.2251565784553781e-25,2.2251545785956824e-25,2.2249540930756729e-25,2.2049372292170454e-25,2.2816188274985738e-25])
N2_n40=np.array([0.00035266355373852023,0.00035266355133986923,0.00035266323437769718,0.00035263145957122521,0.0003494590004449706,0.0003526624902403055])
NH4_n40=np.array([2.9763245715542233e-44,2.9763245614324383e-46,2.9763232239217925e-48,2.9761891381771432e-50,2.9627712054537926e-52,3.0138493796049148e-54])

H_n40=10.0**(-pH_n40)

HNO3_n40=(H_n40*NO3_n40)/(10.**logKaHNO3)
HNO2_n40=(H_n40*NO2_n40)/(10.**logKaHNO2)
NH3_n40=(10.**logKaNH4)*NH4_n40/H_n40

PHNO3_n40=HNO3_n40/H_HNO3
PHNO2_n40=HNO2_n40/H_HNO2
PNO_n40=NO_n40/H_NO
PN2O_n40=N2O_n40/H_N2O
PN2_n40=N2_n40/H_N2
PNH3_n40=NH3_n40/H_NH3


molesPHNO3_n40=PHNO3_n40*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPHNO2_n40=PHNO2_n40*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNO_n40=PNO_n40*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2O_n40=2.0*PN2O_n40*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2_n40=2.0*PN2_n40*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNH3_n40=PNH3_n40*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole

molesNO3_n40=NO3_n40*V/totmole
molesNO2_n40=NO2_n40*V/totmole
molesNO_n40=NO_n40*V/totmole
molesN2O_n40=2.0*N2O_n40*V/totmole
molesN2_n40=2.0*N2_n40*V/totmole
molesNH4_n40=NH4_n40*V/totmole
molesHNO3_n40=HNO3_n40*V/totmole
molesHNO2_n40=HNO2_n40*V/totmole
molesNH3_n40=NH3_n40*V/totmole


#H2=1.0E-50

#initial guess: (-9,-9,-10,-10,-3,-6) - for pH 2,4,6
pH2_n50=np.array([0.0, 1.2199632609050212e-220, 2.7711645550766912e-13, 2.42842466595295e-15, 0.00033230298000337286, 3.1092977966251949e-59])
pH4_n50=np.array([0.0, 2.3935353569204395e-16, 0.0, 2.3136180830113655e-15, 0.00035266249024023264, 3.0349101290327552e-61])
pH6_n50=np.array([0.0, 0.06114226494583281, 0.0, 1.5926060860632585e-15, 0.00024704141825874072, 2.5179910660856573e-63])
#initial guess: (-8,-3,-10,-10,-5,-10) - for pH 8
pH8_n50=np.array([2.5752751244690429e-49, 0.035635455446015686, 1.7938438364269823e-12, 5.7967769100053862e-14, 8.5606006607475583e-06, 4.9561576842053369e-39])

pH_n50=np.array([2,4,6,8])

NO3_n50=np.array([0.0,0.0,0.0,2.5752751244690429e-49])
NO2_n50=np.array([1.2199632609050212e-220,2.3935353569204395e-16,0.06114226494583281,0.035635455446015686])
NO_n50=np.array([2.7711645550766912e-13,0.0,0.0,1.7938438364269823e-12])
N2O_n50=np.array([2.42842466595295e-15,2.3136180830113655e-15,1.5926060860632585e-15,5.7967769100053862e-14])
N2_n50=np.array([0.00033230298000337286,0.00035266249024023264,0.00024704141825874072,8.5606006607475583e-06])
NH4_n50=np.array([3.1092977966251949e-59,3.0349101290327552e-61,2.5179910660856573e-63,4.9561576842053369e-39])

H_n50=10.0**(-pH_n50)

HNO3_n50=(H_n50*NO3_n50)/(10.**logKaHNO3)
HNO2_n50=(H_n50*NO2_n50)/(10.**logKaHNO2)
NH3_n50=(10.**logKaNH4)*NH4_n50/H_n50

PHNO3_n50=HNO3_n50/H_HNO3
PHNO2_n50=HNO2_n50/H_HNO2
PNO_n50=NO_n50/H_NO
PN2O_n50=N2O_n50/H_N2O
PN2_n50=N2_n50/H_N2
PNH3_n50=NH3_n50/H_NH3


molesPHNO3_n50=PHNO3_n50*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPHNO2_n50=PHNO2_n50*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNO_n50=PNO_n50*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2O_n50=2.0*PN2O_n50*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2_n50=2.0*PN2_n50*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNH3_n50=PNH3_n50*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole

molesNO3_n50=NO3_n50*V/totmole
molesNO2_n50=NO2_n50*V/totmole
molesNO_n50=NO_n50*V/totmole
molesN2O_n50=2.0*N2O_n50*V/totmole
molesN2_n50=2.0*N2_n50*V/totmole
molesNH4_n50=NH4_n50*V/totmole
molesHNO3_n50=HNO3_n50*V/totmole
molesHNO2_n50=HNO2_n50*V/totmole
molesNH3_n50=NH3_n50*V/totmole


#H2=1.0E-60

#(-2,-4,-10,-10,-10,-10) - for pH 6 - DIDN'T USE
#pH2_n60=np.array([0.022314358634970805, 0.0047433262060766384, 1.6829956493166389e-15, 8.9570703776617953e-30, 1.4195999877862753e-28, 1.8883521777968567e-86])

#(-1,-3,-10,-10,-10,-10) - for pH 2,6,8
pH2_n60=np.array([0.11967273326665431, 0.0010136106712509824, 3.0739065821229262e-16, 2.9880050681877514e-31, 4.7356688230081782e-30, 3.4489799060628873e-87])
pH6_n60=np.array([0.11205918282594361, 0.034062978941802824, 1.2086594653329753e-18, 4.6196378236456823e-36, 7.3216327822196619e-35, 1.3561382896589224e-93])
pH8_n60=np.array([0.078946281985594086, 0.067261713422871766, 2.3864733058108849e-20, 1.7999715286901013e-39, 2.8543532352658933e-38, 2.6775326592433918e-97])
#(-1.5,-3,-10,-10,-10,-10) - for pH 10,12
pH10_n60=np.array([0.12720757675406252, 0.019000258855174333, 3.1443265430719473e-23, 3.1264774710407085e-45, 4.9551327618886743e-44, 3.5279924581568518e-102])
pH12_n60=np.array([0.063667609606231496, 0.082542626382173689, 2.9287228908861933e-24, 2.7124176777360963e-47, 4.2988923637577537e-46, 3.2860811440784583e-105])

pH_n60=np.array([2,6,8,10,12])

NO3_n60=np.array([0.11967273326665431,0.11205918282594361,0.078946281985594086,0.12720757675406252,0.063667609606231496])
NO2_n60=np.array([0.0010136106712509824,0.034062978941802824,0.067261713422871766,0.019000258855174333,0.082542626382173689])
NO_n60=np.array([3.0739065821229262e-16,1.2086594653329753e-18,2.3864733058108849e-20,3.1443265430719473e-23,2.9287228908861933e-24])
N2O_n60=np.array([2.9880050681877514e-31,4.6196378236456823e-36,1.7999715286901013e-39,3.1264774710407085e-45,2.7124176777360963e-47])
N2_n60=np.array([4.7356688230081782e-30,7.3216327822196619e-35,2.8543532352658933e-38,4.9551327618886743e-44,4.2988923637577537e-46])
NH4_n60=np.array([3.4489799060628873e-87,1.3561382896589224e-93,2.6775326592433918e-97,3.5279924581568518e-102,3.2860811440784583e-105])

H_n60=10.0**(-pH_n60)

HNO3_n60=(H_n60*NO3_n60)/(10.**logKaHNO3)
HNO2_n60=(H_n60*NO2_n60)/(10.**logKaHNO2)
NH3_n60=(10.**logKaNH4)*NH4_n60/H_n60

PHNO3_n60=HNO3_n60/H_HNO3
PHNO2_n60=HNO2_n60/H_HNO2
PNO_n60=NO_n60/H_NO
PN2O_n60=N2O_n60/H_N2O
PN2_n60=N2_n60/H_N2
PNH3_n60=NH3_n60/H_NH3


molesPHNO3_n60=PHNO3_n60*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPHNO2_n60=PHNO2_n60*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNO_n60=PNO_n60*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2O_n60=2.0*PN2O_n60*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2_n60=2.0*PN2_n60*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNH3_n60=PNH3_n60*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole

molesNO3_n60=NO3_n60*V/totmole
molesNO2_n60=NO2_n60*V/totmole
molesNO_n60=NO_n60*V/totmole
molesN2O_n60=2.0*N2O_n60*V/totmole
molesN2_n60=2.0*N2_n60*V/totmole
molesNH4_n60=NH4_n60*V/totmole
molesHNO3_n60=HNO3_n60*V/totmole
molesHNO2_n60=HNO2_n60*V/totmole
molesNH3_n60=NH3_n60*V/totmole


#H2=1.0E-80
#initial guess: (0,-4,-10,-10,-10,-10)
pH4_n80=np.array([0.14620779189694114, 8.2209958670190228e-51, 2.9169267063900341e-75, 2.6906153650638004e-169, 4.2643107376094037e-188, 3.2728561006798088e-198])
pH6_n80=np.array([0.14732695908627527, 0.0, 0.0, 0.0, 0.0, 0.0])
pH8_n80=np.array([0.14620852652290534, 8.218290182611888e-51, 2.9159593214283551e-79, 2.6888280191070524e-177, 4.261504427462383e-196, 3.271760361657524e-206])
pH10_n80=np.array([0.14620852611175469, 8.2199616246899871e-51, 2.9165525505806141e-81, 2.6899213890615046e-181, 4.2632380176379787e-200, 3.2724256743738623e-210])
pH12_n80=np.array([0.14620852612328625, 8.2199322575068027e-51, 2.9165420397759353e-83, 2.6899024593219033e-185, 4.2632075610906877e-204, 3.2724139742646375e-214])

pH_n80=np.array([4,6,8,10,12])

NO3_n80=np.array([0.14620779189694114,0.14732695908627527,0.14620852652290534,0.14620852611175469,0.14620852612328625])
NO2_n80=np.array([8.2209958670190228e-51,0.0,8.218290182611888e-51,8.2199616246899871e-51,8.2199322575068027e-51])
NO_n80=np.array([2.9169267063900341e-75,0.0,2.9159593214283551e-79,2.9165525505806141e-81,2.9165420397759353e-83])
N2O_n80=np.array([2.6906153650638004e-169,0.0,2.6888280191070524e-177,2.6899213890615046e-181,2.6899024593219033e-185])
N2_n80=np.array([4.2643107376094037e-188,0.0,4.261504427462383e-196,4.2632380176379787e-200,4.2632075610906877e-204])
NH4_n80=np.array([3.2728561006798088e-198,0.0,3.271760361657524e-206,3.2724256743738623e-210,3.2724139742646375e-214])


H_n80=10.0**(-pH_n80)

HNO3_n80=(H_n80*NO3_n80)/(10.**logKaHNO3)
HNO2_n80=(H_n80*NO2_n80)/(10.**logKaHNO2)
NH3_n80=(10.**logKaNH4)*NH4_n80/H_n80

PHNO3_n80=HNO3_n80/H_HNO3
PHNO2_n80=HNO2_n80/H_HNO2
PNO_n80=NO_n80/H_NO
PN2O_n80=N2O_n80/H_N2O
PN2_n80=N2_n80/H_N2
PNH3_n80=NH3_n80/H_NH3


molesPHNO3_n80=PHNO3_n80*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPHNO2_n80=PHNO2_n80*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNO_n80=PNO_n80*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2O_n80=2.0*PN2O_n80*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPN2_n80=2.0*PN2_n80*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole
molesPNH3_n80=PNH3_n80*1.0E5*4.*3.14159*Rearth**2.0/(mu*mH*g*Na)/totmole

molesNO3_n80=NO3_n80*V/totmole
molesNO2_n80=NO2_n80*V/totmole
molesNO_n80=NO_n80*V/totmole
molesN2O_n80=2.0*N2O_n80*V/totmole
molesN2_n80=2.0*N2_n80*V/totmole
molesNH4_n80=NH4_n80*V/totmole
molesHNO3_n80=HNO3_n80*V/totmole
molesHNO2_n80=HNO2_n80*V/totmole
molesNH3_n80=NH3_n80*V/totmole

#### HERE
print(pH_n50)
print(molesPN2O_n50)



#############     PLOT

width=.2

plt.figure()
#H2=1E-1
plt.bar(pH_0-.8,molesNO3_0,width,color='royalblue',label='NO$_{3}$',alpha=.8)
plt.bar(pH_0-.8,molesHNO3_0,width,color='royalblue',label='HNO$_3$ (aq)',alpha=.4,bottom=molesNO3_0)
plt.bar(pH_0-.8, width ,molesPHNO3_0,width,color='royalblue',label='HNO$_3$ (g)',alpha=.4,hatch='/',bottom=molesNO3_0+molesHNO3_0)
plt.bar(pH_0-.8,molesNO2_0,width,color='mediumseagreen',label='NO$_{2}$',alpha=.8,bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0)
plt.bar(pH_0-.8,molesHNO2_0,width,color='mediumseagreen',label='HNO$_2$ (aq)',alpha=.4,bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0)
plt.bar(pH_0-.8,molesPHNO2_0,width,color='mediumseagreen',label='HNO$_2$ (g)',alpha=.4,hatch='/',bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0)
plt.bar(pH_0-.8,molesNO_0,width,color='gold',label='NO (aq)',alpha=.4,bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0+molesPHNO2_0)
plt.bar(pH_0-.8,molesPNO_0,width,color='gold',label='NO (g)',alpha=.4,hatch='/',bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0+molesPHNO2_0+molesNO_0)
plt.bar(pH_0-.8,molesN2O_0,width,color='aqua',label='N$_2$O (aq)',alpha=.4,bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0+molesPHNO2_0+molesNO_0+molesPNO_0)
plt.bar(pH_0-.8,molesPN2O_0,width,color='aqua',label='N$_2$O (g)',alpha=.4,hatch='/',bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0+molesPHNO2_0+molesNO_0+molesPNO_0+molesN2O_0)
plt.bar(pH_0-.8,molesN2_0,width,color='darkviolet',label='N$_2$ (aq)',alpha=.4,bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0+molesPHNO2_0+molesNO_0+molesPNO_0+molesN2O_0+molesPN2O_0)
plt.bar(pH_0-.8,molesPN2_0,width,color='darkviolet',label='N$_2$ (g)',alpha=.4,hatch='/',bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0+molesPHNO2_0+molesNO_0+molesPNO_0+molesN2O_0+molesPN2O_0+molesN2_0)
plt.bar(pH_0-.8,molesNH4_0,width,color='crimson',label='NH$_{4}$',alpha=.8,bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0+molesPHNO2_0+molesNO_0+molesPNO_0+molesN2O_0+molesPN2O_0+molesN2_0+molesPN2_0)
plt.bar(pH_0-.8,molesNH3_0,width,color='crimson',label='NH$_{3}$ (aq)',alpha=.4,bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0+molesPHNO2_0+molesNO_0+molesPNO_0+molesN2O_0+molesPN2O_0+molesN2_0+molesPN2_0+molesNH4_0)
plt.bar(pH_0-.8,molesPNH3_0,width,color='crimson',label='NH$_{3}$ (g)',alpha=.4,hatch='/',bottom=molesNO3_0+molesHNO3_0+molesPHNO3_0+molesNO2_0+molesHNO2_0+molesPHNO2_0+molesNO_0+molesPNO_0+molesN2O_0+molesPN2O_0+molesN2_0+molesPN2_0+molesNH4_0+molesNH3_0)


#H2=1E-10
plt.bar(pH_n10-.6,molesNO3_n10,width,color='royalblue',alpha=.8)
plt.bar(pH_n10-.6,molesHNO3_n10,width,color='royalblue',alpha=.4,bottom=molesNO3_n10)
plt.bar(pH_n10-.6,molesPHNO3_n10,width,color='royalblue',alpha=.4,hatch='/',bottom=molesNO3_n10+molesHNO3_n10)
plt.bar(pH_n10-.6,molesNO2_n10,width,color='mediumseagreen',alpha=.8,bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10)
plt.bar(pH_n10-.6,molesHNO2_n10,width,color='mediumseagreen',alpha=.4,bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10)
plt.bar(pH_n10-.6,molesPHNO2_n10,width,color='mediumseagreen',alpha=.4,hatch='/',bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10)
plt.bar(pH_n10-.6,molesNO_n10,width,color='gold',alpha=.4,bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10+molesPHNO2_n10)
plt.bar(pH_n10-.6,molesPNO_n10,width,color='gold',alpha=.4,hatch='/',bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10+molesPHNO2_n10+molesNO_n10)
plt.bar(pH_n10-.6,molesN2O_n10,width,color='aqua',alpha=.4,bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10+molesPHNO2_n10+molesNO_n10+molesPNO_n10)
plt.bar(pH_n10-.6,molesPN2O_n10,width,color='aqua',alpha=.4,hatch='/',bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10+molesPHNO2_n10+molesNO_n10+molesPNO_n10+molesN2O_n10)
plt.bar(pH_n10-.6,molesN2_n10,width,color='darkviolet',alpha=.4,bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10+molesPHNO2_n10+molesNO_n10+molesPNO_n10+molesN2O_n10+molesPN2O_n10)
plt.bar(pH_n10-.6,molesPN2_n10,width,color='darkviolet',alpha=.4,hatch='/',bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10+molesPHNO2_n10+molesNO_n10+molesPNO_n10+molesN2O_n10+molesPN2O_n10+molesN2_n10)
plt.bar(pH_n10-.6,molesNH4_n10,width,color='crimson',alpha=.8,bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10+molesPHNO2_n10+molesNO_n10+molesPNO_n10+molesN2O_n10+molesPN2O_n10+molesN2_n10+molesPN2_n10)
plt.bar(pH_n10-.6,molesNH3_n10,width,color='crimson',alpha=.4,bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10+molesPHNO2_n10+molesNO_n10+molesPNO_n10+molesN2O_n10+molesPN2O_n10+molesN2_n10+molesPN2_n10+molesNH4_n10)
plt.bar(pH_n10-.6,molesPNH3_n10,width,color='crimson',alpha=.4,hatch='/',bottom=molesNO3_n10+molesHNO3_n10+molesPHNO3_n10+molesNO2_n10+molesHNO2_n10+molesPHNO2_n10+molesNO_n10+molesPNO_n10+molesN2O_n10+molesPN2O_n10+molesN2_n10+molesPN2_n10+molesNH4_n10+molesNH3_n10)

#H2=1E-20
plt.bar(pH_n20-.4,molesNO3_n20,width,color='royalblue',alpha=.8)
plt.bar(pH_n20-.4,molesHNO3_n20,width,color='royalblue',alpha=.4,bottom=molesNO3_n20)
plt.bar(pH_n20-.4,molesPHNO3_n20,width,color='royalblue',alpha=.4,hatch='/',bottom=molesNO3_n20+molesHNO3_n20)
plt.bar(pH_n20-.4,molesNO2_n20,width,color='mediumseagreen',alpha=.8,bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20)
plt.bar(pH_n20-.4,molesHNO2_n20,width,color='mediumseagreen',alpha=.4,bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20)
plt.bar(pH_n20-.4,molesPHNO2_n20,width,color='mediumseagreen',alpha=.4,hatch='/',bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20)
plt.bar(pH_n20-.4,molesNO_n20,width,color='gold',alpha=.4,bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20+molesPHNO2_n20)
plt.bar(pH_n20-.4,molesPNO_n20,width,color='gold',alpha=.4,hatch='/',bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20+molesPHNO2_n20+molesNO_n20)
plt.bar(pH_n20-.4,molesN2O_n20,width,color='aqua',alpha=.4,bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20+molesPHNO2_n20+molesNO_n20+molesPNO_n20)
plt.bar(pH_n20-.4,molesPN2O_n20,width,color='aqua',alpha=.4,hatch='/',bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20+molesPHNO2_n20+molesNO_n20+molesPNO_n20+molesN2O_n20)
plt.bar(pH_n20-.4,molesN2_n20,width,color='darkviolet',alpha=.4,bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20+molesPHNO2_n20+molesNO_n20+molesPNO_n20+molesN2O_n20+molesPN2O_n20)
plt.bar(pH_n20-.4,molesPN2_n20,width,color='darkviolet',alpha=.4,hatch='/',bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20+molesPHNO2_n20+molesNO_n20+molesPNO_n20+molesN2O_n20+molesPN2O_n20+molesN2_n20)
plt.bar(pH_n20-.4,molesNH4_n20,width,color='crimson',alpha=.8,bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20+molesPHNO2_n20+molesNO_n20+molesPNO_n20+molesN2O_n20+molesPN2O_n20+molesN2_n20+molesPN2_n20)
plt.bar(pH_n20-.4,molesNH3_n20,width,color='crimson',alpha=.4,bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20+molesPHNO2_n20+molesNO_n20+molesPNO_n20+molesN2O_n20+molesPN2O_n20+molesN2_n20+molesPN2_n20+molesNH4_n20)
plt.bar(pH_n20-.4,molesPNH3_n20,width,color='crimson',alpha=.4,hatch='/',bottom=molesNO3_n20+molesHNO3_n20+molesPHNO3_n20+molesNO2_n20+molesHNO2_n20+molesPHNO2_n20+molesNO_n20+molesPNO_n20+molesN2O_n20+molesPN2O_n20+molesN2_n20+molesPN2_n20+molesNH4_n20+molesNH3_n20)


#H2=1E-30
plt.bar(pH_n30-.2,molesNO3_n30,width,color='royalblue',alpha=.8)
plt.bar(pH_n30-.2,molesHNO3_n30,width,color='royalblue',alpha=.4,bottom=molesNO3_n30)
plt.bar(pH_n30-.2,molesPHNO3_n30,width,color='royalblue',alpha=.4,hatch='/',bottom=molesNO3_n30+molesHNO3_n30)
plt.bar(pH_n30-.2,molesNO2_n30,width,color='mediumseagreen',alpha=.8,bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30)
plt.bar(pH_n30-.2,molesHNO2_n30,width,color='mediumseagreen',alpha=.4,bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30)
plt.bar(pH_n30-.2,molesPHNO2_n30,width,color='mediumseagreen',alpha=.4,hatch='/',bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30)
plt.bar(pH_n30-.2,molesNO_n30,width,color='gold',alpha=.4,bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30+molesPHNO2_n30)
plt.bar(pH_n30-.2,molesPNO_n30,width,color='gold',alpha=.4,hatch='/',bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30+molesPHNO2_n30+molesNO_n30)
plt.bar(pH_n30-.2,molesN2O_n30,width,color='aqua',alpha=.4,bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30+molesPHNO2_n30+molesNO_n30+molesPNO_n30)
plt.bar(pH_n30-.2,molesPN2O_n30,width,color='aqua',alpha=.4,hatch='/',bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30+molesPHNO2_n30+molesNO_n30+molesPNO_n30+molesN2O_n30)
plt.bar(pH_n30-.2,molesN2_n30,width,color='darkviolet',alpha=.4,bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30+molesPHNO2_n30+molesNO_n30+molesPNO_n30+molesN2O_n30+molesPN2O_n30)
plt.bar(pH_n30-.2,molesPN2_n30,width,color='darkviolet',alpha=.4,hatch='/',bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30+molesPHNO2_n30+molesNO_n30+molesPNO_n30+molesN2O_n30+molesPN2O_n30+molesN2_n30)
plt.bar(pH_n30-.2,molesNH4_n30,width,color='crimson',alpha=.8,bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30+molesPHNO2_n30+molesNO_n30+molesPNO_n30+molesN2O_n30+molesPN2O_n30+molesN2_n30+molesPN2_n30)
plt.bar(pH_n30-.2,molesNH3_n30,width,color='crimson',alpha=.4,bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30+molesPHNO2_n30+molesNO_n30+molesPNO_n30+molesN2O_n30+molesPN2O_n30+molesN2_n30+molesPN2_n30+molesNH4_n30)
plt.bar(pH_n30-.2,molesPNH3_n30,width,color='crimson',alpha=.4,hatch='/',bottom=molesNO3_n30+molesHNO3_n30+molesPHNO3_n30+molesNO2_n30+molesHNO2_n30+molesPHNO2_n30+molesNO_n30+molesPNO_n30+molesN2O_n30+molesPN2O_n30+molesN2_n30+molesPN2_n30+molesNH4_n30+molesNH3_n30)



#H2=1E-40
plt.bar(pH_n40,molesNO3_n40,width,color='royalblue',alpha=.8)
plt.bar(pH_n40,molesHNO3_n40,width,color='royalblue',alpha=.4,bottom=molesNO3_n40)
plt.bar(pH_n40,molesPHNO3_n40,width,color='royalblue',alpha=.4,hatch='/',bottom=molesNO3_n40+molesHNO3_n40)
plt.bar(pH_n40,molesNO2_n40,width,color='mediumseagreen',alpha=.8,bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40)
plt.bar(pH_n40,molesHNO2_n40,width,color='mediumseagreen',alpha=.4,bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40)
plt.bar(pH_n40,molesPHNO2_n40,width,color='mediumseagreen',alpha=.4,hatch='/',bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40)
plt.bar(pH_n40,molesNO_n40,width,color='gold',alpha=.4,bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40+molesPHNO2_n40)
plt.bar(pH_n40,molesPNO_n40,width,color='gold',alpha=.4,hatch='/',bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40+molesPHNO2_n40+molesNO_n40)
plt.bar(pH_n40,molesN2O_n40,width,color='aqua',alpha=.4,bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40+molesPHNO2_n40+molesNO_n40+molesPNO_n40)

plt.bar(pH_n40,molesPN2O_n40,width,color='aqua',alpha=.4,hatch='/',bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40+molesPHNO2_n40+molesNO_n40+molesPNO_n40+molesN2O_n40)
plt.bar(pH_n40,molesN2_n40,width,color='darkviolet',alpha=.4,bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40+molesPHNO2_n40+molesNO_n40+molesPNO_n40+molesN2O_n40+molesPN2O_n40)
plt.bar(pH_n40,molesPN2_n40,width,color='darkviolet',alpha=.4,hatch='/',bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40+molesPHNO2_n40+molesNO_n40+molesPNO_n40+molesN2O_n40+molesPN2O_n40+molesN2_n40)
plt.bar(pH_n40,molesNH4_n40,width,color='crimson',alpha=.8,bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40+molesPHNO2_n40+molesNO_n40+molesPNO_n40+molesN2O_n40+molesPN2O_n40+molesN2_n40+molesPN2_n40)
plt.bar(pH_n40,molesNH3_n40,width,color='crimson',alpha=.4,bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40+molesPHNO2_n40+molesNO_n40+molesPNO_n40+molesN2O_n40+molesPN2O_n40+molesN2_n40+molesPN2_n40+molesNH4_n40)
plt.bar(pH_n40,molesPNH3_n40,width,color='crimson',alpha=.4,hatch='/',bottom=molesNO3_n40+molesHNO3_n40+molesPHNO3_n40+molesNO2_n40+molesHNO2_n40+molesPHNO2_n40+molesNO_n40+molesPNO_n40+molesN2O_n40+molesPN2O_n40+molesN2_n40+molesPN2_n40+molesNH4_n40+molesNH3_n40)

#H2=1E-50
#plt.bar(pH_n50+.2,molesNO3_n50,width,color='royalblue',alpha=.8)
#plt.bar(pH_n50+.2,molesHNO3_n50,width,color='royalblue',alpha=.4,bottom=molesNO3_n50)
#plt.bar(pH_n50+.2,molesPHNO3_n50,width,color='royalblue',alpha=.4,hatch='/',bottom=molesNO3_n50+molesHNO3_n50)
#plt.bar(pH_n50+.2,molesNO2_n50,width,color='mediumseagreen',alpha=.8,bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50)
#plt.bar(pH_n50+.2,molesHNO2_n50,width,color='mediumseagreen',alpha=.4,bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50)
#plt.bar(pH_n50+.2,molesPHNO2_n50,width,color='mediumseagreen',alpha=.4,hatch='/',bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50)
#plt.bar(pH_n50+.2,molesNO_n50,width,color='gold',alpha=.4,bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50+molesPHNO2_n50)
#plt.bar(pH_n50+.2,molesPNO_n50,width,color='gold',alpha=.4,hatch='/',bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50+molesPHNO2_n50+molesNO_n50)
#plt.bar(pH_n50+.2,molesN2O_n50,width,color='aqua',alpha=.4,bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50+molesPHNO2_n50+molesNO_n50+molesPNO_n50)

#plt.bar(pH_n50+.2,molesPN2O_n50,width,color='aqua',alpha=.4,hatch='/',bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50+molesPHNO2_n50+molesNO_n50+molesPNO_n50+molesN2O_n50)
#plt.bar(pH_n50+.2,molesN2_n50,width,color='darkviolet',alpha=.4,bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50+molesPHNO2_n50+molesNO_n50+molesPNO_n50+molesN2O_n50+molesPN2O_n50)
#plt.bar(pH_n50+.2,molesPN2_n50,width,color='darkviolet',alpha=.4,hatch='/',bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50+molesPHNO2_n50+molesNO_n50+molesPNO_n50+molesN2O_n50+molesPN2O_n50+molesN2_n50)
#plt.bar(pH_n50+.2,molesNH4_n50,width,color='crimson',alpha=.8,bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50+molesPHNO2_n50+molesNO_n50+molesPNO_n50+molesN2O_n50+molesPN2O_n50+molesN2_n50+molesPN2_n50)
#plt.bar(pH_n50+.2,molesNH3_n50,width,color='crimson',alpha=.4,bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50+molesPHNO2_n50+molesNO_n50+molesPNO_n50+molesN2O_n50+molesPN2O_n50+molesN2_n50+molesPN2_n50+molesNH4_n50)
#plt.bar(pH_n50+.2,molesPNH3_n50,width,color='crimson',alpha=.4,hatch='/',bottom=molesNO3_n50+molesHNO3_n50+molesPHNO3_n50+molesNO2_n50+molesHNO2_n50+molesPHNO2_n50+molesNO_n50+molesPNO_n50+molesN2O_n50+molesPN2O_n50+molesN2_n50+molesPN2_n50+molesNH4_n50+molesNH3_n50)


#H2=1E-60
plt.bar(pH_n60+.2,molesNO3_n60,width,color='royalblue',alpha=.8)
plt.bar(pH_n60+.2,molesHNO3_n60,width,color='royalblue',alpha=.4,bottom=molesNO3_n60)
plt.bar(pH_n60+.2,molesPHNO3_n60,width,color='royalblue',alpha=.4,hatch='/',bottom=molesNO3_n60+molesHNO3_n60)
plt.bar(pH_n60+.2,molesNO2_n60,width,color='mediumseagreen',alpha=.8,bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60)
plt.bar(pH_n60+.2,molesHNO2_n60,width,color='mediumseagreen',alpha=.4,bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60)
plt.bar(pH_n60+.2,molesPHNO2_n60,width,color='mediumseagreen',alpha=.4,hatch='/',bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60)
plt.bar(pH_n60+.2,molesNO_n60,width,color='gold',alpha=.4,bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60+molesPHNO2_n60)
plt.bar(pH_n60+.2,molesPNO_n60,width,color='gold',alpha=.4,hatch='/',bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60+molesPHNO2_n60+molesNO_n60)
plt.bar(pH_n60+.2,molesN2O_n60,width,color='aqua',alpha=.4,bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60+molesPHNO2_n60+molesNO_n60+molesPNO_n60)

plt.bar(pH_n60+.2,molesPN2O_n60,width,color='aqua',alpha=.4,hatch='/',bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60+molesPHNO2_n60+molesNO_n60+molesPNO_n60+molesN2O_n60)
plt.bar(pH_n60+.2,molesN2_n60,width,color='darkviolet',alpha=.4,bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60+molesPHNO2_n60+molesNO_n60+molesPNO_n60+molesN2O_n60+molesPN2O_n60)
plt.bar(pH_n60+.2,molesPN2_n60,width,color='darkviolet',alpha=.4,hatch='/',bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60+molesPHNO2_n60+molesNO_n60+molesPNO_n60+molesN2O_n60+molesPN2O_n60+molesN2_n60)
plt.bar(pH_n60+.2,molesNH4_n60,width,color='crimson',alpha=.8,bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60+molesPHNO2_n60+molesNO_n60+molesPNO_n60+molesN2O_n60+molesPN2O_n60+molesN2_n60+molesPN2_n60)
plt.bar(pH_n60+.2,molesNH3_n60,width,color='crimson',alpha=.4,bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60+molesPHNO2_n60+molesNO_n60+molesPNO_n60+molesN2O_n60+molesPN2O_n60+molesN2_n60+molesPN2_n60+molesNH4_n60)
plt.bar(pH_n60+.2,molesPNH3_n60,width,color='crimson',alpha=.4,hatch='/',bottom=molesNO3_n60+molesHNO3_n60+molesPHNO3_n60+molesNO2_n60+molesHNO2_n60+molesPHNO2_n60+molesNO_n60+molesPNO_n60+molesN2O_n60+molesPN2O_n60+molesN2_n60+molesPN2_n60+molesNH4_n60+molesNH3_n60)



#H2=1E-80
plt.bar(pH_n80+.4,molesNO3_n80,width,color='royalblue',alpha=.8)
plt.bar(pH_n80+.4,molesHNO3_n80,width,color='royalblue',alpha=.4,bottom=molesNO3_n80)
plt.bar(pH_n80+.4,molesPHNO3_n80,width,color='royalblue',alpha=.4,hatch='/',bottom=molesNO3_n80+molesHNO3_n80)
plt.bar(pH_n80+.4,molesNO2_n80,width,color='mediumseagreen',alpha=.8,bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80)
plt.bar(pH_n80+.4,molesHNO2_n80,width,color='mediumseagreen',alpha=.4,bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80)
plt.bar(pH_n80+.4,molesPHNO2_n80,width,color='mediumseagreen',alpha=.4,hatch='/',bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80)
plt.bar(pH_n80+.4,molesNO_n80,width,color='gold',alpha=.4,bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80+molesPHNO2_n80)
plt.bar(pH_n80+.4,molesPNO_n80,width,color='gold',alpha=.4,hatch='/',bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80+molesPHNO2_n80+molesNO_n80)
plt.bar(pH_n80+.4,molesN2O_n80,width,color='aqua',alpha=.4,bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80+molesPHNO2_n80+molesNO_n80+molesPNO_n80)

plt.bar(pH_n80+.4,molesPN2O_n80,width,color='aqua',alpha=.4,hatch='/',bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80+molesPHNO2_n80+molesNO_n80+molesPNO_n80+molesN2O_n80)
plt.bar(pH_n80+.4,molesN2_n80,width,color='darkviolet',alpha=.4,bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80+molesPHNO2_n80+molesNO_n80+molesPNO_n80+molesN2O_n80+molesPN2O_n80)
plt.bar(pH_n80+.4,molesPN2_n80,width,color='darkviolet',alpha=.4,hatch='/',bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80+molesPHNO2_n80+molesNO_n80+molesPNO_n80+molesN2O_n80+molesPN2O_n80+molesN2_n80)
plt.bar(pH_n80+.4,molesNH4_n80,width,color='crimson',alpha=.8,bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80+molesPHNO2_n80+molesNO_n80+molesPNO_n80+molesN2O_n80+molesPN2O_n80+molesN2_n80+molesPN2_n80)
plt.bar(pH_n80+.4,molesNH3_n80,width,color='crimson',alpha=.4,bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80+molesPHNO2_n80+molesNO_n80+molesPNO_n80+molesN2O_n80+molesPN2O_n80+molesN2_n80+molesPN2_n80+molesNH4_n80)
plt.bar(pH_n80+.4,molesPNH3_n80,width,color='crimson',alpha=.4,hatch='/',bottom=molesNO3_n80+molesHNO3_n80+molesPHNO3_n80+molesNO2_n80+molesHNO2_n80+molesPHNO2_n80+molesNO_n80+molesPNO_n80+molesN2O_n80+molesPN2O_n80+molesN2_n80+molesPN2_n80+molesNH4_n80+molesNH3_n80)

#plt.xlim(0,16)
#plt.legend(loc='right')
plt.xlabel('pH',fontsize=16)
plt.ylabel('Mole fraction',fontsize=16)

plt.show()
plt.close()

redpow=np.array([1.0E0,1.0E-10,1.0E-20,1.0E-30,1.0E-40,1.0E-60,1.0E-80])
x=np.array([1,2,3,4,5,6,7])

plt.figure()
plt.plot(x,redpow,marker='o',ls='-',color='k')
plt.ylabel('Reducing Power ([H$_2$])',fontsize=16)
plt.yscale('log')
plt.xlim(0,8)
plt.ylim(1.0E-90,1.0E1)
plt.xticks([])
plt.show()
plt.close()


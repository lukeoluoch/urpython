#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('RunSix.csv')
df


# In[76]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
Time = df['Time']
CoCIN = df['CoCIN']
CoCOUT =df['CoCOUT']
CoHIN=df['CoHIN']
CoHOUT=df['CoHOUT']
ConCIN=df['ConCIN']
ConCOUT=df['ConCOUT']
ConHIN=df['ConHIN']
ConHOUT=df['ConHOUT']
SiCIN=df['SiCIN']
SiCOUT=df['SiCOUT']
SiHIN=df['SiHIN']
SiHOUT=df['SiHOUT']
MPCIN=df['MPCIN']
MPCOUT=df['MPCOUT']
MPHIN=df['MPHIN']
MPHOUT=df['MPHOUT']
plt.figure(figsize = (15,15))
plt.suptitle('Temperature vs Time in Heat Exchanger Configurations')
plt.rcParams.update({'font.size':25})

plt.style.use('ggplot')
plt.subplot(221)
plt.title('Co Current Shell and Tube')
plt.plot(Time/3600,CoCIN,label ='Cold In',linewidth=1,color = 'black')
plt.plot(Time/3600,CoCOUT,label ='Cold Out',linestyle = ':',linewidth=1,color = 'red')
plt.plot(Time/3600,CoHIN,label='Hot In',linestyle = '--',linewidth=1,color = 'green')
plt.plot(Time/3600,CoHOUT,label='Hot Out',linestyle = '-.',linewidth=1,color = 'blue')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(222)
plt.title('Counter Current Shell and Tube')
plt.plot(Time/3600,ConCIN,label ='Cold In',linewidth=1,color = 'black')
plt.plot(Time/3600,ConCOUT,label ='Cold Out',linestyle = ':',linewidth=1,color = 'red')
plt.plot(Time/3600,ConHIN,label='Hot In',linestyle = '--',linewidth=1,color = 'green')
plt.plot(Time/3600,ConHOUT,label='Hot Out',linestyle = '-.',linewidth=1,color = 'blue')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(223)
plt.title('Counter Current Double Pipe')
plt.plot(Time/3600,SiCIN,label ='Cold In',linewidth=1,color = 'black')
plt.plot(Time/3600,SiCOUT,label ='Cold Out',linestyle = ':',linewidth=1,color = 'red')
plt.plot(Time/3600,SiHIN,label='Hot In',linestyle = '--',linewidth=1,color = 'green')
plt.plot(Time/3600,SiHOUT,label='Hot Out',linestyle = '-.',linewidth=1,color = 'blue')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(224)
plt.title('Multipass Shell and Tube')
plt.plot(Time/3600,MPCIN,label ='Cold In',linewidth=1,color = 'black')
plt.plot(Time/3600,MPCOUT,label ='Cold Out',linestyle = ':',linewidth=1,color = 'red')
plt.plot(Time/3600,MPHIN,label='Hot In',linestyle = '--',linewidth=1,color = 'green')
plt.plot(Time/3600,MPHOUT,label='Hot Out',linestyle = '-.',linewidth=1,color = 'blue')
plt.legend(loc='best')
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()


# NTU Calculations

# In[111]:


def UA_Calc(t1,t2,Vcold,Vhot):
    Time = df['Time'][t1:t2]
    CoCIN = df['CoCIN'][t1:t2]
    CoCOUT =df['CoCOUT'][t1:t2]
    CoHIN=df['CoHIN'][t1:t2]
    CoHOUT=df['CoHOUT'][t1:t2]
    ConCIN=df['ConCIN'][t1:t2]
    ConCOUT=df['ConCOUT'][t1:t2]
    ConHIN=df['ConHIN'][t1:t2]
    ConHOUT=df['ConHOUT'][t1:t2]
    SiCIN=df['SiCIN'][t1:t2]
    SiCOUT=df['SiCOUT'][t1:t2]
    SiHIN=df['SiHIN'][t1:t2]
    SiHOUT=df['SiHOUT'][t1:t2]
    MPCIN=df['MPCIN'][t1:t2]
    MPCOUT=df['MPCOUT'][t1:t2]
    MPHIN=df['MPHIN'][t1:t2]
    MPHOUT=df['MPHOUT'][t1:t2]

    plt.figure(figsize = (15,15))
    plt.suptitle('Temperature vs Time in Heat Exchanger Configurations')
    plt.subplot(221)
    plt.title('Co Current Shell and Tube')
    plt.plot(Time,CoCIN,label ='Cold In',linewidth=1,color = 'black')
    plt.plot(Time,CoCOUT,label ='Cold Out',linestyle = ':',linewidth=1,color = 'red')
    plt.plot(Time,CoHIN,label='Hot In',linestyle = '--',linewidth=1,color = 'green')
    plt.plot(Time,CoHOUT,label='Hot Out',linestyle = '-.',linewidth=1,color = 'blue')
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplot(222)
    plt.title('Counter Current Shell and Tube')
    plt.plot(Time,ConCIN,label ='Cold In',linewidth=1,color = 'black')
    plt.plot(Time,ConCOUT,label ='Cold Out',linestyle = ':',linewidth=1,color = 'red')
    plt.plot(Time,ConHIN,label='Hot In',linestyle = '--',linewidth=1,color = 'green')
    plt.plot(Time,ConHOUT,label='Hot Out',linestyle = '-.',linewidth=1,color = 'blue')
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplot(223)
    plt.title('Counter Current Double Pipe')
    plt.plot(Time,SiCIN,label ='Cold In',linewidth=1,color = 'black')
    plt.plot(Time,SiCOUT,label ='Cold Out',linestyle = ':',linewidth=1,color = 'red')
    plt.plot(Time,SiHIN,label='Hot In',linestyle = '--',linewidth=1,color = 'green')
    plt.plot(Time,SiHOUT,label='Hot Out',linestyle = '-.',linewidth=1,color = 'blue')
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplot(224)
    plt.title('Multipass Shell and Tube')
    plt.plot(Time,MPCIN,label ='Cold In',linewidth=1,color = 'black')
    plt.plot(Time,MPCOUT,label ='Cold Out',linestyle = ':',linewidth=1,color = 'red')
    plt.plot(Time,MPHIN,label='Hot In',linestyle = '--',linewidth=1,color = 'green')
    plt.plot(Time,MPHOUT,label='Hot Out',linestyle = '-.',linewidth=1,color = 'blue')
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
    plt.show()


    #grams/second
    mcold = Vcold*63.09 
    mhot = Vhot*63.09 

     #J/gK
    cp = 4.19

    if mcold == mhot:
        Cr = 1
        Cmin = mcold*cp
    if mcold < mhot:
        Cr = mcold/mhot
        Cmin = mcold*cp
    if mcold > mhot:
        Cr = mhot/mcold
        Cmin = mhot*cp

    #CoCurrent
    q = mcold*cp*(CoCOUT-CoCIN)
    qmax = mcold*cp*(CoHIN-CoCIN)
    eff = q/qmax
    E = (2/eff-(1+Cr))/(1+Cr**(2))**(-1/2)
    NTU = -(1+Cr**(2))**(-1/2)*np.log((E-1)/(E+1))
    #W/K
    UA = NTU*Cmin
    UAA_Co = np.sum(UA)/len(UA)
    SE = ss.sem(UA)
    conf_Co = ss.norm.ppf(1-0.025)*SE
    print('UA, for the cocurrent shell and tube is {} +- {}.'.format(UAA_Co,conf_Co))


    #CounterCurrent 
    q = mcold*cp*(ConCOUT-ConCIN)
    qmax = mcold*cp*(ConHIN-ConCIN)
    eff = q/qmax
    E = (2/eff-(1+Cr))/(1+Cr**(2))**(-1/2)
    NTU = -(1+Cr**(2))**(-1/2)*np.log((E-1)/(E+1))
    UA = NTU*Cmin
    UAA_Con = np.sum(UA)/len(UA)
    SE = ss.sem(UA)
    conf_Con = ss.norm.ppf(1-0.025)*SE
    print('UA, for the countercurrent shell and tube is {} +- {}.'.format(UAA_Con,conf_Con))


    #Multipass
    q = mcold*cp*(MPCOUT-MPCIN)
    qmax = mcold*cp*(MPHIN-MPCIN)
    eff = q/qmax
    E = (2/eff-(1+Cr))/(1+Cr**(2))**(-1/2)
    NTU = -(1+Cr**(2))**(-1/2)*np.log((E-1)/(E+1))
    UA = NTU*Cmin
    UAA_MP = np.sum(UA)/len(UA)
    SE = ss.sem(UA)
    conf_MP = ss.norm.ppf(1-0.025)*SE
    print('UA, for the multipass shell and tube is {} +- {}.'.format(UAA_MP,conf_MP))
    UAA = np.array([UAA_Co,UAA_Con,UAA_MP])
    conf = np.array([conf_Co,conf_Con,conf_MP])
    return UAA, conf


# In[176]:


#Test 1 
#0.2 GPM Cold , 0.2 GPM Hot
t1 = 5180
t2 = 5238
Vcold = 0.2
Vhot = 0.2
UA_1, conf_1 = UA_Calc(t1,t2,Vcold,Vhot)


# In[177]:


#Test 2
#0.2 GPM Cold , 0.4 GPM Hot
t1 = 5438
t2 = 5490
Vcold = 0.2
Vhot = 0.4
UA_2, conf_2 = UA_Calc(t1,t2,Vcold,Vhot)


# In[178]:


#Test 3
#0.2 GPM Cold , 0.6 GPM Hot
t1 = 5650
t2 = 5753
Vcold = 0.2
Vhot = 0.6
UA_3, conf_3 = UA_Calc(t1,t2,Vcold,Vhot)


# In[179]:


#Test 4
#0.2 GPM Cold , 0.8 GPM Hot
t1 = 5950
t2 = 6050
Vcold = 0.2
Vhot = 0.8
UA_4, conf_4 = UA_Calc(t1,t2,Vcold,Vhot)


# In[180]:


#Test 5
#0.2 GPM Cold , 1.0 GPM Hot
t1 = 6300
t2 = 6384
Vcold = 0.2
Vhot = 1.0
UA_5, conf_5 = UA_Calc(t1,t2,Vcold,Vhot)


# In[181]:


#Test 6
#0.4 GPM Cold , 0.2 GPM Hot
t1 = 6650
t2 = 6694
Vcold = 0.4
Vhot = 0.2
UA_6, conf_6 = UA_Calc(t1,t2,Vcold,Vhot)


# In[182]:


#Test 7
#0.4 GPM Cold , 0.4 GPM Hot
t1 = 6800
t2 = 7000
Vcold = 0.4
Vhot = 0.4
UA_7, conf_7 = UA_Calc(t1,t2,Vcold,Vhot)


# In[183]:


#Test 8
#0.4 GPM Cold , 0.6 GPM Hot
t1 = 7400
t2 = 7480
Vcold = 0.4
Vhot = 0.6
UA_8, conf_8 = UA_Calc(t1,t2,Vcold,Vhot)


# In[184]:


#Test 9
#0.4 GPM Cold , 0.8 GPM Hot
t1 = 7560
t2 = 7650
Vcold = 0.4
Vhot = 0.8
UA_9, conf_9 = UA_Calc(t1,t2,Vcold,Vhot)


# In[185]:


#Test 10
#0.4 GPM Cold , 1.0 GPM Hot
t1 = 8000
t2 = 8060
Vcold = 0.4
Vhot = 1.0
UA_10, conf_10 = UA_Calc(t1,t2,Vcold,Vhot)


# In[186]:


#Test 11
#0.6 GPM Cold , 0.2 GPM Hot
t1 = 8299
t2 = 8349
Vcold = 0.6
Vhot = 0.2
UA_11, conf_11 = UA_Calc(t1,t2,Vcold,Vhot)


# In[187]:


#Test 12
#0.6 GPM Cold , 0.4 GPM Hot
t1 = 8475
t2 = 8537
Vcold = 0.6
Vhot = 0.4
UA_12, conf_12 = UA_Calc(t1,t2,Vcold,Vhot)


# In[188]:


#Test 13
#0.6 GPM Cold , 0.6 GPM Hot
t1 = 8700
t2 = 8752
Vcold = 0.6
Vhot = 0.6
UA_13, conf_13 = UA_Calc(t1,t2,Vcold,Vhot)


# In[189]:


#Test 14
#0.6 GPM Cold , 0.8 GPM Hot
t1 = 8950
t2 = 9030
Vcold = 0.6
Vhot = 0.8
UA_14, conf_14 = UA_Calc(t1,t2,Vcold,Vhot)


# In[190]:


#Test 15
#0.6 GPM Cold , 1.0 GPM Hot
t1 = 9250
t2 = 9344
Vcold = 0.6
Vhot = 1.0
UA_15, conf_15 = UA_Calc(t1,t2,Vcold,Vhot)


# In[191]:


#Test 16
#0.8 GPM Cold , 0.2 GPM Hot
t1 = 9575
t2 = 9621
Vcold = 0.8
Vhot = 0.2
UA_16, conf_16 = UA_Calc(t1,t2,Vcold,Vhot)

#Test did not go fully to steady state rip.


# In[192]:


#Test 17
#0.8 GPM Cold , 0.4 GPM Hot
t1 = 9730
t2 = 9772
Vcold = 0.8
Vhot = 0.4
UA_17, conf_17 = UA_Calc(t1,t2,Vcold,Vhot)


# In[193]:


#Test 18
#0.8 GPM Cold , 0.6 GPM Hot
t1 = 9900
t2 = 10006
Vcold = 0.8
Vhot = 0.6
UA_18, conf_18 = UA_Calc(t1,t2,Vcold,Vhot)


# In[194]:


#Test 19
#0.8 GPM Cold , 0.8 GPM Hot
t1 = 10250
t2 = 10330
Vcold = 0.8
Vhot = 0.8
UA_19, conf_19 = UA_Calc(t1,t2,Vcold,Vhot)


# In[195]:


#Test 20
#0.8 GPM Cold , 1.0 GPM Hot
t1 = 10550
t2 = -1
Vcold = 0.8
Vhot = 1.0
UA_20, conf_20 = UA_Calc(t1,t2,Vcold,Vhot)


# In[200]:


UAdata = np.vstack((UA_1,UA_2,UA_3,UA_4,UA_5,UA_6,UA_7,UA_8,UA_9,UA_10,UA_11,UA_12,UA_13,UA_14,UA_15,UA_16,UA_17,UA_18,UA_19,UA_20))
conf_data = np.vstack((conf_1,conf_2,conf_3,conf_4,conf_5,conf_6,conf_7,conf_8,conf_9,conf_10,conf_11,conf_12,conf_13,conf_14,conf_15,conf_16))
print(UAdata)
print(conf_data)


# In[ ]:


plt.


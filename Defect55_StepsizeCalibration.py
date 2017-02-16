# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:03:39 2016

@author: dfs1
"""

import numpy as np
import CDSAXSfunctions as CD
import CDplot as CDp

Intensity=np.loadtxt('Defect55_Center_16_Int.txt')
Qx = np.loadtxt('Defect55_Center_16_Qx.txt')
Qz = np.loadtxt('Defect55_Center_16_Qz.txt')


Trapnumber = 6

DW = 1.1
I0 = 9e-5
Bk =1
Pitch = 110
SLD1 = 1; SLD2 = 1.4;
TPAR=np.zeros([Trapnumber+1,2])
SLD=np.zeros([Trapnumber+1,1])
SPAR=np.zeros(3)
SPAR[0]=DW; SPAR[1]=I0; SPAR[2]=Bk;


TPAR[0,0]=84.2; TPAR[0,1]=4.4; SLD[0,0]=SLD1;
TPAR[1,0]=71.8; TPAR[1,1]=38.8;SLD[1,0]=SLD1;
TPAR[2,0]=66.7; TPAR[2,1]=1.8; SLD[2,0]=SLD2;
TPAR[3,0]=72.6; TPAR[3,1]=1.6; SLD[3,0]=SLD2;
TPAR[4,0]=70.4; TPAR[4,1]=30.8;SLD[4,0]=SLD2;
TPAR[5,0]=58.3; TPAR[5,1]=5.1; SLD[5,0]=SLD2;
TPAR[6,0]=36.5; TPAR[6,1]=0; 

Coord=CD.LAM1CoordAssign(TPAR,SLD,Trapnumber,Pitch)
CDp.plotLAM1(Coord,Trapnumber,Pitch)

(FITPAR,FITPARLB,FITPARUB)=CD.PBA_LAM1(TPAR,SPAR,Trapnumber)

MCPAR=np.zeros([7])
MCPAR[0] = 1 # Chainnumber
MCPAR[1] = len(FITPAR)
MCPAR[2] = 5000 #stepnumber
MCPAR[3] = 1 #randomchains
MCPAR[4] = 1 # Resampleinterval
MCPAR[5] = 20 # stepbase
MCPAR[6] = 20 # steplength

def SimInt_LAM1(FITPAR):
    TPARs=np.zeros([Trapnumber+1,2])
    TPARs[:,0:2]=np.reshape(FITPAR[0:(Trapnumber+1)*2],(Trapnumber+1,2))
    SPAR=FITPAR[Trapnumber*2+2:Trapnumber*2+5]
    (Coord)= CD.LAM1CoordAssign(TPAR,SLD,Trapnumber,Pitch)
    F1 = CD.FreeFormTrapezoid(Coord[:,:,0],Qx,Qz,Trapnumber) 
    M=np.power(np.exp(-1*(np.power(Qx,2)+np.power(Qz,2))*np.power(SPAR[0],2)),0.5)
    Formfactor=F1*M
    Formfactor=abs(Formfactor)
    SimInt = np.power(Formfactor,2)*SPAR[1]+SPAR[2]
    return SimInt



def MCMCInit_LAM1(FITPAR,FITPARLB,FITPARUB,MCPAR):
    
    MCMCInit=np.zeros([int(MCPAR[0]),int(MCPAR[1])+1])
    for i in range(int(MCPAR[0])):
        if i <MCPAR[3]: #reversed from matlab code assigns all chains below randomnumber as random chains
            for c in range(int(MCPAR[1])):
                MCMCInit[i,c]=FITPARLB[c]+(FITPARUB[c]-FITPARLB[c])*np.random.random_sample()
                SimInt=SimInt_LAM1(MCMCInit[i,:])
            C=np.sum(CD.Misfit(Intensity,SimInt))
            
            MCMCInit[i,int(MCPAR[1])]=C
            
        else:
            MCMCInit[i,0:int(MCPAR[1])]=FITPAR
            SimInt=SimInt_LAM1(MCMCInit[i,:])
            C=np.sum(CD.Misfit(Intensity,SimInt))
            MCMCInit[i,int(MCPAR[1])]=C
            
           
    return MCMCInit

def MCMC_LAM1(MCMC_List):
    
    MCMCInit=MCMC_List
    
    L = int(MCPAR[1])
    Stepnumber= int(MCPAR[2])
        
    SampledMatrix=np.zeros([Stepnumber,L+1]) 
    SampledMatrix[0,:]=MCMCInit
    Move = np.zeros([L+1])
    
    ChiPrior = MCMCInit[L]
    for step in np.arange(1,Stepnumber,1): 
        Temp = SampledMatrix[step-1,:].copy()
        for p in range(L-1):
            StepControl = MCPAR[5]+MCPAR[6]*np.random.random_sample()
            Move[p] = (FITPARUB[p]-FITPARLB[p])/StepControl*(np.random.random_sample()-0.5) # need out of bounds check
            Temp[p]=Temp[p]+Move[p]
            if Temp[p] < FITPARLB[p]:
                Temp[p]=FITPARLB[p]+(FITPARUB[p]-FITPARLB[p])/1000
            elif Temp[p] > FITPARUB[p]:
                Temp[p]=FITPARUB[p]-(FITPARUB[p]-FITPARLB[p])/1000
        SimPost=SimInt_LAM1(Temp)
        ChiPost=np.sum(CD.Misfit(Intensity,SimPost))
        if ChiPost < ChiPrior:
            SampledMatrix[step,0:L]=Temp[0:L]
            SampledMatrix[step,L]=ChiPost
            ChiPrior=ChiPost
            
        else:
            MoveProb = np.exp(-0.5*np.power(ChiPost-ChiPrior,2))
            if np.random.random_sample() < MoveProb:
                SampledMatrix[step,0:L]=Temp[0:L]
                SampledMatrix[step,L]=ChiPost
                ChiPrior=ChiPost
            else:
                SampledMatrix[step,:]=SampledMatrix[step-1,:]
    AcceptanceNumber=0;
    Acceptancetotal=len(SampledMatrix[:,1])

    for i in np.arange(1,len(SampledMatrix[:,1]),1):
        if SampledMatrix[i,0] != SampledMatrix[i-1,0]:
            AcceptanceNumber=AcceptanceNumber+1
    AcceptanceProbability=AcceptanceNumber/Acceptancetotal
    print(AcceptanceProbability)
    ReSampledMatrix=np.zeros([int(MCPAR[2])/int(MCPAR[4]),len(SampledMatrix[1,:])])

    c=-1
    for i in np.arange(0,len(SampledMatrix[:,1]),MCPAR[4]):
        c=c+1
        ReSampledMatrix[c,:]=SampledMatrix[i,:]
    return (ReSampledMatrix)
    
    
    
MCMCInitial=MCMCInit_LAM1(FITPAR,FITPARLB,FITPARUB,MCPAR)

Acceptprob=0;
while Acceptprob < 0.3 or Acceptprob > 0.4:
       L = int(MCPAR[1])
       Stepnumber= int(MCPAR[2])
        
       SampledMatrix=np.zeros([Stepnumber,L+1]) 
       SampledMatrix[0,:]=MCMCInitial[0,:]
       Move = np.zeros([L+1])
    
       ChiPrior = MCMCInitial[0,L]
       for step in np.arange(1,Stepnumber,1): 
           Temp = SampledMatrix[step-1,:].copy()
           for p in range(L-1):
               StepControl = MCPAR[5]+MCPAR[6]*np.random.random_sample()
               Move[p] = (FITPARUB[p]-FITPARLB[p])/StepControl*(np.random.random_sample()-0.5) # need out of bounds check
               Temp[p]=Temp[p]+Move[p]
               if Temp[p] < FITPARLB[p]:
                   Temp[p]=FITPARLB[p]+(FITPARUB[p]-FITPARLB[p])/1000
               elif Temp[p] > FITPARUB[p]:
                   Temp[p]=FITPARUB[p]-(FITPARUB[p]-FITPARLB[p])/1000
           (SimPost)=SimInt_LAM1(Temp)
           ChiPost=np.sum(CD.Misfit(Intensity,SimPost))
           if ChiPost < ChiPrior:
               SampledMatrix[step,0:L]=Temp[0:L]
               SampledMatrix[step,L]=ChiPost
               ChiPrior=ChiPost
           
           else:
               MoveProb = np.exp(-0.5*np.power(ChiPost-ChiPrior,2))
               if np.random.random_sample() < MoveProb:
                   SampledMatrix[step,0:L]=Temp[0:L]
                   SampledMatrix[step,L]=ChiPost
                   ChiPrior=ChiPost
               else:
                   SampledMatrix[step,:]=SampledMatrix[step-1,:]
       AcceptanceNumber=0
       Acceptancetotal=len(SampledMatrix[:,1])
       for i in np.arange(1,len(SampledMatrix[:,1]),1):
           if SampledMatrix[i,0] != SampledMatrix[i-1,0]:
               AcceptanceNumber=AcceptanceNumber+1
       Acceptprob=AcceptanceNumber/Acceptancetotal
       print(Acceptprob,MCPAR[5],MCPAR[6])
       if Acceptprob < 0.3:
           MCPAR[5]=MCPAR[5]+1
           MCPAR[6]=MCPAR[6]+1
       if Acceptprob > 0.4:
           MCPAR[5]=MCPAR[5]-1
           MCPAR[6]=MCPAR[6]-1
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:03:39 2016

@author: dfs1
"""

import numpy as np
import CDSAXSfunctions as CD
import CDplot as CDp
from multiprocessing import Pool
import time
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
MCPAR[0] = 12 # Chainnumber
MCPAR[1] = len(FITPAR)
MCPAR[2] = 10 #stepnumber
MCPAR[3] = 1 #randomchains
MCPAR[4] = 1 # Resampleinterval
MCPAR[5] = 100 # stepbase
MCPAR[6] = 300 # steplength

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
    
def Uncertainty1T(ReSampledMatrix):
        Xi = np.zeros([101,2,len(ReSampledMatrix[:,0])])
        Yi = np.zeros([101,1,len(ReSampledMatrix[:,0])])
        PopWidth= np.zeros([len(ReSampledMatrix[:,0])])
        PopHeight=np.zeros([len(ReSampledMatrix[:,0])])
        TrapHeight=np.zeros([Trapnumber+1,len(ReSampledMatrix[:,0])])
        for PopNumber in range(len(ReSampledMatrix[:,0])):
            
            
            TPARU=np.zeros([Trapnumber+1,2])
            TPARU[:,0:2]=np.reshape(ReSampledMatrix[PopNumber,0:(Trapnumber+1)*2],(Trapnumber+1,2))
           
            (CoordUnc)= CD.LAM1CoordAssign(TPARU,SLD,Trapnumber,Pitch)
            
            
            for i in np.arange(1,Trapnumber+1,1):
                TrapHeight[i,PopNumber]=TrapHeight[i-1,PopNumber]+TPARU[i-1,1]
            
            for LineNumber in range(1):
                
                EffTrapnumber=0
                #LeftSide
                X1L = CoordUnc[EffTrapnumber,0,LineNumber]
                X2L = CoordUnc[EffTrapnumber+1,0,LineNumber]
                X1R = CoordUnc[EffTrapnumber,1,LineNumber]
                X2R = CoordUnc[EffTrapnumber+1,1,LineNumber]
                Y1 = TrapHeight[EffTrapnumber,PopNumber]
                Y2 = TrapHeight[EffTrapnumber+1,PopNumber]
                Disc=0
                for c in np.arange(0,101,1):
                    
                    if Disc > Y2 and Disc <TrapHeight[Trapnumber,PopNumber]:
                        EffTrapnumber=EffTrapnumber+1
                        
                        X1L = CoordUnc[EffTrapnumber,0,LineNumber]
                        X2L = CoordUnc[EffTrapnumber+1,0,LineNumber]
                        X1R = CoordUnc[EffTrapnumber,1,LineNumber]
                        X2R = CoordUnc[EffTrapnumber+1,1,LineNumber]
                        Y1 = TrapHeight[EffTrapnumber,PopNumber]
                        Y2 = TrapHeight[EffTrapnumber+1,PopNumber]
                    ML = (Y2-Y1)/(X2L-X1L)
                    MR=  (Y2-Y1)/(X2R-X1R)
                    BL = Y1-ML*X1L
                    BR = Y1-MR*X1R
                    Xi[c,0,PopNumber]=(Disc-BL)/ML
                    Xi[c,1,PopNumber]=(Disc-BR)/MR
                    Yi[c,0,PopNumber]=Disc
                    Disc=Disc+TrapHeight[Trapnumber,PopNumber]/100
            Xi[:,:,PopNumber]=Xi[:,:,PopNumber]-(Xi[0,1,PopNumber]-Xi[0,0,PopNumber])/2
            PopWidth[PopNumber]=Xi[50,1,PopNumber]-Xi[50,0,PopNumber]
            PopHeight[PopNumber]=Yi[100,0,PopNumber]
            
        S=np.std(Xi,2)*1.96
        Center = np.average(Xi,2)
        Sy=np.std(Yi,2)*1.96
        YC=np.average(Yi,2)
        OuterEdge=Center
        YInner=YC-Sy
        YOuter=YC+Sy
        OuterEdge[:,0]=OuterEdge[:,0]-S[:,0]
        OuterEdge[:,1]=OuterEdge[:,1]+S[:,1]

        InnerEdge=Center
        InnerEdge[:,0]=InnerEdge[:,0]+S[:,0]
        InnerEdge[:,1]=InnerEdge[:,1]-S[:,1]

        LinePlot=np.zeros([2*101,2])
        InnerPlot=np.zeros([2*101,2])
        OuterPlot=np.zeros([2*101,2])
        
        LinePlot[0:101,0]=Center[:,0]
        LinePlot[101:202,0]=np.flipud(Center[:,1])
        LinePlot[0:101,1]=YC[:,0]
        LinePlot[101:202,1]=np.flipud(YC[:,0])
        
        InnerPlot[0:101,0]=InnerEdge[:,0]
        InnerPlot[101:202,0]=np.flipud(InnerEdge[:,1])
        InnerPlot[0:101,1]=YInner[:,0]
        InnerPlot[101:202,1]=np.flipud(YInner[:,0])
        
        
        OuterPlot[0:101,0]=OuterEdge[:,0]
        OuterPlot[101:202,0]=np.flipud(OuterEdge[:,1])
        OuterPlot[0:101,1]=YOuter[:,0]
        OuterPlot[101:202,1]=np.flipud(YOuter[:,0])
        
        vi=np.zeros([100,1])
        vo=np.zeros([100,1])
        vc=np.zeros([100,1])
        for l in np.arange(1,1+0.0001,2):
            for h in np.arange(1,101,1):
                vi[h-1,l/2]=0.5*(YInner[h,l-1]-YInner[h-1,l-1])*((InnerEdge[h-1,l]-InnerEdge[h-1,l-1])+(InnerEdge[h,l]-InnerEdge[h,l-1]))
                vo[h-1,l/2]=0.5*(YOuter[h,l-1]-YOuter[h-1,l-1])*((OuterEdge[h-1,l]-OuterEdge[h-1,l-1])+(OuterEdge[h,l]-OuterEdge[h,l-1]))
                vc[h-1,l/2]=0.5*(YC[h,l-1]-YC[h-1,l-1])*((Center[h-1,l]-Center[h-1,l-1])+(Center[h,l]-Center[h,l-1]))
        vd=vo-vi
        vt=np.sum(vd)
        vct=np.sum(vc)
        WidthAvg=np.average(PopWidth)
        HeightAvg=np.average(PopHeight)
        WidthStd=np.std(PopWidth)
        HeightStd=np.std(PopHeight)
        
        viLower=np.zeros([10,1])
        voLower=np.zeros([10,1])
        vcLower=np.zeros([10,1])
        for l in np.arange(1,1+0.0001,2):
            for h in np.arange(1,11,1):
                viLower[h-1,l/2]=0.5*(YInner[h,l-1]-YInner[h-1,l-1])*((InnerEdge[h-1,l]-InnerEdge[h-1,l-1])+(InnerEdge[h,l]-InnerEdge[h,l-1]))
                voLower[h-1,l/2]=0.5*(YOuter[h,l-1]-YOuter[h-1,l-1])*((OuterEdge[h-1,l]-OuterEdge[h-1,l-1])+(OuterEdge[h,l]-OuterEdge[h,l-1]))
                vcLower[h-1,l/2]=0.5*(YC[h,l-1]-YC[h-1,l-1])*((Center[h-1,l]-Center[h-1,l-1])+(Center[h,l]-Center[h,l-1]))
        vdLower=voLower-viLower
        vtLower=np.sum(vdLower)
        vctLower=np.sum(vcLower)
        
        viUpper=np.zeros([10,1])
        voUpper=np.zeros([10,1])
        vcUpper=np.zeros([10,1])
        for l in np.arange(1,1+0.0001,2):
            for h in np.arange(91,101,1):
                viUpper[h-91,l/2]=0.5*(YInner[h,l-1]-YInner[h-1,l-1])*((InnerEdge[h-1,l]-InnerEdge[h-1,l-1])+(InnerEdge[h,l]-InnerEdge[h,l-1]))
                voUpper[h-91,l/2]=0.5*(YOuter[h,l-1]-YOuter[h-1,l-1])*((OuterEdge[h-1,l]-OuterEdge[h-1,l-1])+(OuterEdge[h,l]-OuterEdge[h,l-1]))
                vcUpper[h-91,l/2]=0.5*(YC[h,l-1]-YC[h-1,l-1])*((Center[h-1,l]-Center[h-1,l-1])+(Center[h,l]-Center[h,l-1]))
        vdUpper=voUpper-viUpper
        vtUpper=np.sum(vdUpper)
        vctUpper=np.sum(vcUpper)
        return(vt,vct,vtLower,vctLower,vtUpper,vctUpper,WidthAvg,HeightAvg,WidthStd,HeightStd,LinePlot,InnerPlot,OuterPlot)    
    
MCMCInitial=MCMCInit_LAM1(FITPAR,FITPARLB,FITPARUB,MCPAR)

MCMC_List=[0]*int(MCPAR[0])
for i in range(int(MCPAR[0])):
    MCMC_List[i]=MCMCInitial[i,:]
    
    
start_time = time.perf_counter()
if __name__ =='__main__':  
    pool = Pool(processes=12)
    F=pool.map(MCMC_LAM1,MCMC_List)
    F=tuple(F)
    np.save('LAMtest',F) # add savedfilename here
    end_time=time.perf_counter()   
    print(end_time-start_time)    
    ReSampledMatrix=F[0]
    (UNCT_Param)=Uncertainty1T(ReSampledMatrix)
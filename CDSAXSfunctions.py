# -*- coding: utf-8 -*-
"""
Functions to be used in analyzing CDSAXS data
"""
import numpy as np

import CDSAXSfunctions as CD
from scipy.interpolate import interp1d
    
    
def LAM1CoordAssign(TPAR,SLD,Trapnumber,Pitch):
    Coord=np.zeros([Trapnumber+1,5,1])
    for T in range (Trapnumber+1):
        if T==0:
            Coord[T,0,0]=0
            Coord[T,1,0]=TPAR[0,0]
            Coord[T,2,0]=TPAR[0,1]
            Coord[T,3,0]=0
            Coord[T,4,0]=SLD[0,0]
        else:
            Coord[T,0,0]=Coord[T-1,0,0]+0.5*(TPAR[T-1,0]-TPAR[T,0])
            Coord[T,1,0]=Coord[T,0,0]+TPAR[T,0]
            Coord[T,2,0]=TPAR[T,1]
            Coord[T,3,0]=0
            Coord[T,4,0]=SLD[T,0]
 
    return (Coord)
         
         
           
                        
def FreeFormTrapezoid(Coord,Qx,Qz,Trapnumber):
    H1 = Coord[0,3]
    H2 = Coord[0,3]
    form=np.zeros([len(Qx[:,1]),len(Qx[1,:])])
    for i in range(int(Trapnumber)):
        H2 = H2+Coord[i,2];
        if i > 0:
            H1 = H1+Coord[i-1,2]
        x1 = Coord[i,0]
        x4 = Coord[i,1]
        x2 = Coord[i+1,0]
        x3 = Coord[i+1,1]
        if x2==x1:
            x2=x2-0.000001
        if x4==x3:
            x4=x4-0.000001
        SL = Coord[i,2]/(x2-x1)
        SR = -Coord[i,2]/(x4-x3)
        
        A1 = (np.exp(1j*Qx*((H1-SR*x4)/SR))/(Qx/SR+Qz))*(np.exp(-1j*H2*(Qx/SR+Qz))-np.exp(-1j*H1*(Qx/SR+Qz)))
        A2 = (np.exp(1j*Qx*((H1-SL*x1)/SL))/(Qx/SL+Qz))*(np.exp(-1j*H2*(Qx/SL+Qz))-np.exp(-1j*H1*(Qx/SL+Qz)))
        form=form+(1j/Qx)*(A1-A2)*Coord[i,4]
    return form
    
def SCNIntensitySim(Coord,Qx,Qz,Trapnumber,DW,I0,Bk):
    F1 = FreeFormTrapezoid(Coord[:,:,0],Qx,Qz,Trapnumber)
    F2 = FreeFormTrapezoid(Coord[:,:,1],Qx,Qz,Trapnumber)
    F3 = FreeFormTrapezoid(Coord[:,:,2],Qx,Qz,Trapnumber)
    F4 = FreeFormTrapezoid(Coord[:,:,3],Qx,Qz,Trapnumber)
    Formfactor=(F1+F2+F3+F4)
    M=np.power(np.exp(-1*(np.power(Qx,2)+np.power(Qz,2))*np.power(DW,2)),0.5);
    Formfactor=Formfactor*M
    SimInt = np.power(abs(Formfactor),2)*I0+Bk
    return SimInt

def ParBoundSCN(tpar,ppar,SPAR,X):
    XL=X*0.95
    XU=X*0.95
    tparL=tpar*0.9
    tparU=tpar*1.1
    pparL=ppar*0.5
    pparU=ppar*2
    SPARL=SPAR*0.5
    SPARU=SPAR*2
    return(tparL,tparU,pparL,pparU,XL,XU,SPARL,SPARU)
    

def Misfit(Exp,Sim):
    Chi2= abs(np.log(Exp)-np.log(Sim))
    #ms=np.zeros([len(Exp[:,1]),len(Exp[1,:]),2])
    #ms[:,:,0]=Sim
    #ms[:,:,1]=Exp
    #MS= np.nanmin(ms,2)
    #Chi2=np.power((D/MS),2)
    Chi2[np.isnan(Chi2)]=0
    return Chi2
    
def PBA_LAM1(TPAR,SPAR,Trapnumber):
     
    SPARLB=SPAR[0:4]*0.8
    SPARUB=SPAR[0:4]*1.2

    FITPAR=TPAR[:,0:2].ravel()
    FITPARLB=FITPAR*0.8
    FITPARUB=FITPAR*1.2
    FITPAR=np.append(FITPAR,SPAR)
       
    FITPARLB=np.append(FITPARLB,SPARLB)
    
    FITPARUB=np.append(FITPARUB,SPARUB)
    
    return (FITPAR,FITPARLB,FITPARUB)
    
   


        

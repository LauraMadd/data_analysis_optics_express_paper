# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from scipy.special import factorial as fact

plt.style.use('seaborn-white')
#plt.xkcd()
plt.rcParams['image.cmap'] = 'gray'
font = {'family' : 'calibri',
        'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)
SMALL_SIZE = 25
MEDIUM_SIZE = 30
BIGGER_SIZE = 40

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#----------------------------------------Functions-------------------------------

#-------------Functions for Helmholtz---------------------
x = np.linspace(-1,1,100)

yy,xx= np.meshgrid(x,x)


def cart2pol(xx, yy):
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    return rho, phi

rho,phi = cart2pol(xx,yy)
rho_limit = rho>1
rho[rho_limit] = 0


def h_mode(n,m,rho,phi):
    mode = np.exp(1j*2*np.pi*m*np.sqrt(1-rho**2))*np.exp(1j*n*phi)/np.sqrt(1-rho**2)**0.25
    return mode

def helmholtzRadialComponent(n,m,r):
    r_lim = r > 1
    r[r_lim] = 0
    return np.exp(1j*2*np.pi*m*np.sqrt(1-r**2))/np.sqrt(1-r**2)**0.25

def nollIndex(n,m):
    return np.sum(np.arange(n+1))+np.arange(-n,n+1,2).tolist().index(m)

def HelmIndex(n,m):
    if n >= m:
        ind = n**2 + m
    else:
        ind = (m+1)**2 - 1 - n
    return ind
#---------------------------------------------------------
#-------------Functions to geneerate different bases------
def zernikeRadialComponent(n,m,r):
    k=np.arange((n-m)/2+1)
    k_facts=((-1)**k*fact(n-k))/(fact(k)*fact((n+m)/2-k)*fact((n-m)/2-k))
    k_facts=np.meshgrid(k_facts,np.zeros(r.shape))[0]
    k=np.meshgrid(k,np.zeros(r.shape))[0]
    return np.sum(k_facts*r[:,None]**(n-2*k),axis=1)

def lukoszRadialComponent(n,m,r):
    if m==n:
        return zernikeRadialComponent(n,m,r)
    else:
        return zernikeRadialComponent(n,m,r)-zernikeRadialComponent(n-2,m,r)


def generateAberrationDataset(res,base,order,existingN=0):
    xc,yc=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))
    r=np.sqrt(xc**2+yc**2).flatten()
    pupilCoords=np.where(r<=1.0)
    if base=="Zernike" or base=="Lukosz":
        t=np.arctan2(yc,xc).flatten()
        ns,ms=np.meshgrid(np.arange(0,order+1),np.arange(-order,order+1))
        ns_notzero=ns[np.where(np.logical_and(np.abs(ms)<=ns,(ns-ms)%2==0))]
        ms_notzero=ms[np.where(np.logical_and(np.abs(ms)<=ns,(ns-ms)%2==0))]
        dataset=np.zeros((res**2,ns_notzero.shape[0]-existingN),dtype="float32")

        for i in range(ns_notzero.shape[0]):
            ind=nollIndex(ns_notzero[i],ms_notzero[i])
            if ind>existingN:
                if ns_notzero[i]==0:
                    dataset[:,ind-existingN]=1.0
                else:
                    if base=="Zernike":
                        temp=zernikeRadialComponent(ns_notzero[i],np.abs(ms_notzero[i]),r)[pupilCoords]
                    elif base=="Lukosz":
                        temp=lukoszRadialComponent(ns_notzero[i],np.abs(ms_notzero[i]),r)[pupilCoords]

                    if ms_notzero[i]>0:
                        temp=(temp*np.cos(ms_notzero[i]*t[pupilCoords])).astype("float32")
                    elif ms_notzero[i]<0:
                        temp=(temp*np.sin(-ms_notzero[i]*t[pupilCoords])).astype("float32")
                    dataset[pupilCoords,ind-existingN]=((temp-np.min(temp))/(np.max(temp)-np.min(temp)))


    if base =="Helmholtz": #currently only for positive values of n,m
        t=np.arctan2(xc,yc).flatten()
        ns_hlm = np.arange(0,order)
        ms_hlm = np.arange(0,order)
        dataset=np.zeros((res**2,(ns_hlm.shape[0]-existingN)**2),dtype="float32")


        for i in range(ns_hlm.shape[0]):
            for j in range(ms_hlm.shape[0]):
                ind_hlm = HelmIndex(ns_hlm[i],ms_hlm[j])
                if ind_hlm == 0:
                    dataset[pupilCoords,ind_hlm-existingN]=1
                else:
                    temp = helmholtzRadialComponent(ns_hlm[i], ms_hlm[j], r)[pupilCoords]
                    temp = (temp*np.exp(1j*ns_hlm[i]*t[pupilCoords]))
                    temp = np.angle(temp)
                    dataset[pupilCoords,ind_hlm-existingN] = temp#(temp-np.min(temp))/(np.max(temp)-np.min(temp))


    return dataset

def getAberrationPhase(dataset,lam,coeffs):
    return np.dot(dataset[:,:coeffs.shape[0]],coeffs.astype("float32")).reshape((np.sqrt(dataset.shape[0]).astype("uint32"),np.sqrt(dataset.shape[0]).astype("uint32")))/(lam/1000.0)*2**16

#-------------------------------------------------------------------------------
'''Variables to change :
  type, date, canopy order
'''
#-------------------------------------------------------------------------------

# #-----------------------Plot experimental data-----------
SMALL_SIZE = 25
MEDIUM_SIZE = 30
BIGGER_SIZE = 40

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plot fish+fep
# semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\experiments_AO\\23_02\\fish_side+fep\\'
# best_weights=pickle.load(open(semi_path+'20210223-165858_aniso\\'+'_best_weights.p','rb'))
# data_flag='fep+fish'
semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\experiments_AO\\23_02_21\\fish_side\\20210223-180909_aniso\\'
# path_images=semi_path+'comparison_raw_fep_corr_total_corr\\'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_7\\extra\\'
best_weights=pickle.load(open(semi_path+'_best_weights.p','rb'))

res = 1152
pixSize = 1.8399999999999997e-05
lam = 8.000000000000001e-07
base = "Zernike"
order = 8
xc,yc=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))

r_unit = np.sqrt(xc**2+yc**2)
out_pupil = r_unit > 1
xc=xc*res*pixSize/2.0
yc=yc*res*pixSize/2.0

r=np.sqrt(xc**2+yc**2)

zern_dataset = generateAberrationDataset(res,base,order)


stop=45


fig_aniso, axs = plt.subplots(1,3, figsize=(30, 30))
fig_aniso.subplots_adjust(hspace = .1, wspace=.5)
# fig.subplots_adjust(hspace = .5, wspace=.002)

axs = axs.ravel()

#
#
i=0
for j in range(best_weights.shape[0]):
    coeffs = best_weights[j,:]
    pointabberation = ((getAberrationPhase(zern_dataset,lam*10**9,coeffs)*(lam*10**9/1000.0)*2*np.pi/(2**16)))
    pointabberation = 2*np.pi*(pointabberation-np.min(pointabberation))/(np.max(pointabberation)-np.min(pointabberation))-np.pi
    pointabberation[out_pupil] = np.nan
    im = axs[i].imshow(pointabberation,cmap = 'bwr',extent=[-1,1,-1,1])
    axs[i].axis('off')
    i += 1

print('max pointabberation', np.amax(pointabberation), ' min pointabberation', np.amin(pointabberation))

#
fig_aniso.subplots_adjust(right=0.5)
# cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.71])
cbar = plt.colorbar(im, ticks=[-np.pi, np.pi], ax=axs,orientation='vertical',fraction=0.012, pad=0.01 )
cbar.set_ticklabels([r'$- \pi$', r'$\pi$'])
# cbar.ax.tick_params(labelsize=50)
# cbar.fig_aniso.set_title(r'$20 \quad Zernikes$')

plt.show()
fig_aniso.savefig(save_path+'aniso_phase'+str(stop)+'.png',dpi=300, bbox_inches='tight' )

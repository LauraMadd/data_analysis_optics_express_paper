import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from scipy.optimize import curve_fit, leastsq
from scipy.stats import skewnorm
from scipy.special import factorial as fact
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tabulate import tabulate
#-----------------plots_style-------------------------------------------------

plt.style.use('seaborn-white')
#plt.xkcd()
plt.rcParams['image.cmap'] = 'gray'
font = {'family' : 'calibri',
        'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)

SMALL_SIZE = 25
MEDIUM_SIZE = 35
BIGGER_SIZE = 40

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#----------------------------------Scale bar -----------------------------------------
#---------------------------Functions phase maps-------------------------------
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

#-------------------------Paths ---------------------------------


# semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\experiments_AO\\02_03\\fep_fov100\\weighted_nearest\\analysis_w\\'

semi_path_h='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\method_validation\\19_02_21\\hline8\\'


save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_S2\\'

# #-------------------------------Grid used to correct
###--------------Corr--------------------------------------------
total_grid_corr= np.array(Image.open(semi_path_h+'total_corrected_syst.tif'))
#max and min corrected corr
min_total_grid_corr=np.amin(total_grid_corr)
max_total_grid_corr=np.amax(total_grid_corr)
print('max/min grid corr  total',min_total_grid_corr, max_total_grid_corr)
# --Normalization itself
total_grid_corr_norm=(total_grid_corr-min_total_grid_corr)/(max_total_grid_corr-min_total_grid_corr)
total_grid_corr_norm_plot=total_grid_corr_norm[528:1527,1073:1314]
min_total_grid_corr_norm=np.amin(total_grid_corr_norm_plot)
max_total_grid_corr_norm=np.amax(total_grid_corr_norm_plot)
print(' min/max grid corr  total norm', min_total_grid_corr_norm, max_total_grid_corr_norm)

###--------------Raw---------------------------------------------
total_grid_raw= np.array(Image.open(semi_path_h+'total_original_syst.tif'))
#max and min corrected raw
# min_total_grid_raw=np.amin(total_grid_raw)
# max_total_grid_raw=np.amax(total_grid_raw)
# print('max/min grid raw  total',min_total_grid_raw, max_total_grid_raw)
# # --Normalization itself
# total_grid_raw_norm=(total_grid_raw-min_total_grid_raw)/(max_total_grid_raw-min_total_grid_raw)
# total_grid_raw_norm_plot=total_grid_raw_norm[670:1390,670:1390]
# min_total_grid_raw_norm=np.amin(total_grid_raw_norm_plot)
# max_total_grid_raw_norm=np.amax(total_grid_raw_norm_plot)
# print(' min/max grid raw  total norm', min_total_grid_raw_norm, max_total_grid_raw_norm)

# # --Normalization corrected
total_grid_raw_norm=(total_grid_raw-min_total_grid_corr)/(max_total_grid_corr-min_total_grid_corr)
total_grid_raw_norm_plot=total_grid_raw_norm[528:1527,1073:1314]
min_total_grid_raw_norm=np.amin(total_grid_raw_norm_plot)
max_total_grid_raw_norm=np.amax(total_grid_raw_norm_plot)
print(' min/max grid raw  total norm', min_total_grid_raw_norm, max_total_grid_raw_norm)

#figure aberrated image
fig_1=plt.figure('Aberrated image rand', figsize=(30, 15))
plot_raw=plt.imshow(total_grid_raw_norm_plot, vmin=0, vmax=1,cmap = 'jet' )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_raw, ticks=[0, 1], orientation="horizontal", fraction=0.012, pad=0.01)
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_1.savefig(save_path+'h_im_raw_norm_corr_scale.png',dpi=300, bbox_inches='tight')
#
# # figure corrected image
fig_2=plt.figure('corrected image rand', figsize=(30, 15))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_corr=plt.imshow(total_grid_corr_norm_plot, vmin=0, vmax=1,cmap = 'jet' )
# scalebar = ScaleBar(0.195, 'um', font_properties={'family':'calibri', 'size': 40})
# scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_corr, ticks=[0, 1], orientation="horizontal", fraction=0.012, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_2.savefig(save_path+'h_im_corr_scale.png',dpi=300, bbox_inches='tight')
#
# # -----------------------------------------Phase  maps-----------------------------
best_weights_h=pickle.load(open(semi_path_h+'_best_weights.p','rb'))
#
#
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



# # #------------------Phase maps grid
fig_aniso_grid, axs =plt.subplots(8,1,figsize=(30, 15))
fig_aniso_grid.subplots_adjust(hspace = 0.002, wspace=0.002)
# fig.subplots_adjust(hspace = .5, wspace=.002)

axs = axs.ravel()

i=0
for j in range(best_weights_h.shape[0]):
    coeffs = best_weights_h[j,:]
    pointabberation = ((getAberrationPhase(zern_dataset,lam*10**9,coeffs)*(lam*10**9/1000.0)*2*np.pi/(2**16)))
    pointabberation = 2*np.pi*(pointabberation-np.min(pointabberation))/(np.max(pointabberation)-np.min(pointabberation))-np.pi
    pointabberation[out_pupil] = np.nan
    im = axs[i].imshow(pointabberation,cmap = 'bwr',extent=[-1,1,-1,1])
    axs[i].axis('off')
    #axs[i].set_title(str(j))
    i += 1

print('max pointabberation', np.amax(pointabberation), ' min pointabberation', np.amin(pointabberation))

#
fig_aniso_grid.subplots_adjust(right=0.5)
# cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.71])
cbar = plt.colorbar(im, ticks=[-np.pi, np.pi], ax=axs,orientation='horizontal',fraction=0.012, pad=0.01 )
cbar.set_ticklabels([r'$- \pi$', r'$\pi$'])
# cbar.ax.tick_params(labelsize=50)

plt.show()
fig_aniso_grid.savefig(save_path+'aniso_phase_h.png',dpi=300, bbox_inches='tight' )

# #--------------------------------------------------------------------------
# #------------------bar plots

save_path_barplots=save_path+'\\bar_plots\\'

fig_barplt, axs = plt.subplots(4,2,figsize=(20, 20))
fig_barplt.subplots_adjust(hspace = .26, wspace=.2)
axs = axs.ravel()
bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12','13','14')
y_ticks=np.linspace(-1.5,1.5,3)
x = (np.arange(3, 15, 1))
y_pos = x
width = 1
for i in range(best_weights_h.shape[0]) :
    print(i)
    axs[i].bar(x, best_weights_h[i,3:], width,align='edge')
    axs[i].set_yticks(y_ticks)
    axs[i].set_xticks(y_pos+0.5) # values
    axs[i].set_xticklabels(bars) # labels
    axs[i].set_ylim(-1.6,1.6)
    axs[i].axhline(0, color='black',ls='--')
    axs[i].set_title(r'$ Roi \quad$'+str(i+1),fontsize= 30)
    axs[i].grid(True)
for ax in axs.flat:
    ax.label_outer()
fig_barplt.text(0.5, 0.01, r'$ Zernike \quad index$', ha='center', fontsize= BIGGER_SIZE)
fig_barplt.text(0.04, 0.5, r'$Weights \quad (\mu m) $', va='center', rotation='vertical', fontsize=BIGGER_SIZE)
plt.show()
fig_barplt.savefig(save_path+'bar_plot_h.png',dpi=300, bbox_inches='tight')

#==============================================================================

semi_path_v='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\19_02_21\\vline8\\'


save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_S2\\'

# #-------------------------------Grid used to correct
###--------------Corr--------------------------------------------
total_grid_corr= np.array(Image.open(semi_path_v+'total_corrected_syst.tif'))
#max and min corrected corr
min_total_grid_corr=np.amin(total_grid_corr)
max_total_grid_corr=np.amax(total_grid_corr)
print('max/min grid corr  total',min_total_grid_corr, max_total_grid_corr)
# --Normalization itself
total_grid_corr_norm=(total_grid_corr-min_total_grid_corr)/(max_total_grid_corr-min_total_grid_corr)
total_grid_corr_norm_plot=total_grid_corr_norm[779:1020,533:1532]
min_total_grid_corr_norm=np.amin(total_grid_corr_norm_plot)
max_total_grid_corr_norm=np.amax(total_grid_corr_norm_plot)
print(' min/max grid corr  total norm', min_total_grid_corr_norm, max_total_grid_corr_norm)

###--------------Raw---------------------------------------------
total_grid_raw= np.array(Image.open(semi_path_v+'total_original_syst.tif'))
#max and min corrected raw
# min_total_grid_raw=np.amin(total_grid_raw)
# max_total_grid_raw=np.amax(total_grid_raw)
# print('max/min grid raw  total',min_total_grid_raw, max_total_grid_raw)
# # --Normalization itself
# total_grid_raw_norm=(total_grid_raw-min_total_grid_raw)/(max_total_grid_raw-min_total_grid_raw)
# total_grid_raw_norm_plot=total_grid_raw_norm[670:1390,670:1390]
# min_total_grid_raw_norm=np.amin(total_grid_raw_norm_plot)
# max_total_grid_raw_norm=np.amax(total_grid_raw_norm_plot)
# print(' min/max grid raw  total norm', min_total_grid_raw_norm, max_total_grid_raw_norm)

# # --Normalization corrected
total_grid_raw_norm=(total_grid_raw-min_total_grid_corr)/(max_total_grid_corr-min_total_grid_corr)
total_grid_raw_norm_plot=total_grid_raw_norm[779:1020,533:1532]
min_total_grid_raw_norm=np.amin(total_grid_raw_norm_plot)
max_total_grid_raw_norm=np.amax(total_grid_raw_norm_plot)
print(' min/max grid raw  total norm', min_total_grid_raw_norm, max_total_grid_raw_norm)

#figure aberrated image
fig_3=plt.figure('Aberrated image rand', figsize=(15, 30))
plot_raw=plt.imshow(total_grid_raw_norm_plot, vmin=0, vmax=1,cmap = 'jet' )
# scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right')
#
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_raw, ticks=[0, 1], orientation="vertical", fraction=0.012, pad=0.01)
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_3.savefig(save_path+'v_im_raw_norm_corr_scale.png',dpi=300, bbox_inches='tight')

# figure corrected image
fig_4=plt.figure('corrected image rand', figsize=(15, 30))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_corr=plt.imshow(total_grid_corr_norm_plot, vmin=0, vmax=1,cmap = 'jet' )
# scalebar = ScaleBar(0.195, 'um', font_properties={'family':'calibri', 'size': 40})
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_corr, ticks=[0, 1], orientation="vertical", fraction=0.012, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_4.savefig(save_path+'v_im_corr_scale.png',dpi=300, bbox_inches='tight')

# -----------------------------------------Phase  maps-----------------------------
semi_path_v='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\19_02_21\\vline8\\'
best_weights_v=pickle.load(open(semi_path_v+'_best_weights.p','rb'))

#
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



# # #------------------Phase maps grid
fig_aniso_grid, axs =plt.subplots(1,8,figsize=(15, 30))
fig_aniso_grid.subplots_adjust(hspace = 0.002, wspace=0.002)
# fig.subplots_adjust(hspace = .5, wspace=.002)

axs = axs.ravel()

i=0
for j in range(best_weights_v.shape[0]):
    coeffs = best_weights_v[j,:]
    pointabberation = ((getAberrationPhase(zern_dataset,lam*10**9,coeffs)*(lam*10**9/1000.0)*2*np.pi/(2**16)))
    pointabberation = 2*np.pi*(pointabberation-np.min(pointabberation))/(np.max(pointabberation)-np.min(pointabberation))-np.pi
    pointabberation[out_pupil] = np.nan
    im = axs[i].imshow(pointabberation,cmap = 'bwr',extent=[-1,1,-1,1])
    axs[i].axis('off')
    #axs[i].set_title(str(j))
    i += 1

print('max pointabberation', np.amax(pointabberation), ' min pointabberation', np.amin(pointabberation))

#
fig_aniso_grid.subplots_adjust(right=0.5)
# cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.71])
cbar = plt.colorbar(im, ticks=[-np.pi, np.pi], ax=axs,orientation='vertical',fraction=0.012, pad=0.01 )
cbar.set_ticklabels([r'$- \pi$', r'$\pi$'])
# cbar.ax.tick_params(labelsize=50)

plt.show()
fig_aniso_grid.savefig(save_path+'aniso_phase_v.png',dpi=300, bbox_inches='tight' )

# #--------------------------------------------------------------------------
# #------------------bar plots

# save_path_barplots=save_path+'\\bar_plots\\'

fig_barplt, axs = plt.subplots(4,2,figsize=(20, 20))
fig_barplt.subplots_adjust(hspace = .26, wspace=.2)
axs = axs.ravel()
bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12','13','14')
y_ticks=np.linspace(-1.5,1.5,3)
x = (np.arange(3, 15, 1))
y_pos = x
width = 1
for i in range(best_weights_v.shape[0]) :
    print(i)
    axs[i].bar(x, best_weights_v[i,3:], width,align='edge')
    axs[i].set_yticks(y_ticks)
    axs[i].axhline(0, color='black',ls='--')
    axs[i].set_xticks(y_pos+0.5) # values
    axs[i].set_xticklabels(bars) # labels
    axs[i].set_ylim(-1.6,1.6)
    axs[i].set_title(r'$ Roi \quad$'+str(i+1),fontsize= 30)
    axs[i].grid(True)
for ax in axs.flat:
    ax.label_outer()
fig_barplt.text(0.5, 0.01, r'$ Zernike \quad index$', ha='center', fontsize= BIGGER_SIZE)
fig_barplt.text(0.04, 0.5, r'$Weights \quad (\mu m) $', va='center', rotation='vertical', fontsize=BIGGER_SIZE)
plt.show()
fig_barplt.savefig(save_path+'bar_plot_v.png',dpi=300, bbox_inches='tight')

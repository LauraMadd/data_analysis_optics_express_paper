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

#-----------------plots_style-------------------------------------------------
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

#---------------------------Functions phase maps-------------------------------
#------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------
# Script to plot raw and corrected for baseline and aberrated datasets +bar plots.
# Change paths and ylimits and ythicks bar plots
#-------------------------Paths ---------------------------------
semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\test_gui_AO\\30_03\\fep_repetition\\'

# semi_path='W:\\staff-groups\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\30_03\\fep_repetition\\'
# save_path=semi_path+'\\data_analysis\\'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_6\\'

#---------------change following------------------------------------------------
semi_path_baseline=semi_path+'fep_spatial_base\\'
semi_path_abb=semi_path+'fep_spatial_astm2\\'
# flag_aberrated= '180_astm2'
# flag_base='base_180'
flag_aberrated= 'astm2'
flag_base='base'
#-------------------------------------------------------------------------------
##-----------------------------------Total images-------------------------------
#---------------------------- Aberrated images--------------------------------
total_im_corr= np.array(Image.open(semi_path_abb+'total_corrected_syst.tif'))
# max and min corrected raw
min_total_corr=np.amin(total_im_corr)
max_total_corr=np.amax(total_im_corr)
print('max/min corr total',min_total_corr, max_total_corr)

# --Normalization
total_im_corr_norm=(total_im_corr)/(max_total_corr)
# total_im_corr_norm_plot=(total_im_corr_norm)
total_im_corr_norm_plot=total_im_corr_norm[800:1300, 800:1300]
#max and min corrected normalized
min_total_im_corr_norm=np.amin(total_im_corr_norm_plot)
max_total_im_corr_norm=np.amax(total_im_corr_norm_plot)
print(' min/max corr total norm', min_total_im_corr_norm, max_total_im_corr_norm)

####Raw
total_im_raw= np.array(Image.open(semi_path_abb+'total_original_syst.tif'))
#max and min corrected raw
min_total_raw=np.amin(total_im_raw)
max_total_raw=np.amax(total_im_raw)
print('max/min raw total',min_total_raw, max_total_raw)

# --Normalization
total_im_raw_norm=(total_im_raw)/(max_total_corr)
# total_im_raw_norm_plot=(total_im_raw_norm)
total_im_raw_norm_plot=total_im_raw_norm[800:1300, 800:1300]
#max and min corrected normalized
# # im_corr_norm_plot=im_corr_norm[70:120,70:120]
min_total_im_raw_norm=np.amin(total_im_raw_norm_plot)
max_total_im_raw_norm=np.amax(total_im_raw_norm_plot)
print(' min/max raw total norm', min_total_im_raw_norm, max_total_im_raw_norm)
# #-------------------------------------------------------------------------------
# # ------------------------ figure aberrated image-------------------------------
fig_1=plt.figure('Aberrated image', figsize=(20, 20))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_raw=plt.imshow(total_im_raw_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')

plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_raw, ticks=[0, 1], orientation="vertical", fraction=0.047, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_1.savefig(save_path+flag_aberrated+'_im_raw.png',dpi=300, bbox_inches='tight')
#
# figure corr fep_correction
fig_2=plt.figure('Corrected image', figsize=(20, 20))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_fep=plt.imshow(total_im_corr_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')

plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_fep, ticks=[0, 1], orientation="vertical", fraction=0.047, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_2.savefig(save_path+flag_aberrated+'_im_corr.png',dpi=300, bbox_inches='tight')
# # #-------------------------------------------------------------------------------
# # #-----------------------------------Images baseline------------------------------
# # ##Corrected
# total_im_corr_base= np.array(Image.open(semi_path_baseline+'total_corrected_syst.tif'))
# #max and min corrected raw
# min_total_corr_base=np.amin(total_im_corr_base)
# max_total_corr_base=np.amax(total_im_corr_base)
# print('max/min corr total',min_total_corr_base, max_total_corr_base)
#
# # --Normalization
# total_im_corr_norm_base=(total_im_corr_base)/(max_total_corr_base)
# # total_im_corr_norm_plot=(total_im_corr_norm)
# total_im_corr_norm_plot_base=total_im_corr_norm_base[800:1300, 800:1300]
# #max and min corrected normalized
# min_total_im_corr_norm_base=np.amin(total_im_corr_norm_plot_base)
# max_total_im_corr_norm_base=np.amax(total_im_corr_norm_plot_base)
# print(' min/max corr total norm_base', min_total_im_corr_norm_base, max_total_im_corr_norm_base)
# # #
# # # ####Raw
# total_im_raw_base= np.array(Image.open(semi_path_baseline+'total_original_syst.tif'))
# #max and min corrected raw
# min_total_raw_base=np.amin(total_im_raw_base)
# max_total_raw_base=np.amax(total_im_raw_base)
# print('max/min raw total',min_total_raw_base, max_total_raw_base)
#
# # --Normalization
# total_im_raw_norm_base=(total_im_raw_base)/(max_total_corr_base)
# # total_im_raw_norm_plot_base=(total_im_raw_norm)
# total_im_raw_norm_plot_base=total_im_raw_norm_base[800:1300, 800:1300]
# #max and min corrected normalized
# # # im_corr_norm_plot=im_corr_norm[70:120,70:120]
# min_total_im_raw_norm_base=np.amin(total_im_raw_norm_plot_base)
# max_total_im_raw_norm_base=np.amax(total_im_raw_norm_plot_base)
# print(' min/max raw total norm_base', min_total_im_raw_norm_base, max_total_im_raw_norm_base)
#
#
# #-------------------------------------------------------------------------------
# #-------------------------figure baseline image-----------------------------
# fig_3=plt.figure('Aberrated_base image', figsize=(20, 20))
# # plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
# plot_raw=plt.imshow(total_im_raw_norm_plot_base, vmin=0, vmax=1,cmap = 'jet', )
# scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
# plt.gca().add_artist(scalebar)
# cbar_ab = plt.colorbar(plot_raw, ticks=[0, 1], orientation="vertical", fraction=0.047, pad=0.01)
# #cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
# cbar_ab.set_ticklabels([r'$0$', r'$1$'])
# plt.axis('off')
# plt.show()
# fig_3.savefig(save_path+flag_base+'_im_raw_base.png',dpi=300, bbox_inches='tight')
#
# fig_4=plt.figure('Corr_base image', figsize=(20, 20))
# # plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
# plot_base=plt.imshow(total_im_corr_norm_plot_base, vmin=0, vmax=1,cmap = 'jet', )
# scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
# plt.gca().add_artist(scalebar)
# cbar_ab = plt.colorbar(plot_base, ticks=[0, 1], orientation="vertical", fraction=0.047, pad=0.01)
# #cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
# cbar_ab.set_ticklabels([r'$0$', r'$1$'])
# plt.axis('off')
# plt.show()
# fig_4.savefig(save_path+flag_base+'_im_corr.png',dpi=300, bbox_inches='tight')
# #
# # # #--------------------------------------------------------------------------------
# # # # #-----------------------------------------Phase  maps-----------------------------
# # #
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
# # ##-------------------------------------------------------------------------
# # ##------------------Phase maps baseline
best_weights_base=pickle.load(open(semi_path_baseline+'_best_weights.p','rb'))
error_weights_base=np.load(semi_path_baseline+'error_weights.npy')
# fig_aniso_baseline, axs =plt.subplots(3,3,figsize=(20, 20))
# fig_aniso_baseline.subplots_adjust(hspace = .5, wspace=.002)
# axs = axs.ravel()
# i=0
point_order=[2,1,0,5,4,3,8,7,6] #grid  3x3
# for j in (point_order):
#     coeffs = best_weights_base[j,:]
#     pointabberation = ((getAberrationPhase(zern_dataset,lam*10**9,coeffs)*(lam*10**9/1000.0)*2*np.pi/(2**16)))
#     pointabberation = 2*np.pi*(pointabberation-np.min(pointabberation))/(np.max(pointabberation)-np.min(pointabberation))-np.pi
#     pointabberation[out_pupil] = np.nan
#     im = axs[i].imshow(pointabberation,cmap = 'bwr',extent=[-1,1,-1,1])
#     axs[i].axis('off')
#     i += 1
# print('max pointabberation', np.amax(pointabberation), ' min pointabberation', np.amin(pointabberation))
# # #-------------------------------------Figure------------------------------------
# fig_aniso_baseline.subplots_adjust(right=0.5)
# cbar = plt.colorbar(im, ticks=[-np.pi, np.pi], ax=axs,orientation='vertical',fraction=0.047, pad=0.01 )
# cbar.set_ticklabels([r'$- \pi$', r'$\pi$'])
# plt.show()
# fig_aniso_baseline.savefig(save_path+flag_base+'_aniso_phase_baseline.png',dpi=300, bbox_inches='tight' )
# # # ##------------------Phase maps aberrated-----------------------------------------
best_weights=pickle.load(open(semi_path_abb+'_best_weights.p','rb'))
error_weights=np.load(semi_path_abb+'error_weights.npy')
fig_aniso, axs =plt.subplots(3,3,figsize=(20, 20))
fig_aniso.subplots_adjust(hspace = .002, wspace=.002)

axs = axs.ravel()
i=0
for j in (point_order):
    print('point order',j)
    coeffs = best_weights[j,:]
    pointabberation = ((getAberrationPhase(zern_dataset,lam*10**9,coeffs)*(lam*10**9/1000.0)*2*np.pi/(2**16)))
    pointabberation = 2*np.pi*(pointabberation-np.min(pointabberation))/(np.max(pointabberation)-np.min(pointabberation))-np.pi
    pointabberation[out_pupil] = np.nan
    im = axs[i].imshow(pointabberation,cmap = 'bwr',extent=[-1,1,-1,1])
    axs[i].axis('off')
    i += 1
print('max pointabberation', np.amax(pointabberation), ' min pointabberation', np.amin(pointabberation))
fig_aniso.subplots_adjust(right=0.5)

cbar = plt.colorbar(im, ticks=[-np.pi, np.pi], ax=axs,orientation='vertical',fraction=0.047, pad=0.01 )
cbar.set_ticklabels([r'$- \pi$', r'$\pi$'])
 # cbar.ax.tick_params(labelsize=50)
plt.show()
fig_aniso.savefig(save_path+flag_aberrated+'_aniso_phase.png',dpi=300, bbox_inches='tight' )
# #-------------------------------------------------------------------------------
#
# #------------------------------Bar plots----------------------------------------
#
# # #--------------------------Bar plot baseline with aberrated---------------------
# # #
# # fig_bar_bis, axs =plt.subplots(3,3,figsize=(30, 30),linewidth=2.0)
# # fig_bar_bis.subplots_adjust(hspace = .2, wspace=.1)
# # # color_i=['blue', 'red']
# # # offset=[0,0.2]
# # axs = axs.ravel()
# #
# # order_plots=[2,1,0,5,4,3,8,7,6] #grid  3x3
# # bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12','13','14','15','16',
# #     '17','18','19','20')
# # j=0
# # for i in (order_plots) :
# #     print(i)
# #
# #
# #     y_pos = np.arange(len(bars))
# #     N = len(bars)
# #     x_1 = (np.arange(0, N, 1))
# #     x_2= (np.arange(0, N, 1)-0.4)
# #     width = 0.4
# #     # x_2= (np.arange(0, N, 1)-0.4)
# #     axs[i].bar(x_1, best_weights[order_plots[i], 3:], width, yerr=error_weights[order_plots[i],:], color= 'blue',align='edge', label='aberrated ' )
# #     axs[i].bar(x_2, best_weights_base[order_plots[i], 3:], width,yerr=error_weights_base[order_plots[i],:],color='red',align='edge',label=' baseline')
# #
# #     # if(i==0):
# #     #     x_2= (np.arange(0, N, 1)-0.4)
# #     #     axs[i].bar(x_2, best_weights_base[order_plots[i], 3:], width,color='red',align='edge',label=' baseline')
# #     # if(i==2):
# #     #     x_2= (np.arange(0, N, 1)-0.4)
# #     #     axs[i].bar(x_2, best_weights_base[order_plots[i], 3:], width,color='red',align='edge',label=' baseline')
# #     #
# #     # if(i==4):
# #     #     x_2=(np.arange(0, N, 1)-0.4)
# #     #     axs[i].bar(x_2, best_weights_base[order_plots[i], 3:], width,color='red',align='edge',label='baseline')
# #
# #     axs[i].grid(True)
# #
# #     # axs[i].set_yticks(y_pos)
# #     axs[i].set_xticks(y_pos)
# #     axs[i].set_xticklabels(bars, fontsize=MEDIUM_SIZE)
# #     # axs[i].set_yticklabels([-1,0,2,3])
# #     ## astm2 dataset
# #     # axs[i].set_ylim([-1.5,2])
# #     # axs[i].set_yticks([-1.5,-1,0,1,2])
# #     # axs[i].set_yticklabels([-1.5,-1,0,1,2],fontsize=MEDIUM_SIZE)
# #
# #     ## ast2 dataset
# #     # axs[i].set_ylim([-2.5,1])
# #     # axs[i].set_yticks([-2.5,-2,-1,0,1])
# #     # axs[i].set_yticklabels([-2.5,-2,-1,0,1],fontsize=MEDIUM_SIZE)
# #
# #
# #     # astm2_180  dataset
# #     axs[i].set_ylim([-1,2])
# #     axs[i].set_yticks([-1,0,1,2])
# #     axs[i].set_yticklabels([-1,0,1,2],fontsize=MEDIUM_SIZE)
# #
# #     # plt.xticks(y_pos, bars, fontsize=MEDIUM_SIZE)
# #     # plt.ylabel(r'$Weights \quad (μm)$')
# #     # plt.xlabel(r'$Aberration\quad index$')
# #     #
# #     # plt.legend(loc='upper right',  fontsize=MEDIUM_SIZE)
# #
# # fig_bar_bis.text(0.5, 0.02, r'$Aberration\quad index$', ha='center', fontsize=MEDIUM_SIZE)
# # fig_bar_bis.text(0.08, 0.5, r'$Weights \quad (μm)$', va='center', rotation='vertical',fontsize=MEDIUM_SIZE)
# #
# # # for ax in axs.flat:
# # #     ax.set(xlabel=r'$Aberration\quad index$', ylabel=r'$Weights \quad (μm)$')
# #
# # # Hide x labels and tick labels for top plots and y ticks for right plots.
# # # for ax in axs.flat:
# # #     ax.label_outer()
# #
# # plt.legend(loc='lower center', bbox_to_anchor=(-0.7, -1),fancybox=False, shadow=False, ncol=2)
# # # plt.legend(loc='lower center',fancybox=False, shadow=False, ncol?=2)
# # plt.show()
# # fig_bar_bis.savefig(save_path+flag_aberrated+'_bar_plot_pairs.png',dpi=300, bbox_inches='tight')
#
#
# #
# #---------------------------Bar plots differences
#
fig_bar_diff, axs =plt.subplots(3,3,figsize=(30, 30),linewidth=2.0)
fig_bar_diff.subplots_adjust(hspace = .2, wspace=.1)

axs = axs.ravel()

order_plots=[2,1,0,5,4,3,8,7,6] #grid  3x3
bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12','13','14','15','16',
    '17','18','19','20')
j=0
diff=np.zeros((len(order_plots), len(bars)),dtype=np.float)
for i in (order_plots) :
    print(i)


    y_pos = np.arange(len(bars))
    N = len(bars)
    x_1 = (np.arange(0, N, 1))
    width = 0.4
    # x_2= (np.arange(0, N, 1)-0.4)
    diff[i,:]= (best_weights[order_plots[i], 3:]-best_weights_base[order_plots[i], 3:])
    axs[i].bar(x_1,diff[i,:] , width,color= 'blue',align='edge', label='ab-base ' )
    # axs[i].bar(x_2, best_weights_base[order_plots[i], 3:], width,color='red',align='edge',label=' baseline')

    # if(i==0):
    #     x_2= (np.arange(0, N, 1)-0.4)
    #     axs[i].bar(x_2, best_weights_base[order_plots[i], 3:], width,color='red',align='edge',label=' baseline')
    # if(i==2):
    #     x_2= (np.arange(0, N, 1)-0.4)
    #     axs[i].bar(x_2, best_weights_base[order_plots[i], 3:], width,color='red',align='edge',label=' baseline')
    #
    # if(i==4):
    #     x_2=(np.arange(0, N, 1)-0.4)
    #     axs[i].bar(x_2, best_weights_base[order_plots[i], 3:], width,color='red',align='edge',label='baseline')

    axs[i].grid(True)
    #
    # axs[i].set_ylim([-0.5,0.5])

    axs[i].set_xticks(y_pos)
    axs[i].set_xticklabels(bars,fontsize=MEDIUM_SIZE)

    # axs[i].set_yticks([-1,0,1,2,3])
    #astm2 dataset
    # axs[i].set_ylim([-1,2])
    # axs[i].set_yticks([-1,0,1,2])
    # axs[i].set_yticklabels([-1,0,1,2],fontsize=MEDIUM_SIZE)


    ## ast2 dataset
    # axs[i].set_ylim([-2.5,1])
    # axs[i].set_yticks([-2.5,-2,-1,0,1])
    # axs[i].set_yticklabels([-2.5,-2,-1,0,1],fontsize=MEDIUM_SIZE)

    # #astm2_180 dataset
    axs[i].set_ylim([-1,2])
    axs[i].set_yticks([-1,0,1,2])
    axs[i].set_yticklabels([-1,0,1,2],fontsize=MEDIUM_SIZE)


fig_bar_diff.text(0.5, 0.02, r'$Zernike \quad index$', ha='center', fontsize=MEDIUM_SIZE)
fig_bar_diff.text(0.08, 0.5, r'$Difference \quad Weights \quad (μm)$', va='center', rotation='vertical',fontsize=MEDIUM_SIZE)
plt.legend(loc='upper center', bbox_to_anchor=(-0.7, -1),fancybox=False, shadow=False, ncol=2)
plt.show()
fig_bar_diff.savefig(save_path+flag_aberrated+'_bar_plot_diff.png',dpi=300, bbox_inches='tight')


#
#
#
# #---------------------------Bar plots differences lR
# ##----------------------Bar plots  comparison left right per row --------
#
fig_bar_diff_lr_base, axs =plt.subplots(3,figsize=(20, 15),linewidth=2.0)
fig_bar_diff_lr_base.subplots_adjust(hspace = .2, wspace=.5)
# color_i=['blue', 'red']
# offset=[0,0.2]
order_plots=[[1,0],[4,3],[7,6]] ## base
# order_plots=[[1,2],[4,5],[7,8]] ## base 180
bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12','13','14','15','16',
    '17','18','19','20')
j=0
for i in range (len(order_plots)) :
    print(i)

    # best_weights_i =best_weights[i,:]#  for OSA

    # y_aniso = np.copy(best_weights_i[3:])
    y_pos = np.arange(len(bars))
    N = len(bars)
    x_1 = (np.arange(0, N, 1))
    width = 0.4
    # x_2= (np.arange(0, N, 1)-0.4)
    diff_base=best_weights_base[order_plots[i][1], 3:]-best_weights_base[order_plots[i][0],3:]
    error_diff_base=error_weights_base[order_plots[i][1],:]+error_weights_base[order_plots[i][0],:]
    axs[i].bar(x_1, diff_base, width,yerr=error_diff_base,color= 'blue',align='edge',label='R-L baseline' )
    # axs[i].bar(x_2, best_weights[order_plots[i][1], 3:], width,color='red',align='edge',label='ROI right')
    axs[i].grid(True)
    axs[i].axhline(0, color='black',ls='--')
    # axs[i].set_ylim([-2,2])
    # axs[i].set_ylim([-1,1])
    axs[i].set_xticks(y_pos)
    axs[i].set_xticklabels(bars,fontsize=MEDIUM_SIZE)
    ##base
    axs[i].set_ylim([-2.2,2.2]) # astm2
    axs[i].set_yticks([-2,-1,0,1,2])
    axs[i].set_yticklabels([-2,-1,0,1,2],fontsize=MEDIUM_SIZE)



    ##base 180
    # axs[i].set_ylim([-1.5,1.5]) # astm2
    # axs[i].set_yticks([-1,0,1])
    # axs[i].set_yticklabels([-1,0,1],fontsize=MEDIUM_SIZE)
        # plt.ylabel(r'$Weights \quad (μm)$')
    # plt.ylabel(r'$Weights \quad (μm)$')
    # plt.xlabel(r'$Aberration\quad index$')

    axs[i].legend(loc='lower left',  fontsize=MEDIUM_SIZE)

fig_bar_diff_lr_base.text(0.5, 0.04, r'$Zernike \quad index$', ha='center', fontsize=MEDIUM_SIZE)
fig_bar_diff_lr_base.text(0.04, 0.5, r'$\Delta weights \quad (μm)$', va='center', rotation='vertical',fontsize=MEDIUM_SIZE)
for ax in axs.flat:
    ax.label_outer()

# plt.legend(loc='upper center', bbox_to_anchor=(1, -0.2),fancybox=False, shadow=False, ncol=2)
# plt.show()
# fig_bar_diff_lr_base.savefig(save_path+flag_base+'_bar_plot_diff_rl.png',dpi=300, bbox_inches='tight')

dst_pair=[12,9,6]
fig_bar_diff_lr, axs =plt.subplots(3,figsize=(12, 15),linewidth=2.0)
fig_bar_diff_lr.subplots_adjust(hspace = .5, wspace=.5)
order_plots=[[1,0],[4,3],[7,6]] ## base
# order_plots=[[1,2],[4,5],[7,8]] ## base 180
bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12','13','14','15','16',
    '17','18','19','20')
j=0
for i in range (len(order_plots)) :
    print(i)

    # best_weights_i =best_weights[i,:]#  for OSA

    # y_aniso = np.copy(best_weights_i[3:])
    y_pos = np.arange(len(bars))
    N = len(bars)
    x_1 = (np.arange(0, N, 1))
    width = 0.4
    # x_2= (np.arange(0, N, 1)-0.4)
    diff=best_weights[order_plots[i][1], 3:]-best_weights[order_plots[i][0], 3:]
    error_diff=error_weights[order_plots[i][1],:]+error_weights[order_plots[i][0],:]
    axs[i].bar(x_1, diff, width,yerr=error_diff,color= 'blue',align='edge',label='R-L' )
    axs[i].axhline(0, color='black',ls='--')
    # axs[i].bar(x_2, best_weights[order_plots[i][1], 3:], width,color='red',align='edge',label='ROI right')
    axs[i].grid(True)
    # axs[i].set_ylim([-2,2]) # astm2
    # axs[i].set_ylim([-0.5,2]) #  ast2
    # axs[i].set_ylim([-0.5,0.5])# astm2_180

    axs[i].set_xticks(y_pos)
    axs[i].set_xticklabels(bars,fontsize=MEDIUM_SIZE)
    ##astm2
    # axs[i].set_ylim([-2,1.5]) # astm2
    # axs[i].set_yticks([-2,-1.5,-1,0,1])
    # axs[i].set_yticklabels([-2,-1.5,-1,0,1],fontsize=MEDIUM_SIZE)

    ## astm2 r-L
    axs[i].set_ylim([-2.2,2.2]) # astm2
    axs[i].set_yticks([-2,-1,0,1,2])
    axs[i].set_yticklabels([-2,-1,0,1,2],fontsize=MEDIUM_SIZE)

    ##astm2_180
    # axs[i].set_ylim([-2,1.5]) # astm2
    # axs[i].set_yticks([-2,-1.5,-1,0,1])
    # axs[i].set_yticklabels([-2,-1.5,-1,0,1],fontsize=MEDIUM_SIZE)
    # axs[i].set_title('Pair '+str(dst_pair[i])+ r'$\quad \mu m$')
    ##ast2\
    # axs[i].set_ylim([-1,2]) # astm2
    # axs[i].set_yticks([-1,-0,1,2])
    # axs[i].set_yticklabels([-1,-0,1,2],fontsize=MEDIUM_SIZE)

    axs[i].legend(loc='lower left',  fontsize=MEDIUM_SIZE)

fig_bar_diff_lr.text(0.5, 0.02, r'$Zernike \quad index$', ha='center', fontsize=MEDIUM_SIZE)
fig_bar_diff_lr.text(0.04, 0.5, r'$\Delta weights \quad (μm)$', va='center', rotation='vertical',fontsize=MEDIUM_SIZE)

# for ax in axs.flat:
#     ax.set(xlabel=r'$Aberration\quad index$', ylabel=r'$Weights \quad (μm)$')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# plt.legend(loc='upper center', bbox_to_anchor=(1, -0.2),fancybox=False, shadow=False, ncol=2)
plt.show()
fig_bar_diff_lr.savefig(save_path+flag_aberrated+'_bar_plot_diff_rl.png',dpi=300, bbox_inches='tight')

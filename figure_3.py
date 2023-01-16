import sys

import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from scipy.optimize import curve_fit, leastsq
from scipy.stats import skewnorm
from scipy.special import factorial as fact
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from tabulate import tabulate


plt.style.use('seaborn-white')
#plt.xkcd()
plt.rcParams['image.cmap'] = 'gray'
font = {'family' : 'calibri',
        'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)
SMALL_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 50

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#-------------------------------------------------------------------------------
'''Variables to change :
  type_aniso_aniso, date, bar points, point, osa order, bars, zernikes, weights
'''
#-------------------------------------------------------------------------------
#-------------------------Bar plot best weights---------------------------------
type_aniso = 'aniso_sph'
date_aniso = 'precise_22_12'

type_iso = 'iso_sph'
date_iso = 'precise_22_12'

type_base = "baseline"
date_base = "precise_22_12"

semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\method_validation\\'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_3\\'


iso_best_weights=pickle.load(open(semi_path+date_aniso+'\\'+type_iso+'\\best_weights.p','rb'))
aniso_best_weights=pickle.load(open(semi_path+date_aniso+'\\'+type_aniso+'\\_best_weights.p','rb'))
base_weights=pickle.load(open(semi_path+date_base+'\\'+type_base+'\\_best_weights.p','rb'))
bar_points=[12,5] #select which points to get bar plots for

canopy_order = [12,8,4,0,13,9,5,1,14,10,6,2,15,11,7,3] #for a grid subplot of bar plots
#
# #-------------------------------------------------------------------------------
# #------------------------------Bar plot, figure 3 panel b  ----------------------------------------
#
# ----- changes zernike order from osa to noll indexing
noll_order= np.array([1,2,3,4,5,6,7,8,9,10,11])-1
osa_order = np.array([0,2,1,4,3,5,7,8,6,9,12])

# -----
sem_weights=np.zeros((9), dtype=np.float)
meas_point7=np.zeros((9,3), dtype=np.float)
av_sem=np.zeros((1), dtype=np.float)
best_weights_iso = iso_best_weights[:]
for i in bar_points:
    print(i)

    best_weights = aniso_best_weights[i,:]#  for OSA
    best_weights_base = base_weights[i,:]
    # best_weights_iso = iso_best_weights[i,:]
    fig_bar=plt.figure('barplot_best_weights_',figsize = (16,3),linewidth=2.0)


    #bar plot best_weights
    bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12')# osa order
    # bars = ('5','6', '7', '8', '9', '10','11')# noll order
    #bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12','13','14','15','16','17','18','19','20')

    # color=['green', 'orange', 'purple', 'red', 'lightskyblue', 'black', 'yellow','grey','magenta']

    y_aniso = np.copy(best_weights[3:])
    y_iso = np.copy(best_weights_iso[3:])
    y_base = np.copy(best_weights_base[3:])

    # y_base[:8]=0 #to plot baseline only on spherical
    y_pos = np.arange(len(bars))
    N = len(y_base)
    x = (np.arange(0, N, 1)-0.1)
    width = 0.3

    # ax1=fig_bar.add_subplot(1,1,1)
    plt.bar(x, y_base, width,color= 'darkblue',align='edge',label='baseline' )
    plt.xticks(y_pos+0.5, bars)
    plt.bar(x+width, y_iso, width,color= 'darkgray',align='edge', label='isoplanatic' )
    plt.bar(x+width+width, y_aniso, width,color= 'darkorange',align='edge', label='anisoplanatic' )
    # plt.bar(x, y, width,color= 'green',label='dynamic ROI ' )
    plt.ylabel(r'$Weights \quad (μm)$')
    plt.xlabel(r'$Zernike \quad index$')
    # plt.xticks(y_pos+1, bars)
    plt.axhline(0, color='black',ls='--')
    plt.ylim(-3, 3)
    plt.title(r'$Recovered \quad Zernike \quad coefficients \quad point \quad $'+str(canopy_order.index(i)+1))
    # plt.legend(loc='upper left',  fontsize=SMALL_SIZE)
    if(i==5):
        plt.legend(loc='lower left',ncol=3)
        for k in range(y_aniso.shape[0]-1):
            print(k)
            meas_point7[k,0]=y_base[k]
            meas_point7[k,1]=y_iso[k]
            meas_point7[k,2]=y_aniso[k]
        sem_weights=np.std(meas_point7, axis=1)

        sem=sem_weights[sem_weights!=0]
        av_sem=np.average(sem)

    table_results=tabulate([['sem ',sem_weights ],['averave ',av_sem]],headers=['sem on point 7'])

    f = open(save_path+'sem_point7.txt', 'w')
    f.write(table_results)
    f.close()

    # plt.grid(True)

    plt.show()
    fig_bar.savefig(save_path+'bar_plot'+str(i)+'_bis.png',dpi=300, bbox_inches='tight')

# #-------------------------------------------------------------------------------
# #--------------------------------Plots figure 3  panel a method_validation----------------
alpha=3
#load total images
im_aberrated=np.array(Image.open(semi_path+date_aniso+'\\'+type_iso+'\\'+ 'total_original_syst.tif'))
im_iso=np.array(Image.open(semi_path+date_iso+'\\'+type_iso+'\\'+ 'total_corrected_syst.tif'))
im_aniso=np.array(Image.open(semi_path+date_aniso+'\\'+type_aniso+'\\'+ 'total_corrected_syst.tif'))
# # im aberrated
max_im_ab=np.amax(im_aberrated)
min_im_ab=np.amin(im_aberrated)
print('im_ab',max_im_ab,min_im_ab )
im_aberrated_norm= (im_aberrated-min_im_ab)/(max_im_ab-min_im_ab)
max_im_ab_norm=np.amax(im_aberrated_norm)
min_im_ab_norm=np.amin(im_aberrated_norm)
print('im_ab_norm',max_im_ab_norm,min_im_ab_norm )
#Panel a left
# fig_aberr=plt.figure('Aberrated image', figsize=(45, 30))
fig_aberr=plt.figure('Aberrated image', figsize=(30, 30))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
# plot_aberrated=plt.imshow(im_aberrated_norm[500:1658, 479:1670], vmin=np.amin(im_aberrated_norm[500:1658, 479:1670]), vmax=np.amax(im_aberrated_norm[500:1658, 479:1670]),cmap = 'jet' )
plot_aberrated=plt.imshow(im_aberrated_norm[500:1658, 479:1670]*alpha, vmin=0, vmax=1, cmap = 'jet', )
# plot_aberrated=plt.imshow(im_aberrated_norm[600:1558, 579:1570], vmin=0, vmax=1, cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_aberrated, ticks=[min_im_ab_norm, max_im_ab_norm], orientation="horizontal", fraction=0.048, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_aberr.savefig(save_path+'im_total_ab_alpha'+str(alpha)+'.png',dpi=300, bbox_inches='tight')

#Panel a middle
# im  iso
max_im_iso=np.amax(im_iso)
min_im_iso=np.amin(im_iso)
print('im_iso',max_im_iso,min_im_iso )
im_iso_norm= (im_iso-min_im_iso)/(max_im_iso-min_im_iso)
max_im_iso_norm=np.amax(im_iso_norm)
min_im_iso_norm=np.amin(im_iso_norm)
print('im_iso_norm',max_im_iso_norm,min_im_iso_norm )
max_im_iso_norm=np.amax(im_iso_norm[500:1658, 479:1670])
min_im_iso_norm=np.amin(im_iso_norm[500:1658, 479:1670])
print('im_iso_norm',max_im_iso_norm,min_im_iso_norm )

fig_iso=plt.figure('Iso image ', figsize=(30, 30))
plot_iso=plt.imshow(im_iso_norm[500:1658, 479:1670]*alpha,vmin=0, vmax=1,cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)

cbar_iso = plt.colorbar(plot_iso,ticks=[min_im_iso_norm, max_im_iso_norm], orientation="horizontal", fraction=0.048, pad=0.01)
cbar_iso.set_ticklabels([r'$0$', r'$1$'])


plt.axis('off')
plt.show()
fig_iso.savefig(save_path+'im_total_iso_alpha'+str(alpha)+'.png',dpi=300, bbox_inches='tight')

#Panel a right
# im aniso
max_im_aniso=np.amax(im_aniso)
min_im_aniso=np.amin(im_aniso)
print('im_aniso',max_im_aniso,min_im_aniso )
im_aniso_norm=(im_aniso-min_im_aniso)/(max_im_aniso-min_im_aniso)

fig_aniso=plt.figure('Aniso image', figsize=(30, 30))
plot_aniso=plt.imshow(im_aniso_norm[500:1658, 479:1670]*alpha,vmin=0, vmax=1, cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
plt.axis('off')
max_im_aniso_norm=np.amax(im_aniso_norm)
min_im_aniso_norm=np.amin(im_aniso_norm)
print('im_aniso_norm',max_im_aniso_norm,min_im_aniso_norm )
max_im_aniso_norm=np.amax(im_aniso_norm[500:1658, 479:1670])
min_im_aniso_norm=np.amin(im_aniso_norm[500:1658, 479:1670])
print('im_aniso_norm',max_im_aniso_norm,min_im_aniso_norm )
#fraction is 0.047 if we plot the whole fov
cbar_aniso = plt.colorbar(plot_aniso, ticks=[min_im_aniso_norm, max_im_aniso_norm], orientation="horizontal", fraction=0.048, pad=0.01)
cbar_aniso.set_ticklabels([r'$0$', r'$1$'])
plt.show()
fig_aniso.savefig(save_path+'im_total_aniso_alpha'+str(alpha)+'.png',dpi=300, bbox_inches='tight')
# -----------------------Plot single points-------------------------------------
gauss_fit_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\method_validation\\precise_22_12\\stack\\analysis_stack_0.5\\gauss_fit_cgh\\'
# gauss_fit_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Hidde\\precise_22_12\\iso_sph\\data_analysis\\gauss_fit_bis\\'
# save_path=save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_3\\iso\\'
folders=os.listdir(gauss_fit_path)
# variables to save all data
results_fit_corr=np.zeros((len(folders), 200), dtype=float)
results_fit_raw=np.zeros((len(folders), 200), dtype=float)
results_raw=np.zeros((len(folders), 16), dtype=float)
results_corr=np.zeros((len(folders), 16), dtype=float)
results_FWHM_fit_raw=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
results_FWHM_fit_corr=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
results_max_corr=np.zeros((len(folders), 1), dtype=float)
results_max_raw=np.zeros((len(folders), 1), dtype=float)
results_fwhm_corr=np.zeros((len(folders), 1), dtype=float)
results_fwhm_raw=np.zeros((len(folders), 1), dtype=float)
results_max_corr_fit=np.zeros((len(folders), 1), dtype=float)
results_max_raw_fit=np.zeros((len(folders), 1), dtype=float)
results_i_corr=np.zeros((len(folders), 1), dtype=float)
results_i_raw=np.zeros((len(folders), 1), dtype=float)
for i in range (len(folders)):
    data_path=gauss_fit_path+folders[i]+'\\'
    files = [['original.tif','corrected.tif']]
    print(folders[i])
    # if folders[i]=='0006':
    #     continue
    for name in files[0]:
        print(name)
        image = Image.open(data_path+name)
        if (name == 'original.tif'):
            im_raw=np.array(image)
            print(name)
        im_corr=np.array(image)
    #---------------Im corr
    min_corr=np.amin(im_corr)
    max_corr=np.amax(im_corr)
    print('max/min corr',min_corr, max_corr)
    # --Normalization corr
    # im_corr_norm=(im_corr-min_corr)/(max_corr-min_corr)

    im_corr_norm=(im_corr)/(max_corr)
    index_im_corr_norm=np.unravel_index(im_corr_norm.argmax(),im_corr_norm.shape)
    im_corr_norm_plot=im_corr_norm[index_im_corr_norm[0]-10:index_im_corr_norm[0]+10,index_im_corr_norm[1]-10:index_im_corr_norm[1]+10]
    # im_corr_norm_plot=im_corr_norm[70:120,70:120]
    min_corr_norm=np.amin(im_corr_norm_plot)
    max_corr_norm=np.amax(im_corr_norm_plot)
    print(' min/max corr_norm', min_corr_norm, max_corr_norm)
    index_im_corr_norm=np.unravel_index(im_corr_norm_plot.argmax(),im_corr_norm_plot.shape)
    print('index im_corr_norm', index_im_corr_norm)
    #------------------------Im original
    min_raw=np.amin(im_raw)
    max_raw=np.amax(im_raw)
    print('max/min raw',min_raw, max_raw)
    # --Normalization raw
    # im_raw_norm=(im_raw-min_corr)/(max_corr-min_corr)
    im_raw_norm=(im_raw)/(max_corr)
    index_im_raw_norm=np.unravel_index(im_raw_norm.argmax(),im_raw_norm.shape)
    im_raw_norm_plot=im_raw_norm[index_im_raw_norm[0]-10:index_im_raw_norm[0]+10,index_im_raw_norm[1]-10:index_im_raw_norm[1]+10]
    # im_raw_norm_plot=im_raw_norm[44:64,44:64]
    min_raw_norm=np.amin(im_raw_norm_plot)
    max_raw_norm=np.amax(im_raw_norm_plot)
    print(' min/max raw_norm', min_raw_norm, max_raw_norm)
    index_im_raw_norm=np.unravel_index(im_raw_norm_plot.argmax(),im_raw_norm_plot.shape)
    print('index im_raw_norm', index_im_raw_norm)
#
#
#     fig1, (ax1, ax2)= plt.subplots(1, 2,figsize = (5,5),linewidth=2.0)
#     plot_raw=ax1.imshow(im_raw_norm_plot, vmin=0, vmax=np.amax(im_raw_norm), cmap = 'jet')
#     cbar_ax1 = plt.colorbar(plot_raw,ax=ax1,ticks=[0, round(np.amax(im_raw_norm_plot),2)], orientation="horizontal", fraction=0.047, pad=0.01)
#     cbar_ax1.set_ticklabels(['0', str(round(np.amax(im_raw_norm_plot),2))])
#     cbar_ax1.ax.tick_params(labelsize=15)
#     ax1.hlines(y=index_im_raw_norm[0],xmin=index_im_raw_norm[1]-8,  xmax=index_im_raw_norm[1]+8, colors='black', linestyles='dashed', lw=1)
#     ax1.vlines(x=index_im_raw_norm[1], ymin=index_im_raw_norm[0]-8, ymax=index_im_raw_norm[0]+8, colors='red', linestyles='dashed', lw=1)
#
#     ax1.axis('off')
#     scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
#     ax1.add_artist(scalebar)
#   ## Plot figure 3 panel d top
    plot_aniso=ax2.imshow(im_corr_norm_plot, vmin=0, vmax=1, cmap = 'jet')
    cbar_ax2 = plt.colorbar(plot_aniso,ax=ax2,ticks=[0, max_corr_norm], orientation="horizontal", fraction=0.047, pad=0.01)
    cbar_ax2.ax.tick_params(labelsize=15)
    cbar_ax2.set_ticklabels(['0', '1'])
    # ax2.hlines(y=54, xmin=54-20, xmax=54+20, colors='gold', linestyles='dashed', lw=1)
    ax2.hlines(y=index_im_corr_norm[0], xmin=index_im_corr_norm[1]-8, xmax=index_im_corr_norm[1]+8, colors='gold', linestyles='dashed', lw=1)
    ax2.vlines(x=index_im_corr_norm[1], ymin=index_im_corr_norm[0]-8, ymax=index_im_corr_norm[0]+8, colors='red', linestyles='dashed', lw=1)
    index_im_corr_norm
    ax2.axis('off')
    scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
    ax2.add_artist(scalebar)
    plt.show()
    # fig1.savefig(save_path+'iso_roi_images_'+str(i)+'.png',dpi=300, bbox_inches='tight')
#
    # --- Fit line prifile  + FWHM calculation -------------------------------------
    def lor(x, amp1, cen1, wid1,b,off):
        return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + b*x + off
    def gauss(x, amp1,cen1,sigma1,b):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + b*x

    #--- weights, saved_ints, and xfit2 have to be altered
    # x=np.linspace(0,110,110)*0.195:
    x=np.linspace(-8,8,16)*0.195
    meancorr = sum(x * im_corr_norm_plot[(index_im_corr_norm[1]-8):(index_im_corr_norm[1]+8),index_im_corr_norm[0]]) / sum(im_corr_norm_plot[(index_im_corr_norm[1]-8):(index_im_corr_norm[1]+8),index_im_corr_norm[0]])
    sigmacorr = np.sqrt(sum(im_corr_norm_plot[(index_im_corr_norm[1]-8):(index_im_corr_norm[1]+8),index_im_corr_norm[0]] * (x - meancorr)**2) / sum(im_corr_norm_plot[(index_im_corr_norm[1]-8):(index_im_corr_norm[1]+8),index_im_corr_norm[0]]))

    parscorr, covcorr = curve_fit(f=lor,xdata=x,ydata=im_corr_norm_plot[(index_im_corr_norm[1]-8):(index_im_corr_norm[1]+8),index_im_corr_norm[0]],p0=[max(im_corr_norm_plot[(index_im_corr_norm[1]-8):(index_im_corr_norm[1]+8),index_im_corr_norm[0]]), meancorr, sigmacorr,1,0])


    meanraw = sum(x * im_raw_norm_plot[(index_im_raw_norm[1]-8):(index_im_raw_norm[1]+8),index_im_raw_norm[0]]) / sum(im_raw_norm_plot[(index_im_raw_norm[1]-8):(index_im_raw_norm[1]+8),index_im_raw_norm[0]])
    sigmaraw = np.sqrt(sum(im_raw_norm_plot[(index_im_raw_norm[1]-8):(index_im_raw_norm[1]+8),index_im_raw_norm[0]] * (x - meanraw)**2) / sum(im_raw_norm_plot[(index_im_raw_norm[1]-8):(index_im_raw_norm[1]+8),index_im_raw_norm[0]]))

    parsraw, covraw = curve_fit(f=lor,xdata=x,ydata=im_raw_norm_plot[(index_im_raw_norm[1]-8):(index_im_raw_norm[1]+8),index_im_raw_norm[0]],p0=[max(im_raw_norm_plot[(index_im_corr_norm[1]-8):(index_im_corr_norm[1]+8),index_im_raw_norm[0]]), meanraw, sigmaraw,1,0])

    xfit = np.linspace(-8,8,200)*0.195
    fit_raw = lor(xfit,*parsraw.tolist())
    fit_corr = lor(xfit,*parscorr.tolist())

    FWHM_fit_raw = xfit[fit_raw >= 0.5*(np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw)]
    FWHM_fit_corr = xfit[fit_corr >= 0.5*(np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr)]

    # FWHM_fit_raw = xfit[fit_raw >= 0.5*(np.amax(fit_raw)-np.amin(fit_raw))]
    # FWHM_fit_corr = xfit[fit_corr >= 0.5*np.amax((fit_corr)-np.amin(fit_raw))]

    # save results for each point in global arrays.
    results_fit_corr[i, :]=fit_corr
    results_fit_raw[i, :]=fit_raw
    results_raw[i, :]=im_raw_norm_plot[ (index_im_raw_norm[1]-8):(index_im_raw_norm[1]+8),index_im_raw_norm[0]]
    results_corr[i, :]=im_corr_norm_plot[(index_im_raw_norm[1]-8):(index_im_raw_norm[1]+8), index_im_raw_norm[0]]

    results_max_corr[i, :]=max_corr_norm
    results_max_raw[i, :]=max_raw_norm

    results_i_corr[i, :]=np.sum(im_corr)
    results_i_raw[i, :]=np.sum(im_raw)

    results_max_corr_fit[i, :]=np.amax(fit_corr)
    results_max_raw_fit[i, :]=np.amax(fit_raw)

    results_fwhm_raw[i,:]=round(FWHM_fit_raw[-1]-FWHM_fit_raw[0],3)
    results_fwhm_corr[i,:]=round(FWHM_fit_corr[-1]-FWHM_fit_corr[0],3)

    results_FWHM_fit_raw[i].append(FWHM_fit_raw)
    results_FWHM_fit_corr[i].append(FWHM_fit_corr)
    # -----------line profile single
    # fig2=plt.figure('line profile single',figsize = (11,11),linewidth=2.0)
    # plt.plot(xfit,fit_corr,'-', color='darkblue',label = 'correction fit')
    # plt.plot(xfit,fit_raw,'-', color='darkorange',label = 'aberration fit')
    # plt.plot(x,im_raw_norm_plot[(index_im_raw_norm[1]-8):(index_im_raw_norm[1]+8), index_im_raw_norm[0]], '.', color='darkorange',label='aberration')
    # plt.plot(x,im_corr_norm_plot[(index_im_corr_norm[1]-8):(index_im_corr_norm[1]+8),index_im_corr_norm[0]], '.', color='darkblue',label='correction')
    # # plt.plot(FWHM_fit_raw,0.5*(np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw)*np.ones(np.shape(FWHM_fit_raw)),'.g',label = 'FWHM_raw =' + str(round(FWHM_fit_raw[-1]-FWHM_fit_raw[0],3)) + '$\mu m$')
    # # plt.plot(FWHM_fit_corr,0.5*(np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr)*np.ones(np.shape(FWHM_fit_corr)),color='darkblue',linestyle='dashed',label = 'FWHM_corr=' + str(round(FWHM_fit_corr[-1]-FWHM_fit_corr[0],3)) + '$\mu m$')
    # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
    # plt.ylabel(r'$Normalized \quad intensity $')
    # plt.legend(loc='upper left')
    # # plt.grid()
    # # plt.xticks(round(np.linspace(-1, 1,10))
    # # plt.title('aniso fit FWHM corr vs raw')
    # plt.show()
    # fig2.savefig(save_path+'iso_xy_line_profile_quant_'+str(i)+'.png',dpi=330, bbox_inches='tight')

    # fig2bis=plt.figure('line profile single',figsize = (10,10),linewidth=2.0)
    # plt.plot(xfit,fit_corr,'-', color='darkblue',label = 'correction fit')
    # plt.plot(xfit,fit_raw,'-', color='darkorange',label = 'aberration fit')
    # plt.plot(x,im_raw_norm_plot[(index_im_raw_norm[1]-8):(index_im_raw_norm[1]+8), index_im_raw_norm[0]], 'o', color='darkorange',label='aberration')
    # plt.plot(x,im_corr_norm_plot[(index_im_corr_norm[1]-8):(index_im_corr_norm[1]+8),index_im_corr_norm[0]], 'o', color='darkblue',label='correction')
    # # plt.plot(FWHM_fit_raw,0.5*(np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw)*np.ones(np.shape(FWHM_fit_raw)),'.g')
    # # plt.plot(FWHM_fit_corr,0.5*(np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr)*np.ones(np.shape(FWHM_fit_corr)),color='darkblue',linestyle='dashed')
    # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
    # plt.ylabel(r'$Normalized \quad intensity $')
    # # plt.legend(loc='upper left')
    # # plt.grid()
    # # plt.xticks(round(np.linspace(-1, 1,10))
    # # plt.title('aniso fit FWHM corr vs raw')
    # plt.show()
    # fig2bis.savefig(save_path+'xy_line_profile'+str(i)+'.png',dpi=300, bbox_inches='tight')

# ----------------- Lines profiles
# fig_profiles, axs = plt.subplots(4,4,figsize=(20, 20))
# fig_profiles.subplots_adjust(hspace = .1, wspace=.1)
# axs = axs.ravel()
#
# for j in range(len(folders)):
#
#
#     axs[j].plot(xfit,results_fit_raw[j, :],'-', color='darkorange', label='raw')
#     axs[j].plot(xfit,results_fit_corr[j,:],'-', color='darkblue',label='corrected')
#
#     axs[j].plot(x,results_raw[j, :],'o', color='darkorange')
#     axs[j].plot(x,results_corr[j,:],'o', color='darkblue')
#
#     # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
#     # plt.ylabel(r'$Normalized \quad intensity $')
#
#     # axs[j].legend(loc='upper left')
#     axs[j].grid()
#     axs[j].set_xticks(np.linspace(-2, 2,5))
#     axs[j].set_yticks(np.linspace(0,1,3))
#
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
# fig_profiles.text(0.5, 0.01, r'$ Lateral \quad distance \quad (\mu m)$', ha='center', fontsize= MEDIUM_SIZE)
# fig_profiles.text(0.02, 0.5, r'$Normalized \quad intensity $', va='center', rotation='vertical', fontsize=MEDIUM_SIZE)
#
# # legend_elements=[Line2D('.-', color='darkorange',label = 'raw'), Line2D('.-', color='darkblue',label = 'corrected')]
# axs[15].legend( loc='upper center', bbox_to_anchor=(0.5, -0.4),fancybox=False, shadow=False, ncol=2)
# plt.show()
# fig_profiles.savefig(save_path+'total_xy_line_profile.png',dpi=300, bbox_inches='tight')
# ----------------------------Figure 3 panel e top ---------------------------------------------------
fig_profiles, axs = plt.subplots(1,2,figsize=(10, 5))
fig_profiles.subplots_adjust(hspace = .1, wspace=.1)
axs = axs.ravel()
points=[0,6]
for index, j  in enumerate(points):


    axs[index].plot(xfit,results_fit_raw[j, :],'-', color='darkorange', label='raw')
    axs[index].plot(xfit,results_fit_corr[j,:],'-', color='darkblue',label='corrected')

    axs[index].plot(x,results_raw[j, :],'o', color='darkorange')
    axs[index].plot(x,results_corr[j,:],'o', color='darkblue')
    # axs[index].plot(results_FWHM_fit_raw[j][0],0.5*(np.amax(results_fit_corr[j])-np.amin(results_fit_corr[j]))+np.amin(results_fit_corr[j])*np.ones(np.shape(results_FWHM_fit_raw[j][0])),color='darkblue',linestyle='dashed')

    # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
    # plt.ylabel(r'$Normalized \quad intensity $')

    # axs[j].legend(loc='upper left')
    # axs[index].grid()
    axs[index].set_xticks(np.linspace(-1, 1,3))
    axs[index].set_yticks(np.linspace(0,1,3))

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
fig_profiles.text(0.5, -0.05, r'$ Lateral \quad distance \quad (\mu m)$', ha='center', fontsize= BIGGER_SIZE)
fig_profiles.text(0.03, 0.5, r'$Normalized \quad intensity $', va='center', rotation='vertical', fontsize=BIGGER_SIZE)

legend_elements=[Line2D('.-', color='darkorange',label = 'raw'), Line2D('.-', color='darkblue',label = 'corrected')]
axs[15].legend( loc='upper center', bbox_to_anchor=(0.5, -0.4),fancybox=False, shadow=False, ncol=2)
plt.show()
fig_profiles.savefig(save_path+'total_xy_line_p0_p6.png',dpi=300, bbox_inches='tight')
# -------------------Quantifications
av_fwhm_lateral_corr=np.average(results_fwhm_corr)
std_fwhm_lateral_corr=np.std(results_fwhm_corr)
av_fwhm_lateral_raw=np.average(results_fwhm_raw)
std_fwhm_lateral_raw=np.std(results_fwhm_raw)

results_fwhm_raw_bis=(np.delete(results_fwhm_raw,(0)))
results_fwhm_raw_bis=(np.delete(results_fwhm_raw_bis,(5)))
av_fwhm_lateral_raw_bis=np.average(results_fwhm_raw_bis)
std_fwhm_lateral_raw_bis=np.std(results_fwhm_raw_bis)
results_fwhm_corr_bis=(np.delete(results_fwhm_corr,(0)))
results_fwhm_corr_bis=(np.delete(results_fwhm_corr_bis,(5)))
av_fwhm_lateral_corr_bis=np.average(results_fwhm_corr_bis)
std_fwhm_lateral_corr_bis=np.std(results_fwhm_corr_bis)


fwhm_improvement_lateral=results_fwhm_corr/results_fwhm_raw
av_fwhm_improvement_lateral=np.average(fwhm_improvement_lateral)
std_fwhm_improvement_lateral=np.std(fwhm_improvement_lateral)
I_improvement_perc=(results_max_corr-results_max_raw)*100/results_max_corr
av_I_lateral_perc=np.average(I_improvement_perc)
std_I_lateral_perc=np.std(I_improvement_perc)
I_improvement_factor=(results_max_corr/results_max_raw)
av_I_lateral_factor=np.average(I_improvement_factor)
std_I_lateral_factor=np.std(I_improvement_factor)
table_results=tabulate([['av_I_Improvement percentage ',av_I_lateral_perc ],['std_I_Improvement percentage ',std_I_lateral_perc],['av_I_Improvement factor ',av_I_lateral_factor ],['std_I_Improvement factor ',std_I_lateral_factor],['av_fwhm_improvement_lateral',av_fwhm_improvement_lateral ],['std_fwhm_improvement_lateral',std_fwhm_improvement_lateral],['av_fwhm_lateral_corr',av_fwhm_lateral_corr ],['std_fwhm_lateral_corr',std_fwhm_lateral_corr],['av_fwhm_lateral_raw',av_fwhm_lateral_raw ],['std_fwhm_lateral_raw',std_fwhm_lateral_raw],['av_fwhm_lateral_corr_bis',av_fwhm_lateral_corr_bis ],['std_fwhm_lateral_corr_bis',std_fwhm_lateral_corr_bis],['av_fwhm_lateral_raw_bis',av_fwhm_lateral_raw_bis ],['std_fwhm_lateral_raw_bis',std_fwhm_lateral_raw_bis]],headers=['Correction improvement'])

f = open(save_path+'Corrected-improvement-xy.txt', 'w')
f.write(table_results)
f.close()
# #----------------------plot intensities----------------------------------------
# fig_int_max =plt.figure('intensities', figsize=(10, 3), linewidth=2.0)
# x_p = np.arange(0,len(folders), 1)
# plt.plot(x_p, results_max_raw_fit,'o', color='darkorange',  label = 'raw')
# plt.plot(x_p, results_max_corr_fit,'o', color='darkblue',label = 'corrected')
# plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE),
# plt.ylabel(r'$ I \quad  max. \quad norm.$',fontsize= SMALL_SIZE)
# plt.xticks(x_p, x_p+1)
# plt.ylim([0,1.2])
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
# plt.show()
# fig_int_max.savefig(save_path+'intensities_norm.png',dpi=300, bbox_inches='tight')
#
# fig_int =plt.figure('intensities', figsize=(10, 3), linewidth=2.0)
# x_p = np.arange(0,len(folders), 1)
# plt.plot(x_p, results_i_raw,'o',color='darkorange',  label = 'raw')
# plt.plot(x_p, results_i_corr, 'o', color='darkblue',label = 'corrected')
# plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE)
# plt.ylabel(r'$  Intensity \quad (arb. unit)$',fontsize= SMALL_SIZE)
# plt.xticks(x_p, x_p+1)
# plt.ylim([0,np.amax(results_i_corr)+100000])
# plt.yticks(np.linspace(0,9,4)*100000)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
# plt.show()
# fig_int.savefig(save_path+'intensities.png',dpi=300, bbox_inches='tight')





# # -------------------------------- plot lines profile xz, panel e and d bottom----------------------------

gauss_fit_z_path="M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\method_validation\\precise_22_12\\stack\\stack_20201222-152845_aniso_0.5um\\data_analysis\\gauss_fit_z\\"

folders=os.listdir(gauss_fit_z_path)
# variables to save all data
results_z_fit_corr=np.zeros((len(folders), 200), dtype=float)
results_z_fit_raw=np.zeros((len(folders), 200), dtype=float)
results_z_raw=np.zeros((len(folders), 92), dtype=float)
results_z_corr=np.zeros((len(folders), 92), dtype=float)

results_z_FWHM_fit_raw=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
results_z_FWHM_fit_corr=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

results_z_max_corr=np.zeros((len(folders), 1), dtype=float)
results_z_max_raw=np.zeros((len(folders), 1), dtype=float)
results_z_fwhm_corr=np.zeros((len(folders), 1), dtype=float)
results_z_fwhm_raw=np.zeros((len(folders), 1), dtype=float)
results_z_max_corr_fit=np.zeros((len(folders), 1), dtype=float)
results_z_max_raw_fit=np.zeros((len(folders), 1), dtype=float)
results_z_i_corr=np.zeros((len(folders), 1), dtype=float)
results_z_i_raw=np.zeros((len(folders), 1), dtype=float)
for i in range (len(folders)):
    data_path=gauss_fit_z_path+folders[i]+'\\'
    files = [['z_original.tif','z_corrected.tif']]
    print(folders[i])
    for name in files[0]:
        print(name)
        image = Image.open(data_path+name)
        if (name == 'z_original.tif'):
            im_raw=np.array(image)
            print(name)
        im_corr=np.array(image)
    #---------------Im corr
    min_corr=np.min(im_corr)
    max_corr=np.max(im_corr)
    print('max/min corr',min_corr, max_corr)
    # --Normalization corr
    # if i==6 or i==3:
    #     im_corr_norm=(im_corr)/(max_corr)
    #     print('if')
    # else:
    #     im_corr_norm=(im_corr-min_corr)/(max_corr-min_corr)
    im_corr_norm=(im_corr)/(max_corr)
    index_im_corr_norm=np.unravel_index(im_corr_norm.argmax(),im_corr_norm.shape)
    im_corr_norm_plot=im_corr_norm[:,index_im_corr_norm[1]-20:index_im_corr_norm[1]+20]
    # im_corr_norm_plot=im_corr_norm
    min_corr_norm=np.min(im_corr_norm_plot)
    max_corr_norm=np.max(im_corr_norm_plot)
    print(' min/max corr_norm', min_corr_norm, max_corr_norm)
    index_im_corr_norm=np.unravel_index(im_corr_norm_plot.argmax(),im_corr_norm_plot.shape)
    print('index im_corr_norm', index_im_corr_norm)
#     #------------------------Im original
    min_raw=np.min(im_raw)
    max_raw=np.max(im_raw)
    print('max/min raw',min_raw, max_raw)
    # --Normalization raw
    im_raw_norm=(im_raw)/(max_corr)
    # if i==6 or i==3:
    #     im_raw_norm=(im_raw)/(max_corr)
    #     print('if')
    # else:
    #     im_raw_norm=(im_raw-min_corr)/(max_corr-min_corr)
    index_im_raw_norm=np.unravel_index(im_raw_norm.argmax(),im_raw_norm.shape)
    im_raw_norm_plot=im_raw_norm[:,index_im_raw_norm[1]-20:index_im_raw_norm[1]+20]
    # im_raw_norm_plot=im_raw_norm
    min_raw_norm=np.min(im_raw_norm_plot)
    max_raw_norm=np.max(im_raw_norm_plot)
    print(' min/max raw_norm', min_raw_norm, max_raw_norm)
    index_im_raw_norm=np.unravel_index(im_raw_norm_plot.argmax(),im_raw_norm_plot.shape)
    print('index im_raw_norm', index_im_raw_norm)
#
#
#     # fig1, (ax1, ax2)= plt.subplots(1, 2,figsize = (5,5),linewidth=2.0)
#     # plot_raw=ax1.imshow(im_raw_norm_plot, vmin=0, vmax=round(np.amax(im_raw_norm_plot),2), cmap = 'jet')
#     # cbar_ax1 = plt.colorbar(plot_raw,ax=ax1,ticks=[0, round(np.amax(im_raw_norm_plot),2)], orientation="horizontal", fraction=0.021, pad=0.02)
#     # cbar_ax1.set_ticklabels(['0', str(round(np.amax(im_raw_norm_plot),2))])
#     # cbar_ax1.ax.tick_params(labelsize=15)
#     # ax1.vlines(x=index_im_raw_norm[1], ymin=0, ymax=im_raw_norm_plot.shape[0]-1, colors='red', linestyles='dashed', lw=1)
#     # ax1.hlines(y=index_im_raw_norm[0],xmin=0,  xmax=im_raw_norm_plot.shape[1]-1, colors='black', linestyles='dashed', lw=1)
#     #
#     #
#     # ax1.axis('off')
#     # scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
#     # ax1.add_artist(scalebar)
#     #
#     # plot_aniso=ax2.imshow(im_corr_norm_plot, vmin=0, vmax=1, cmap = 'jet')
#     # cbar_ax2 = plt.colorbar(plot_aniso,ax=ax2,ticks=[0, max_corr_norm], orientation="horizontal", fraction=0.021, pad=0.02)
#     # cbar_ax2.ax.tick_params(labelsize=15)
#     # cbar_ax2.set_ticklabels(['0', '1'])
#     #
#     # ax2.vlines(x=index_im_corr_norm[1], ymin=0, ymax=im_corr_norm_plot.shape[0]-1, colors='red', linestyles='dashed', lw=1)
#     # ax2.hlines(y=index_im_corr_norm[0], xmin=0, xmax=im_corr_norm_plot.shape[1]-1, colors='gold', linestyles='dashed', lw=1)
#     # index_im_corr_norm
#     # ax2.axis('off')
#     # scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
#     # ax2.add_artist(scalebar)
#     # plt.show()
#     # fig1.savefig(save_path+'zy_roi_images_'+str(i)+'.png',dpi=300, bbox_inches='tight')
#
#     # --- Fit line prifile  + FWHM calculation -------------------------------------
    def lor(x, amp1, cen1, wid1,b,off):
        return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + b*x + off
    def gauss(x, amp1,cen1,sigma1,b):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + b*x

    #--- weights, saved_ints, and xfit2 have to be altered

    x=np.linspace(-46,46,92)*0.195
    meancorr = sum(x * im_corr_norm_plot[:,index_im_corr_norm[1]]) / sum(im_corr_norm_plot[:,index_im_corr_norm[1]])
    sigmacorr = np.sqrt(sum(im_corr_norm_plot[:,index_im_corr_norm[1]] * (x - meancorr)**2) / sum(im_corr_norm_plot[:,index_im_corr_norm[1]]))

    parscorr, covcorr = curve_fit(f=lor,xdata=x,ydata=im_corr_norm_plot[:,index_im_corr_norm[1]],p0=[max(im_corr_norm_plot[:,index_im_corr_norm[1]]), meancorr, sigmacorr,1,0])


    meanraw = sum(x * im_raw_norm_plot[:,index_im_raw_norm[1]]) / sum(im_raw_norm_plot[:,index_im_raw_norm[1]])
    sigmaraw = np.sqrt(sum(im_raw_norm_plot[:,index_im_raw_norm[1]] * (x - meanraw)**2) / sum(im_raw_norm_plot[:,index_im_raw_norm[1]]))

    parsraw, covraw = curve_fit(f=lor,xdata=x,ydata=im_raw_norm_plot[:,index_im_raw_norm[1]],p0=[max(im_raw_norm_plot[:,index_im_raw_norm[1]]), meanraw, sigmaraw,1,0])

    xfit = np.linspace(-46,46,200)*0.195
    fit_raw = lor(xfit,*parsraw.tolist())
    fit_corr = lor(xfit,*parscorr.tolist())

    FWHM_fit_raw = xfit[fit_raw >= 0.5*(np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw)]
    FWHM_fit_corr = xfit[fit_corr >= 0.5*(np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr)]

    # FWHM_fit_raw = xfit[fit_raw >= 0.5*(np.amax(fit_raw)-np.amin(fit_raw))]
    # FWHM_fit_corr = xfit[fit_corr >= 0.5*np.amax((fit_corr)-np.amin(fit_raw))]

    # save results for each point in global arrays.
    results_z_fit_corr[i, :]=fit_corr
    results_z_fit_raw[i, :]=fit_raw
    results_z_raw[i, :]=im_raw_norm_plot[ :,index_im_raw_norm[1]]
    results_z_corr[i, :]=im_corr_norm_plot[:, index_im_raw_norm[1]]

    results_z_max_corr[i, :]=max_corr_norm
    results_z_max_raw[i, :]=max_raw_norm

    results_z_i_corr[i, :]=np.sum(im_corr)
    results_z_i_raw[i, :]=np.sum(im_raw)

    results_z_max_corr_fit[i, :]=np.amax(fit_corr)
    results_z_max_raw_fit[i, :]=np.amax(fit_raw)

    results_z_fwhm_raw[i,:]=round(FWHM_fit_raw[-1]-FWHM_fit_raw[0],3)
    results_z_fwhm_corr[i,:]=round(FWHM_fit_corr[-1]-FWHM_fit_corr[0],3)

    results_z_FWHM_fit_raw[i].append(FWHM_fit_raw)
    results_z_FWHM_fit_corr[i].append(FWHM_fit_corr)
# #     # -----------line profile single
#     # fig2=plt.figure('line profile single',figsize = (11,11),linewidth=2.0)
#     # plt.plot(xfit,fit_corr,'-', color='darkblue',label = 'correction fit')
#     # plt.plot(xfit,fit_raw,'-', color='darkorange',label = 'aberration fit')
#     # plt.plot(x,im_raw_norm_plot[:, index_im_raw_norm[1]], '.', color='darkorange',label='aberration')
#     # plt.plot(x,im_corr_norm_plot[:,index_im_corr_norm[1]], '.', color='darkblue',label='correction')
#     # plt.plot(FWHM_fit_raw,0.5*(np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw)*np.ones(np.shape(FWHM_fit_raw)),'.g',label = 'FWHM_raw =' + str(round(FWHM_fit_raw[-1]-FWHM_fit_raw[0],3)) + '$\mu m$')
#     # plt.plot(FWHM_fit_corr,0.5*(np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr)*np.ones(np.shape(FWHM_fit_corr)),color='darkblue',linestyle='dashed',label = 'FWHM_corr=' + str(round(FWHM_fit_corr[-1]-FWHM_fit_corr[0],3)) + '$\mu m$')
#     # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
#     # plt.ylabel(r'$Normalized \quad intensity $')
#     # plt.legend(loc='upper left')
#     # plt.grid()
#     # # plt.xticks(round(np.linspace(-1, 1,10))
#     # # plt.title('aniso fit FWHM corr vs raw')
#     # plt.show()
#     # fig2.savefig(save_path+'zy_line_profile_quant_'+str(i)+'.png',dpi=300, bbox_inches='tight')
#     #
    # fig2bis=plt.figure('line profile single',figsize = (10,10),linewidth=2.0)
    # plt.plot(xfit,fit_corr,'-', color='darkblue',label = 'correction fit')
    # plt.plot(xfit,fit_raw,'-', color='darkorange',label = 'aberration fit')
    # plt.plot(x,im_raw_norm_plot[:, index_im_raw_norm[1]], 'o', color='darkorange',label='aberration')
    # plt.plot(x,im_corr_norm_plot[:,index_im_corr_norm[1]], 'o', color='darkblue',label='correction')
    # # plt.plot(FWHM_fit_raw,0.5*(np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw)*np.ones(np.shape(FWHM_fit_raw)),'.g')
    # # plt.plot(FWHM_fit_corr,0.5*(np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr)*np.ones(np.shape(FWHM_fit_corr)),color='darkblue',linestyle='dashed')
    # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
    # plt.ylabel(r'$Normalized \quad intensity $')
    # # plt.legend(loc='upper left')
    # # plt.grid()
    # # plt.xticks(round(np.linspace(-1, 1,10))
    # # plt.title('aniso fit FWHM corr vs raw')
    # plt.show()
    # fig2bis.savefig(save_path+'zy_line_profile'+str(i)+'.png',dpi=300, bbox_inches='tight')
#
# #----------------- Lines profiles
# # fig_profiles, axs = plt.subplots(4,4,figsize=(20, 20))
# # fig_profiles.subplots_adjust(hspace = .1, wspace=.1)
# # axs = axs.ravel()
# #
# # for j in range(len(folders)):
# #
# #     axs[j].plot(xfit,results_z_fit_raw[j, :],'-', color='darkorange', label='raw')
# #     axs[j].plot(xfit,results_z_fit_corr[j,:],'-', color='darkblue',label='corrected')
# #
# #     axs[j].plot(x,results_z_raw[j, :],'o', color='darkorange')
# #     axs[j].plot(x,results_z_corr[j,:],'o', color='darkblue')
# #
# #     # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
# #     # plt.ylabel(r'$Normalized \quad intensity $')
# #
# #     # axs[j].legend(loc='upper left')
# #     axs[j].grid()
# #     # axs[j].set_xticks(np.linspace(-2, 2,5))
# #     axs[j].set_yticks(np.linspace(0,1,3))
# #
# # # Hide x labels and tick labels for top plots and y ticks for right plots.
# # for ax in axs.flat:
# #     ax.label_outer()
# # fig_profiles.text(0.5, 0.01, r'$ Axial \quad distance \quad (\mu m)$', ha='center', fontsize= MEDIUM_SIZE)
# # fig_profiles.text(0.02, 0.5, r'$Normalized \quad intensity $', va='center', rotation='vertical', fontsize=MEDIUM_SIZE)
# #
# # # legend_elements=[Line2D('.-', color='darkorange',label = 'raw'), Line2D('.-', color='darkblue',label = 'corrected')]
# # axs[15].legend( loc='upper center', bbox_to_anchor=(0.5, -0.4),fancybox=False, shadow=False, ncol=2)
# # plt.show()
# # fig_profiles.savefig(save_path+'total_zy_line_profile.png',dpi=300, bbox_inches='tight')
# #-------------------------------------------------------------------------------
fig_profiles, axs = plt.subplots(1,2,figsize=(10, 5))
fig_profiles.subplots_adjust(hspace = .1, wspace=.1)
axs = axs.ravel()
points=[0,6]
for index, j  in enumerate(points):


    axs[index].plot(xfit,results_z_fit_raw[j, :],'-', color='darkorange', label='raw')
    axs[index].plot(xfit,results_z_fit_corr[j,:],'-', color='darkblue',label='corrected')

    axs[index].plot(x,results_z_raw[j, :],'o', color='darkorange')
    axs[index].plot(x,results_z_corr[j,:],'o', color='darkblue')
    # axs[index].plot(results_FWHM_fit_raw[j][0],0.5*(np.amax(results_fit_corr[j])-np.amin(results_fit_corr[j]))+np.amin(results_fit_corr[j])*np.ones(np.shape(results_FWHM_fit_raw[j][0])),color='darkblue',linestyle='dashed')

    # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
    # plt.ylabel(r'$Normalized \quad intensity $')

    # axs[j].legend(loc='upper left')
    # axs[index].grid()
    axs[index].set_xticks(np.linspace(-8, 8,5))
    axs[index].set_yticks(np.linspace(0,1,3))

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
fig_profiles.text(0.5, -0.05, r'$ Axial \quad distance \quad (\mu m)$', ha='center', fontsize= BIGGER_SIZE)
fig_profiles.text(0.03, 0.5, r'$Normalized \quad intensity $', va='center', rotation='vertical', fontsize=BIGGER_SIZE)

# legend_elements=[Line2D('.-', color='darkorange',label = 'raw'), Line2D('.-', color='darkblue',label = 'corrected')]
# axs[15].legend( loc='upper center', bbox_to_anchor=(0.5, -0.4),fancybox=False, shadow=False, ncol=2)
plt.show()
fig_profiles.savefig(save_path+'total_zy_line_p0_p6.png',dpi=300, bbox_inches='tight')
# # #-------------------Quantifications

av_fwhm_axial_corr=np.average(results_z_fwhm_corr)
std_fwhm_axial_corr=np.std(results_z_fwhm_corr)
av_fwhm_axial_raw=np.average(results_z_fwhm_raw)
std_fwhm_axial_raw=np.std(results_z_fwhm_raw)

results_z_fwhm_raw_bis=(np.delete(results_z_fwhm_raw,(0)))
results_z_fwhm_raw_bis=(np.delete(results_z_fwhm_raw_bis,(5)))
av_fwhm_axial_raw_bis=np.average(results_z_fwhm_raw_bis)
std_fwhm_axial_raw_bis=np.std(results_z_fwhm_raw_bis)
results_z_fwhm_corr_bis=(np.delete(results_z_fwhm_corr,(0)))
results_z_fwhm_corr_bis=(np.delete(results_z_fwhm_corr_bis,(5)))
av_fwhm_axial_corr_bis=np.average(results_z_fwhm_corr_bis)
std_fwhm_axial_corr_bis=np.std(results_z_fwhm_corr_bis)


fwhm_improvement_axial=results_z_fwhm_corr/results_z_fwhm_raw
av_fwhm_improvement_axial=np.average(fwhm_improvement_axial)
std_fwhm_improvement_axial=np.std(fwhm_improvement_axial)

I_axial_improvement_perc=(results_z_max_corr-results_z_max_raw)*100/results_z_max_corr
av_I_axial_perc=np.average(I_axial_improvement_perc)
std_I_axial_perc=np.std(I_axial_improvement_perc)
I_axial_improvement_factor=(results_z_max_corr/results_z_max_raw)
av_I_axial_factor=np.average(I_axial_improvement_factor)
std_I_axial_factor=np.std(I_axial_improvement_factor)

table_results_z=tabulate([['av_I_axial_Improvement percentage ',av_I_axial_perc ],['std_I_axial_Improvement percentage ',std_I_axial_perc],['av_I_axial_Improvement factor ',av_I_axial_factor ],['std_I_axial_Improvement factor ',std_I_axial_factor],['av_fwhm_improvement_axial',av_fwhm_improvement_axial ],['std_fwhm_improvement_axial',std_fwhm_improvement_axial],['av_fwhm_axial_corr',av_fwhm_axial_corr ],['std_fwhm_axial_corr',std_fwhm_axial_corr],['av_fwhm_axial_raw',av_fwhm_axial_raw ],['std_fwhm_axial_raw',std_fwhm_axial_raw],['av_fwhm_axial_corr_bis',av_fwhm_axial_corr_bis ],['std_fwhm_axial_corr_bis',std_fwhm_axial_corr_bis],['av_fwhm_axial_raw_bis',av_fwhm_axial_raw_bis ],['std_fwhm_axial_raw_bis',std_fwhm_axial_raw_bis]],headers=['Correction improvement axial'])

f = open(save_path+'Corrected-improvement_z.txt', 'w')
f.write(table_results_z)
f.close()
# #----------------------plot intensities----------------------------------------
# fig_int_max =plt.figure('intensities', figsize=(10, 3), linewidth=2.0)
# x_p = np.arange(0,len(folders), 1)
# plt.plot(x_p, results_max_raw_fit,'o', color='darkorange',  label = 'raw')
# plt.plot(x_p, results_max_corr_fit,'o', color='darkblue',label = 'corrected')
# plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE),
# plt.ylabel(r'$ I \quad  max. \quad norm.$',fontsize= SMALL_SIZE)
# plt.xticks(x_p, x_p+1)
# plt.ylim([0,1.2])
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
# plt.show()
# fig_int_max.savefig(save_path+'intensities_norm.png',dpi=300, bbox_inches='tight')
#
# fig_int =plt.figure('intensities', figsize=(10, 3), linewidth=2.0)
# x_p = np.arange(0,len(folders), 1)
# plt.plot(x_p, results_i_raw,'o',color='darkorange',  label = 'raw')
# plt.plot(x_p, results_i_corr, 'o', color='darkblue',label = 'corrected')
# plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE)
# plt.ylabel(r'$  Intensity \quad (arb. unit)$',fontsize= SMALL_SIZE)
# plt.xticks(x_p, x_p+1)
# plt.ylim([0,np.amax(results_i_corr)+100000])
# plt.yticks(np.linspace(0,9,4)*100000)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
# plt.show()
# fig_int.savefig(save_path+'intensities.png',dpi=300, bbox_inches='tight')

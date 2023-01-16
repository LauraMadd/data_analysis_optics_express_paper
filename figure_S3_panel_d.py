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



SMALL_SIZE = 15
MEDIUM_SIZE = 30
BIGGER_SIZE=30
# # for total plot line profiles
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
##-------------------------------------------------------------------------------

#--------------------------------------------------------------------
# Script used to plot metrics and intensities profiles from datasets of spatial
#proximity experiments. The folder containing the data to plot has to be changed
# semi_path_metrics amd semi_path_points as well as the save_path and the flag save

#-------------------------Paths ---------------------------------
#----------to change------------
semi_path_metrics='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\test_gui_AO\\30_03\\fep_repetition\\fep_spatial_astm2\\'
semi_path_points='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\test_gui_AO\\30_03\\fep_repetition\\data_analysis\\gauss_fit_astm2\\'
flag_save='astm2'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_6\\'
#----------------------------------
#-------------------------------------------------------------------------------
# # #----------------------------------Roi xy planes -------------------------------

#
folders=os.listdir(semi_path_points)
results_fit_corr=np.zeros((len(folders), 2000), dtype=float)
results_fit_raw=np.zeros((len(folders), 2000), dtype=float)
results_raw=np.zeros((len(folders), 40), dtype=float)
results_corr=np.zeros((len(folders), 40), dtype=float)
results_max_corr=np.zeros((len(folders), 1), dtype=float)
results_max_raw=np.zeros((len(folders), 1), dtype=float)
for i in range (len(folders)):


    data_path=semi_path_points+'\\'+folders[i]+'\\'
    print(data_path)
    files = [['original.tif','corrected.tif']]
    print(folders[i])
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

    # --Normalization
    im_corr_norm=(im_corr)/(max_corr)
    im_corr_norm_plot=im_corr_norm
    # im_corr_norm_plot=im_corr_norm[70:120,70:120]
    min_corr_norm=np.amin(im_corr_norm_plot)
    max_corr_norm=np.amax(im_corr_norm_plot)
    print(' min/max corr_norm', min_corr_norm, max_corr_norm)

    radius_roi=5
    # x=np.linspace(0,radius_roi*2,radius_roi*2)*0.195
    index_im_corr_norm=np.unravel_index(im_corr_norm_plot.argmax(),im_corr_norm_plot.shape)
    print('index im_corr_norm', index_im_corr_norm)

    #------------------------Im original
    min_raw=np.amin(im_raw)
    max_raw=np.amax(im_raw)
    print('max/min raw',min_raw, max_raw)

    # --Normalization
    im_raw_norm=(im_raw)/(max_corr)
    im_raw_norm_plot=im_raw_norm
    # im_raw_norm_plot=im_raw_norm[70:120,70:120]
    min_raw_norm=np.amin(im_raw_norm_plot)
    max_raw_norm=np.amax(im_raw_norm_plot)
    print(' min/max raw_norm', min_raw_norm, max_raw_norm)

    radius_roi=8
    # x=np.linspace(0,radius_roi*2,radius_roi*2)*0.195
    index_im_raw_norm=np.unravel_index(im_raw_norm_plot.argmax(),im_raw_norm_plot.shape)
    print('index im_raw_norm', index_im_raw_norm)

#
# #     #---------------------Plot xy focus---------------------------------------------
    # fig4, (ax1, ax2)= plt.subplots(1, 2,figsize = (5,5),linewidth=2.0)
    # plot_raw=ax1.imshow(im_raw_norm_plot, vmin=0, vmax=max_corr_norm, cmap = 'jet')
    # cbar_ax1 = plt.colorbar(plot_raw,ax=ax1,ticks=[0, max_corr_norm], orientation="horizontal", fraction=0.047, pad=0.01)
    # cbar_ax1.set_ticklabels([r'$0$', r'$1$'])
    # cbar_ax1.ax.tick_params(labelsize=15)
    # ax1.hlines(y=index_im_raw_norm[0], xmin=index_im_raw_norm[1]-radius_roi, xmax=index_im_raw_norm[1]+radius_roi, colors='black', linestyles='dashed', lw=1)
    # ax1.axis('off')
    # scalebar = ScaleBar(0.195, 'um')
    # ax1.add_artist(scalebar)
    #
    # plot_corr=ax2.imshow(im_corr_norm_plot, vmin=0, vmax=1, cmap = 'jet')
    # cbar_ax2 = plt.colorbar(plot_corr,ax=ax2,ticks=[0, max_corr_norm], orientation="horizontal", fraction=0.047, pad=0.01)
    # cbar_ax2.ax.tick_params(labelsize=15)
    # cbar_ax2.set_ticklabels([r'$0$', r'$1$'])
    # # ax2.hlines(y=54, xmin=54-20, xmax=54+20, colors='gold', linestyles='dashed', lw=1)
    # ax2.hlines(y=index_im_corr_norm[0], xmin=index_im_corr_norm[1]-radius_roi, xmax=index_im_corr_norm[1]+radius_roi, colors='black', linestyles='dashed', lw=1)
    # ax2.axis('off')
    # scalebar = ScaleBar(0.195, 'um')
    # ax2.add_artist(scalebar)
    # plt.show()
    # fig4.savefig(save_path+'xy_image_base'+str(folders[i])+'.png',dpi=300, bbox_inches='tight')


    #--------------------------------Line profile ----------------------------------
# --- Fit functions  -------------------------------------
    def lor(x, amp1, cen1, wid1,b,off):
        return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + b*x + off
    def gauss(x, amp1,cen1,sigma1,b):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + b*x
    #--- weights, saved_ints, and xfit2 have to be altered
    # x=np.linspace(0,110,110)*0.195:,index_im_corr_norm[0]
    x=np.linspace(-20,20,40)*0.195
    meancorr = sum(x * im_corr_norm_plot[:,index_im_corr_norm[0]]) / sum(im_corr_norm_plot[:,index_im_corr_norm[0]])
    sigmacorr = np.sqrt(sum(im_corr_norm_plot[:,index_im_corr_norm[0]] * (x - meancorr)**2) / sum(im_corr_norm_plot[:,index_im_corr_norm[0]]))

    parscorr, covcorr = curve_fit(f=lor,xdata=x,ydata=im_corr_norm_plot[:,index_im_corr_norm[0]],p0=[max(im_corr_norm_plot[:,index_im_corr_norm[0]]), meancorr, sigmacorr,1,0])


    meanraw = sum(x * im_raw_norm_plot[:,index_im_raw_norm[0]]) / sum(im_raw_norm_plot[:,index_im_raw_norm[0]])
    sigmaraw = np.sqrt(sum(im_raw_norm_plot[:,index_im_raw_norm[0]] * (x - meanraw)**2) / sum(im_raw_norm_plot[:,index_im_raw_norm[0]]))

    parsraw, covraw = curve_fit(f=lor,xdata=x,ydata=im_raw_norm_plot[:,index_im_raw_norm[0]],p0=[max(im_raw_norm_plot[:,index_im_raw_norm[0]]), meanraw, sigmaraw,1,0])

    xfit = np.linspace(-20,20,2000)*0.195
    fit_raw = lor(xfit,*parsraw.tolist())
    fit_corr = lor(xfit,*parscorr.tolist())
    # FWHM_fit_raw = xfit[fit_raw >= 0.5*np.amax(fit_raw)]
    # FWHM_fit_corr = xfit[fit_corr >= 0.5*np.amax(fit_corr)]
    # FWHM_fit_raw = xfit[fit_raw >= 0.5*(np.amax(fit_raw)-np.amin(fit_raw))]
    # FWHM_fit_corr = xfit[fit_corr >= 0.5*np.amax((fit_corr)-np.amin(fit_raw))]
    FWHM_fit_raw = xfit[fit_raw >= 0.5*(np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw)]
    FWHM_fit_corr = xfit[fit_corr >= 0.5*(np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr)]

    # save results for each point in global arrays.
    results_fit_corr[i, :]=fit_corr
    results_fit_raw[i, :]=fit_raw
    results_raw[i, :]=im_raw_norm_plot[:,index_im_raw_norm[0]]
    results_corr[i, :]=im_corr_norm_plot[:,index_im_corr_norm[0]]

    results_max_corr[i, :]=max_corr_norm
    results_max_raw[i, :]=max_raw_norm

    # #----------------------Figure panel g----------------------------------------
    # fig5=plt.figure('line profile aniso fit FWHM p0 aniso vs p0 raw',figsize = (11,11),linewidth=2.0)
    # plt.plot(xfit,fit_corr,'-b',label = 'correction fit')
    # plt.plot(xfit,fit_raw,'-r',label = 'aberration fit')
    # plt.plot(x,im_raw_norm_plot[index_im_raw_norm[0], :], '.r',label='aberration')
    # plt.plot(x,im_corr_norm_plot[index_im_corr_norm[0], :], '.b',label='correction')
    # plt.plot(FWHM_fit_raw,0.5*((np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw))*np.ones(np.shape(FWHM_fit_raw)),color='red',linestyle='dashed',label = 'FWHM=' + str(round(FWHM_fit_raw[-1]-FWHM_fit_raw[0],3)) + '$\mu m$')
    # plt.plot(FWHM_fit_corr,0.5*((np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr))*np.ones(np.shape(FWHM_fit_corr)),color='blue',linestyle='dashed',label = 'FWHM=' + str(round(FWHM_fit_corr[-1]-FWHM_fit_corr[0],3)) + r'$\mu m$')
    # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
    # plt.ylabel(r'$Normalized \quad intensity $')
    # plt.legend(loc='upper left')
    # # plt.title('aniso fit FWHM corr vs raw')
    # plt.show()
    # fig5.savefig(save_path+flag_save+'_xy_line_profile'+str(folders[i])+ '.png',dpi=300, bbox_inches='tight')

    #---------------------------------Fig total line profiles------------------
fig_profiles, axs = plt.subplots(3,3,figsize=(15, 15))
fig_profiles.subplots_adjust(hspace = .1, wspace=.01)
axs = axs.ravel()
order_plots=[2,1,0,5,4,3,8,7,6]
for j_index, j in enumerate(order_plots):


    axs[j_index].plot(xfit,results_fit_raw[j, :],'-', color='darkorange',label = 'aberration fit')
    axs[j_index].plot(xfit,results_fit_corr[j,:],'-', color='darkblue',label = 'correction fit')

    axs[j_index].plot(x,results_raw[j, :],'o', color='darkorange',label = 'aberration')
    axs[j_index].plot(x,results_corr[j,:],'o', color='darkblue',label = 'correction')

    axs[j_index].set_xticks(range(0, int(np.max(x))+1,1))
    # axs[j_index].title.set_text(str(j))

    # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
    # plt.ylabel(r'$Normalized \quad intensity $')

    # axs[j].legend(loc='upper left')
    axs[j_index].grid()
    axs[j_index].set_xticks(np.linspace(-4,4,5))
    axs[j_index].set_yticks(np.linspace(0,1,3))
# for ax in axs.flat:
#     ax.set(xlabel=r'$ Lateral \quad distance \quad (\mu m)$', ylabel=r'$Normalized \quad intensity $')
#
# # Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
fig_profiles.text(0.5, 0.04, r'$ Lateral \quad distance \quad (\mu m)$', ha='center', fontsize=MEDIUM_SIZE)
fig_profiles.text(0.06, 0.5, r'$Normalized \quad intensity $', va='center', rotation='vertical',fontsize=MEDIUM_SIZE)
# plt.legend(loc='lower right', mode="expand", ncol=4)
axs[4].legend(loc='lower center', bbox_to_anchor=(0.3, -2),fancybox=False, shadow=False, ncol=4)
plt.show()
fig_profiles.savefig(save_path+flag_save+'_total_xy_line_profile_new.png',dpi=300, bbox_inches='tight')
#Quantifications
I_improvement_factor=(results_max_corr/results_max_raw)
av_I_lateral_factor=np.average(I_improvement_factor)
std_I_lateral_factor=np.std(I_improvement_factor)
table_results=tabulate([['av_I_Improvement factor ',av_I_lateral_factor ],['std_I_Improvement factor ',std_I_lateral_factor]],headers=['Correction improvement'])

f = open(save_path+'Corrected-improvement-xy.txt', 'w')
f.write(table_results)
f.close()
# ----------------------plot intensities----------------------------------------
# fig_int =plt.figure('intensities', figsize=(15, 30))
# x_p = np.arange(0,len(folders), 1)
# plt.plot(x_p, results_max_raw, 'ro', label = 'aberrated')
# plt.plot(x_p, results_max_corr, 'bo',label = 'corrected')
# plt.xlabel(r'$ROI$'), plt.ylabel(r'$ Normalized \quad intensity \quad (arb. unit)$')
# plt.xticks(x_p, x_p+1)
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(loc='center right')
# plt.show()
# fig_int.savefig(save_path+'intensities'+str(folders[i])+ '.png',dpi=300, bbox_inches='tight')


#----------------------------Plot methrics--------------------------------------
# --------------------Functions to fit metrics------------------------------------------
# def lor(x, amp1, cen1, wid1,b):
#     return (amp1*wid1**2/((x-cen1)**2+wid1**2))+b*x
# def gauss(x, amp1,cen1,sigma1,b):
#     return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))+b*x
# #------------------------------------------------------------------------------
# zernikes=21
# start=3
# weights=np.linspace(-3,3,15)
# n_points=9
# error_weights=np.zeros((n_points,zernikes-start), dtype=float)
# for j in range (n_points):
#     print(j)
#     ints =saved_ints[:,:,j]
#     area = saved_area[:,:,j]
#     com = ints*area
#
#     for i in range(zernikes-start):
#         if i == 1:
#             continue
#
#         print('zerny', str(i+3), 'point',str(j))
#         #=============================================================================
#         #----------------------------- intensity fit---------------------
#         xfit2 = np.linspace(weights[0],weights[-1],1000)
#         meani = sum(weights * ints[i,:]) / sum(ints[i,:])
#         sigmai = np.sqrt(sum(ints[i,:] * (weights - meani)**2) / sum(ints[i,:]))
#         parsi, covi = curve_fit(f=lor,xdata=weights,ydata=ints[i,:],p0=[max(ints[i,:]), meani, sigmai,1])
#      #=============================================================================
#      #----------------------------- combined fit---------------------
#         meanc = sum(weights* com[i,:]) / sum(com[i,:])
#         sigmac = np.sqrt(sum(com[i,:]* (weights - meanc)**2) / sum(com[i,:]))
#         parsc, covc = curve_fit(f=lor,xdata=weights,ydata=com[i,:],p0=[max(com[i,:]), meanc, sigmac,1])
#         fit_error=np.sqrt(np.diag(covc))
#         error_weights[j,i]=fit_error[1]
#     # =============================================================================
#      #----------------------------- Plots---------------------
#         fig=plt.figure('Point'+str(int(j))+'Zernike' + str(i+3), figsize = (5,5),linewidth=2.0)
#         plt.plot(weights,ints[i,:]/np.amax(ints[-1]),'og', label = 'ints')
#         # plt.plot(weights,ints[i],'og', label = 'ints')
#         plt.plot(xfit2,lor(xfit2,*parsi.tolist())/np.max(ints[-1]),'--g')
#         plt.plot(weights,area[i,:]/np.max(area[-1]),'or',label = 'area')
#         # plt.plot(weights,area[i]/np.max(area[-1]),'or',label = 'area')
#         plt.plot(weights,com[i,:]/np.max(com[-1]), 'ob', label='comb')
#         plt.plot(xfit2,lor(xfit2,*parsc.tolist())/np.max(com[-1]),'--b')
#
#         plt.ylabel(r'$ norm. metric \quad (arb.units)$')
#         plt.xlabel(r'$ Weights \quad (\mu m)$')
#         plt.title(r'$ Point=\quad$'+str(int(j))+r'$ \quad Zernike$'+ str(i+3))#str(i+3))
#         plt.grid(True)
#         plt.legend()
#         # plt.show()
#         fig.savefig(save_path+'metric_p_'+str(int(j))+'zerny'+str(i+3)+'.png',dpi=300, bbox_inches='tight')
# print(error_weights, error_weights.shape)
# np.save(semi_path_metrics+'error_weights.npy', error_weights )

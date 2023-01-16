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
MEDIUM_SIZE = 30
BIGGER_SIZE = 40

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#-------------------------Bar plot best weights---------------------------------

# semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\23_02\\fep\\'
semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaprive_optics\\test_gui_AO\\23_02\\fish_side\\'
# path_images=semi_path+'comparison_raw_fep_corr_total_corr\\'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_S4\\'

# # #------------------------------Bar plot, figure supplementary 5 panel f  ----------------------------------------
# # best_weights=pickle.load(open(semi_path+date_aniso+'\\'+type_aniso+'\\_best_weights.p','rb'))
# # best_weights=pickle.load(open(semi_path+'20210223-160846_aniso\\'+'_best_weights.p','rb'))
best_weights=pickle.load(open(semi_path+'20210223-180909_aniso\\'+'_best_weights.p','rb'))
#
#
#
# # ----- changes zernike order from osa to noll indexing
# # noll_order= np.array([1,2,3,4,5,6,7,8,9,10,11])-1
# # osa_order = np.array([0,2,1,4,3,5,7,8,6,9,12])
# # noll_best_weights = np.zeros((16,11))
# # for nll in noll_order:
# #     n_osa = osa_order[nll]
# #     noll_best_weights[:,nll] = aniso_best_weights[:,n_osa]
# # # -----
color_i=['blue', 'green', 'red']
offset=[0,0.2,0.4]
for i in range(best_weights.shape[0]) :
    print(i)

    best_weights_i =best_weights[i,:]#  for OSA


    fig_bar=plt.figure('barplot_best_weights_',figsize = (25,4),linewidth=2.0)


#     # bar plot best_weights
    # bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12')# osa order
    # bars = ('5','6', '7', '8', '9', '10','11')# noll order
    # bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12','13','14','15','16',
    #     '17','18','19','20','21','22','23', '24','25', '26', '27', '28', '29','30',
    #  '31', '32','33','34','35','36','37','38','39','40','41','42','43', '44')
#
#     # color=['green', 'orange', 'purple', 'red', 'lightskyblue', 'black', 'yellow','grey','magenta']

    y_aniso = np.copy(best_weights_i[3:])



    # y_base[:8]=0 #to plot baseline only on spherical
    y_pos = np.arange(len(bars))
    N = len(y_aniso)
    x = (np.arange(0, N, 1)-offset[i])
    width = 0.2

    # ax1=fig_bar.add_subplot(1,1,1)
    j=i+1
    plt.bar(x, y_aniso, width,color= color_i[i],align='edge',label='ROI '+str(j) )
    plt.xticks(y_pos, bars)


    # plt.bar(x, y, width,color= 'green',label='dynamic ROI ' )
    plt.ylabel(r'$Weights \quad (μm)$')
    plt.xlabel(r'$Zernike \quad index$')
    # # plt.xticks(y_pos+1, bars)
    plt.axhline(0, color='black',ls='--')
    # plt.ylim(-3, 3)
    # plt.title(r'$ Corrections \quad point \quad$'+str(canopy_order.index(i)+1))
    plt.legend(loc='lower right',  fontsize=SMALL_SIZE)
    plt.grid(True)

plt.show()
fig_bar.savefig(save_path+'bar_plot.png',dpi=300, bbox_inches='tight')

# # # ##-----------------------------------Total images, figure S5 panel a-------------------------------
# # ###Corrected
total_im_corr= np.array(Image.open(semi_path+'after_correction.tiff'))
#max and min corrected raw
min_total_corr=np.amin(total_im_corr)
max_total_corr=np.amax(total_im_corr)
print('max/min corr total',min_total_corr, max_total_corr)

# --Normalization
total_im_corr_norm=(total_im_corr)/(max_total_corr)
# total_im_corr_norm_plot=(total_im_corr_norm)
total_im_corr_norm_plot=total_im_corr_norm[800:1000,650:1450]
#max and min corrected normalized
min_total_im_corr_norm=np.amin(total_im_corr_norm_plot)
max_total_im_corr_norm=np.amax(total_im_corr_norm_plot)
print(' min/max corr total norm', min_total_im_corr_norm, max_total_im_corr_norm)

####Raw
total_im_raw= np.array(Image.open(semi_path+'before_correction.tiff'))
#max and min corrected raw
min_total_raw=np.amin(total_im_raw)
max_total_raw=np.amax(total_im_raw)
print('max/min raw total',min_total_raw, max_total_raw)

# --Normalization
total_im_raw_norm=(total_im_raw)/(max_total_corr)
# total_im_raw_norm_plot=(total_im_raw_norm)
total_im_raw_norm_plot=total_im_raw_norm[800:1000,650:1450]
#max and min corrected normalized
# # im_corr_norm_plot=im_corr_norm[70:120,70:120]
min_total_im_raw_norm=np.amin(total_im_raw_norm_plot)
max_total_im_raw_norm=np.amax(total_im_raw_norm_plot)
print(' min/max raw total norm', min_total_im_raw_norm, max_total_im_raw_norm)
#
#figure aberrated image
fig_1=plt.figure('Aberrated image', figsize=(10, 4))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_raw=plt.imshow(total_im_raw_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_raw, ticks=[0, 1], orientation="vertical", fraction=0.012, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_1.savefig(save_path+'im_total_raw_norm.png',dpi=300, bbox_inches='tight')


#
#figure corrected image
fig_2=plt.figure('Corr image', figsize=(10, 4))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_corr=plt.imshow(total_im_corr_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_corr, ticks=[0, 1], orientation="vertical", fraction=0.012, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_2.savefig(save_path+'im_total_corr_norm.png',dpi=300, bbox_inches='tight')

# brightfield image plot
bf_image=np.array(Image.open(semi_path+'cgh_0.0.tif'))
bf_image_plot=bf_image[150:250,80:420]
fig_3=plt.figure('Corr image', figsize=(30, 15))
plt.imshow(bf_image_plot)
plt.axis('off')
plt.show()
fig_3.savefig(save_path+'bf_image.png',dpi=300, bbox_inches='tight')
#
# # # #----------------------------------Roi xy planes -------------------------------
# # sub_folder='data_analysis\\gauss_fit\\0000\\'
sub_folder='data_analysis\\gauss_fit_cgh\\'
folders=os.listdir(semi_path+sub_folder)
# results_FWHM_fit_corr=np.zeros((len(folders), 633), dtype=float)
# results_FWHM_fit_raw=np.zeros((len(folders), 633), dtype=float)
results_FWHM_fit_corr=[[],[],[]]
results_FWHM_fit_raw=[[],[],[]]
n_points_fit=2000
results_fit_corr=np.zeros((len(folders), n_points_fit), dtype=float)
results_fit_raw=np.zeros((len(folders), n_points_fit), dtype=float)
results_raw=np.zeros((len(folders), 40), dtype=float)
results_corr=np.zeros((len(folders), 40), dtype=float)
results_max_corr=np.zeros((len(folders), 1), dtype=float)
results_max_raw=np.zeros((len(folders), 1), dtype=float)

results_max_corr_fit=np.zeros((len(folders), 1), dtype=float)
results_max_raw_fit=np.zeros((len(folders), 1), dtype=float)
results_i_corr=np.zeros((len(folders), 1), dtype=float)
results_i_raw=np.zeros((len(folders), 1), dtype=float)
for i in range (len(folders)):

# rois_files=fnmatch.filter(os.listdir('M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\02_02\\beads_0.1_um\\set_2\\stack_20210202-175600\\4analysis\\'), '*.tif')
    data_path=semi_path+sub_folder+folders[i]+'\\'
    files = [['original.tif','corrected.tif']]

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
#
#     #---------------------Plot xy focus, figure S5 panel c---------------------------------------------
    fig4, (ax1, ax2)= plt.subplots(1, 2,figsize = (5,5),linewidth=2.0)
    plot_raw=ax1.imshow(im_raw_norm_plot, vmin=0, vmax=max_corr_norm, cmap = 'jet')
    cbar_ax1 = plt.colorbar(plot_raw,ax=ax1,ticks=[0, max_corr_norm], orientation="horizontal", fraction=0.047, pad=0.01)
    cbar_ax1.set_ticklabels([r'$0$', r'$1$'])
    cbar_ax1.ax.tick_params(labelsize=15)
    ax1.hlines(y=index_im_raw_norm[0], xmin=index_im_raw_norm[1]-radius_roi, xmax=index_im_raw_norm[1]+radius_roi, colors='black', linestyles='dashed', lw=1)
    ax1.axis('off')
    scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
    ax1.add_artist(scalebar)

    plot_corr=ax2.imshow(im_corr_norm_plot, vmin=0, vmax=1, cmap = 'jet')
    cbar_ax2 = plt.colorbar(plot_corr,ax=ax2,ticks=[0, max_corr_norm], orientation="horizontal", fraction=0.047, pad=0.01)
    cbar_ax2.ax.tick_params(labelsize=15)
    cbar_ax2.set_ticklabels([r'$0$', r'$1$'])
    # ax2.hlines(y=54, xmin=54-20, xmax=54+20, colors='gold', linestyles='dashed', lw=1)
    ax2.hlines(y=index_im_corr_norm[0], xmin=index_im_corr_norm[1]-radius_roi, xmax=index_im_corr_norm[1]+radius_roi, colors='black', linestyles='dashed', lw=1)
    ax2.axis('off')
    scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
    ax2.add_artist(scalebar)
    plt.show()
    fig4.savefig(save_path+'xy_image'+str(folders[i])+'.png',dpi=300, bbox_inches='tight')
#
#     #
#     # #--------------------------------Line profile ----------------------------------
#     #
#     # #-------------------------------------------------------------------------------
#     # --- Fit line prifile  + FWHM calculation -------------------------------------
    def lor(x, amp1, cen1, wid1,b,off):
        return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + b*x + off
    def gauss(x, amp1,cen1,sigma1,b):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + b*x

    #--- weights, saved_ints, and xfit2 have to be altered

    # x=np.linspace(0,40,40)*0.195
    x=np.linspace(-20,20,40)*0.195

    meancorr = sum(x * im_corr_norm_plot[:,index_im_corr_norm[0]]) / sum(im_corr_norm_plot[:,index_im_corr_norm[0]])
    sigmacorr = np.sqrt(sum(im_corr_norm_plot[:,index_im_corr_norm[0]] * (x - meancorr)**2) / sum(im_corr_norm_plot[:,index_im_corr_norm[0]]))

    parscorr, covcorr = curve_fit(f=lor,xdata=x,ydata=im_corr_norm_plot[:,index_im_corr_norm[0]],p0=[max(im_corr_norm_plot[:,index_im_corr_norm[0]]), meancorr, sigmacorr,1,0])


    meanraw = sum(x * im_raw_norm_plot[:,index_im_raw_norm[0]]) / sum(im_raw_norm_plot[:,index_im_raw_norm[0]])
    sigmaraw = np.sqrt(sum(im_raw_norm_plot[:,index_im_raw_norm[0]] * (x - meanraw)**2) / sum(im_raw_norm_plot[:,index_im_raw_norm[0]]))

    parsraw, covraw = curve_fit(f=lor,xdata=x,ydata=im_raw_norm_plot[:,index_im_raw_norm[0]],p0=[max(im_raw_norm_plot[:,index_im_raw_norm[0]]), meanraw, sigmaraw,1,0])

      #----fit along y
    # meancorr = sum(x * im_corr_norm_plot[index_im_corr_norm[0],:]) / sum(im_corr_norm_plot[index_im_corr_norm[0],:])
    # sigmacorr = np.sqrt(sum(im_corr_norm_plot[index_im_corr_norm[0],:] * (x - meancorr)**2) / sum(im_corr_norm_plot[index_im_corr_norm[0],:]))
    # parscorr, covcorr = curve_fit(f=lor,xdata=x,ydata=im_corr_norm_plot[index_im_corr_norm[0],:],p0=[max(im_corr_norm_plot[index_im_corr_norm[0],:]), meancorr, sigmacorr,1,0])
    # meanraw = sum(x * im_raw_norm_plot[index_im_raw_norm[0],:]) / sum(im_raw_norm_plot[index_im_raw_norm[0],:])
    # sigmaraw = np.sqrt(sum(im_raw_norm_plot[index_im_raw_norm[0],:] * (x - meanraw)**2) / sum(im_raw_norm_plot[index_im_raw_norm[0],:]))
    #
    # parsraw, covraw = curve_fit(f=lor,xdata=x,ydata=im_raw_norm_plot[index_im_raw_norm[0],:],p0=[max(im_raw_norm_plot[index_im_raw_norm[0],:]), meanraw, sigmaraw,1,0])

    # xfit = np.linspace(0,40,2000)*0.195
    xfit = np.linspace(-20,20,n_points_fit)*0.195

    fit_raw = lor(xfit,*parsraw.tolist())
    fit_corr = lor(xfit,*parscorr.tolist())
    # FWHM_fit_raw = xfit[fit_raw >= 0.5*np.amax(fit_raw)]
    # FWHM_fit_corr = xfit[fit_corr >= 0.5*np.amax(fit_corr)]
    FWHM_fit_raw = xfit[fit_raw >= 0.5*(np.amax(fit_raw)-np.amin(fit_raw))]
    FWHM_fit_corr = xfit[fit_corr >= 0.5*(np.amax(fit_corr)-np.amin(fit_raw))]
    # save results for each point in global arrays.
    results_fit_corr[i, :]=fit_corr
    results_fit_raw[i, :]=fit_raw
    results_raw[i, :]=im_raw_norm_plot[:, index_im_raw_norm[0]]
    results_corr[i, :]=im_corr_norm_plot[:,index_im_corr_norm[0]]
    #along y
    # results_raw[i, :]=im_raw_norm_plot[ index_im_raw_norm[0],:]
    # results_corr[i, :]=im_corr_norm_plot[index_im_corr_norm[0],:]
    results_FWHM_fit_corr[i].append(FWHM_fit_corr)
    results_FWHM_fit_raw[i].append(FWHM_fit_raw)
    results_max_corr[i, :]=max_corr_norm
    results_max_raw[i, :]=max_raw_norm


    results_i_corr[i, :]=np.sum(im_corr)
    results_i_raw[i, :]=np.sum(im_raw)

    results_max_corr_fit[i, :]=np.amax(fit_corr)
    results_max_raw_fit[i, :]=np.amax(fit_raw)
#     #----------------------Figure panel g single lines----------------------------------------
#     # fig5=plt.figure('line profile aniso fit FWHM p0 aniso vs p0 raw',figsize = (11,11),linewidth=2.0)
#     # plt.plot(xfit,fit_corr,'-', color='darkblue',label = 'correction fit')
#     # plt.plot(xfit,fit_raw,'-', color='darkorange',label = 'aberration fit')
#     # plt.plot(x,im_raw_norm_plot[index_im_raw_norm[0], :], 'o', color='darkorange',label='aberration')
#     # plt.plot(x,im_corr_norm_plot[index_im_corr_norm[0],:], 'o', color='darkblue',label='correction')
#     #
#     # # plt.plot(x,im_corr_norm_plot[index_im_corr_norm[1]-radius_roi:index_im_corr_norm[1]+radius_roi,:], '.', color='darkblue',label='correction')
#     # # plt.plot(FWHM_fit_raw,0.5*((np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw))*np.ones(np.shape(FWHM_fit_raw)),color='red',linestyle='dashed',label = 'FWHM=' + str(round(FWHM_fit_raw[-1]-FWHM_fit_raw[0],3)) + '$\mu m$')
#     # # plt.plot(FWHM_fit_corr,0.5*((np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr))*np.ones(np.shape(FWHM_fit_corr)),color='blue',linestyle='dashed',label = 'FWHM=' + str(round(FWHM_fit_corr[-1]-FWHM_fit_corr[0],3)) + r'$\mu m$')
#     # plt.plot(FWHM_fit_raw,0.5*(np.amax(fit_raw)-np.amin(fit_raw))*np.ones(np.shape(FWHM_fit_raw)),color='darkorange',linestyle='dashed', label = 'FWHM=' + str(round(FWHM_fit_raw[-1]-FWHM_fit_raw[0],3)) + '$\mu m$')
#     # plt.plot(FWHM_fit_corr,0.5*(np.amax(fit_corr)-np.amin(fit_corr))*np.ones(np.shape(FWHM_fit_corr)),color='darkblue',linestyle='dashed',label = 'FWHM=' + str(round(FWHM_fit_corr[-1]-FWHM_fit_corr[0],3)) + r'$\mu m$')
#     # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
#     # plt.ylabel(r'$Normalized \quad intensity $')
#     # plt.legend(loc='upper left')
#     # # plt.title('aniso fit FWHM corr vs raw')
#     # plt.show()
#     # fig5.savefig(save_path+'xy_line_profile'+str(folders[i])+ '.png',dpi=300, bbox_inches='tight')
# fig_profiles, axs = plt.subplots(1,3,figsize=(15, 5))
# fig_profiles.subplots_adjust(hspace = .3, wspace=.3)
# axs = axs.ravel()
#---------------Figure S5 panel d
for  i in range(results_corr.shape[0]):


    axs[i].plot(xfit,results_fit_raw[i, :],'-', color='darkorange',label = 'aberration fit')
    axs[i].plot(xfit,results_fit_corr[i,:],'-', color='darkblue',label = 'correction fit')

    axs[i].plot(x,results_raw[i, :],'o', color='darkorange',label = 'aberration')
    axs[i].plot(x,results_corr[i,:],'o', color='darkblue',label = 'correction')

    # axs[i].plot(results_FWHM_fit_raw[i][0],0.5*(np.amax(results_fit_raw[i])-np.amin(results_fit_raw[i]))*np.ones(np.shape(results_FWHM_fit_raw[i][0])),color='darkorange',linestyle='dashed')
    # axs[i].plot(results_FWHM_fit_corr[i][0],0.5*(np.amax(results_fit_corr[i])-np.amin(results_fit_corr[i]))*np.ones(np.shape(results_FWHM_fit_corr[i][0])),color='darkblue',linestyle='dashed')
    axs[i].set_xticks(range(0, int(np.max(x))+1,1))
    # axs[j_index].title.set_text(str(j))

    # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
    # plt.ylabel(r'$Normalized \quad intensity $')

    # axs[j].legend(loc='upper left')
    axs[i].grid()
    axs[i].set_xticks(range(-4, int(np.max(x)+2),2))
    axs[i].set_yticks(np.linspace(0,1,3))
for ax in axs.flat:
    ax.label_outer()
fig_profiles.text(0.5, -0.05, r'$ Lateral \quad distance \quad (\mu m)$', ha='center', fontsize=MEDIUM_SIZE)
fig_profiles.text(0.04, 0.5, r'$Normalized \quad intensity $', va='center', rotation='vertical',fontsize=MEDIUM_SIZE)
# plt.legend(loc='lower right', mode="expand", ncol=4)
axs[2].legend(loc='lower center', bbox_to_anchor=(0.3, -0.6),fancybox=False, shadow=False, ncol=4, fontsize=MEDIUM_SIZE)
plt.show()
fig_profiles.savefig(save_path+'total_xy_line_profile_int.png',dpi=300, bbox_inches='tight')
# #
#
# #----------------------plot intensities--------------------------------------
improvement=(results_max_corr-results_max_raw)*100/results_max_corr
I_improvement_factor=(results_max_corr/results_max_raw)
av_I_lateral_factor=np.average(I_improvement_factor)
std_I_lateral_factor=np.std(I_improvement_factor)
table_results=tabulate([['Improvement percentage ',improvement ], ['Improvement factor', I_improvement_factor], ['av_I_lateral_factor',av_I_lateral_factor],['std improvement factor', std_I_lateral_factor]],\
                                  headers=['Correction improvement'])

f = open(save_path+'Corrected-improvement.txt', 'w')
f.write(table_results)
f.close()

# x_p = np.arange(0,len(folders), 1)
# # plt.plot(x_p, results_max_raw,'o', color='darkorange',  label = 'raw')
# # plt.plot(x_p, results_max_corr,'o', color='darkblue',label = 'corrected')
# plt.plot(x_p, results_max_raw_fit,'o', color='darkorange',  label = 'raw')
# plt.plot(x_p, results_max_corr_fit,'o', color='darkblue',label = 'corrected')
# plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE),
# plt.ylabel(r'$ I \quad  max. \quad norm.$',fontsize= SMALL_SIZE)
# plt.title('Lateral',fontsize= SMALL_SIZE)
# plt.xticks(x_p, x_p+1)
# plt.ylim([0,1.2])
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
# plt.show()
# fig_int_max.savefig(save_path+'lateral_intensities_norm.png',dpi=300, bbox_inches='tight')
#
# fig_int =plt.figure('intensities', figsize=(10, 3), linewidth=2.0)
# x_p = np.arange(0,len(folders), 1)
# plt.plot(x_p, results_i_raw,'o',color='darkorange',  label = 'raw')
# plt.plot(x_p, results_i_corr, 'o', color='darkblue',label = 'corrected')
# plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE)
# plt.ylabel(r'$ I \quad (arb. unit)$',fontsize= SMALL_SIZE)
# plt.xticks(x_p, x_p+1)
# plt.title('Lateral',fontsize= SMALL_SIZE)
# plt.ylim([0,np.amax(results_i_corr)+100000])
# # plt.yticks(np.linspace(0,9,4)*100000)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
# plt.show()
# fig_int.savefig(save_path+'lateral_intensities.png',dpi=300, bbox_inches='tight')
#-------------------------------------------------------------------------------
# #----------------------plot fidelity ----------------------------------------
#===================== reference coords: SLM  coords
# coords_ref_temp=np.load(semi_path+"Final_coords_holoName.npy")
# coords_ref_temp=np.transpose(coords_ref_temp)
# n_points=coords_ref_temp.shape[1]
#
# coords_ref=np.zeros((3,n_points))
#
#
# coords_ref=coords_ref_temp
# #=================corr_coords : coordinates adter correction.
# # coords from image
# #-----system parameters
# cam_h=2048
# off_y=cam_h/2
# off_x=cam_h/2
# pix_size=6.5
# M_2=9./300
# n_points=coords_ref.shape[1]
# defocus_voltage=-0.0153346  *coords_ref[2,0]+ 2.65583
# coords_corr_im=np.load(semi_path+"centers_corr_cgh_bis.npy")
# coords_raw_im=np.load(semi_path+"centers_raw_cgh_bis.npy")
#
# # coords_corr_im_bis=np.load(semi_path+"centers_corr.npy")
# # coords_raw_im_bis=np.load(semi_path+"centers_raw.npy")
# centers=np.zeros((2,n_points))
# #swap for carthesian coordinates
# centers[0,:]=coords_corr_im[:,1]
# centers[1,:]=coords_corr_im[:,0]
# # # transformation in samples
# coords_corr_sample=np.ones((4,n_points), dtype=np.float64)
# xc_im=centers[0,:]-off_x
# yc_im=off_y-centers[1,:]
# coords_corr_sample[0,:]=xc_im*M_2*pix_size
# coords_corr_sample[1,:]=yc_im*M_2*pix_size
# coords_corr_sample[2,:]= 136.763 *(0.596613*defocus_voltage-1.47153)
# # #transformation to SLM
# camera_to_slm_matrix=np.load('M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\23_02\\t_aff_rand_bigrange_20_fov130\\T_affine.npy')
# coords_corr_slm=np.zeros((3,n_points))
# coords_corr_slm=np.dot(camera_to_slm_matrix,coords_corr_sample)
#
# coords_flip=np.copy(coords_corr_slm)
# coords_flip[1,:]=coords_corr_slm[1,:]*(-1.0) # flipy
#
#
# #Apply rot matrix -90
# coords_rot=np.copy(coords_corr_slm)
# rot_matrix=np.zeros((2,2))
# rot_matrix[0,1]=1
# rot_matrix[1,0]=-1
# coords_rot[:2,:]=np.dot(rot_matrix,coords_flip[:2,:])
#
# coords_corr=coords_rot
# # coords_corr=coords_corr_slm
#
# #calculation fidelity
# dst=np.zeros((n_points),dtype=np.float64)
# dst[:] =np.sqrt((coords_ref[0,:]-coords_corr[0,:])**2+(coords_ref[1,:]-\
#             coords_corr[1,:])**2+(coords_ref[2,:]-coords_corr[2,:])**2)
#
# average_dst=np.average(dst)
# sigma_dst=np.std(dst)
# #-------------------------------FIDELITY RAW------------------------------------
# centers_raw=np.zeros((2,n_points))
# #swap for carthesian coordinates
# centers_raw[0,:]=coords_raw_im[:,1]
# centers_raw[1,:]=coords_raw_im[:,0]
# # # transformation in samples
# coords_corr_sample_raw=np.ones((4,n_points), dtype=np.float64)
# xc_im_raw=centers_raw[0,:]-off_x
# yc_im_raw=off_y-centers_raw[1,:]
# coords_corr_sample_raw[0,:]=xc_im_raw*M_2*pix_size
# coords_corr_sample_raw[1,:]=yc_im_raw*M_2*pix_size
# coords_corr_sample_raw[2,:]= 136.763 *(0.596613*defocus_voltage-1.47153)
#
# coords_corr_slm_raw=np.zeros((3,n_points))
# coords_corr_slm_raw=np.dot(camera_to_slm_matrix,coords_corr_sample_raw)
#
# coords_flip_raw=np.copy(coords_corr_slm_raw)
# coords_flip_raw[1,:]=coords_corr_slm_raw[1,:]*(-1.0) # flipy
#
#
# #Apply rot matrix -90
# coords_rot_raw=np.copy(coords_corr_slm_raw)
# rot_matrix=np.zeros((2,2))
# rot_matrix[0,1]=1
# rot_matrix[1,0]=-1
# coords_rot_raw[:2,:]=np.dot(rot_matrix,coords_flip_raw[:2,:])
#
# coords_corr_raw=coords_rot_raw
# # coords_corr=coords_corr_slm
#
# #calculation fidelity
# dst_raw=np.zeros((n_points),dtype=np.float64)
# dst_raw[:] =np.sqrt((coords_ref[0,:]-coords_corr_raw[0,:])**2+(coords_ref[1,:]-\
#             coords_corr_raw[1,:])**2+(coords_ref[2,:]-coords_corr_raw[2,:])**2)
# # np.save(semi_path+'dst_raw',dst)
# dst_raw=np.load(semi_path+'dst_raw.npy')
# average_dst_raw=np.average(dst_raw)
# sigma_dst_raw=np.std(dst_raw)
#
#
# dst_corr_raw=np.zeros((n_points),dtype=np.float64)
# dst_corr_raw[:] =np.sqrt((coords_raw_im[:,0]-coords_corr_im[:,0])**2+(coords_raw_im[:,1]-\
#             coords_corr_im[:,1])**2)*0.195
#
# average_dst_corr_raw=np.average(dst_corr_raw)
# sigma_dst_corr_raw=np.std(dst_corr_raw)
#
# #
# fig3 = plt.figure(figsize=(10,10), linewidth=2.0)
# x= np.linspace(1,n_points, n_points)
# x_ticks=[]
# x_ticks=x
# plt.plot(x,dst,'o', color='darkblue')
# plt.axhline(y=average_dst,color='darkblue', alpha=.5, linestyle='--')
# plt.xticks(x_ticks)
# plt.xlabel(r'$ ROIs $',labelpad=10)
# plt.ylabel(r'$ Fidelity\quad (\mu m)$')
# plt.ylim([0,1.5])
# plt.show()
# fig3.savefig(save_path+'fidelity.png',dpi=300, bbox_inches='tight')
# # #
# fig3bis = plt.figure(figsize=(10,10), linewidth=2.0)
# x= np.linspace(1,n_points, n_points)
# x_ticks=[]
# x_ticks=x
# plt.plot(x,dst_raw,'o', color='darkblue')
# plt.axhline(y=average_dst_raw,color='darkblue', alpha=.5, linestyle='--')
# plt.xticks(x_ticks)
# plt.xlabel(r'$ ROIs $',labelpad=10)
# plt.ylabel(r'$ Fidelity\quad (\mu m)$')
# plt.ylim([0,1.5])
# plt.show()
# fig3bis.savefig(save_path+'fidelity_raw.png',dpi=300, bbox_inches='tight')
# # #plot comparison fidelity
# fig4 = plt.figure(figsize=(10,10), linewidth=2.0)
# x= np.linspace(1,n_points, n_points)
# x_ticks=[]
# x_ticks=x
# plt.plot(x,dst,'o', color='darkblue', label= 'corrected')
# plt.axhline(y=average_dst, color='darkblue', alpha=.5, linestyle='--')
# plt.plot(x,dst_raw,'o',color='darkorange',label= 'raw')
# plt.axhline(y=average_dst_raw,color='darkorange', alpha=.5, linestyle='--')
# plt.xticks(x_ticks)
# plt.xlabel(r'$ ROIs $')
# plt.ylabel(r'$ Fidelity\quad (\mu m)$')
# plt.ylim([0,1.5])
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2)
# plt.show()
# fig4.savefig(save_path+'fidelity_comparison.png',dpi=300, bbox_inches='tight')


#
# fig5 = plt.figure(figsize=(10,10), linewidth=2.0)
# x= np.linspace(1,n_points, n_points)
# x_ticks=[]
# x_ticks=x
# plt.axhline(y=average_dst_corr_raw, color='darkblue', alpha=.5, linestyle='--')
# plt.plot(x,dst_corr_raw,'o',color='darkblue')
# plt.xticks(x_ticks)
# plt.xlabel(r'$ ROIs $')
# plt.ylabel(r'$ Distance \quad (\mu m)$')
# plt.ylim([-0.5,1.5])
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2)
# plt.show()
# fig5.savefig(save_path+'distance.png',dpi=300, bbox_inches='tight')



# --- Metric improvement fep + fish correction, figure S5 panel e---------------

folder= 'M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\test_gui_AO\\23_02\\fish_side\\20210223-180909_aniso\\'
points = ['000','001','002']
metric_1_fepfish=np.zeros((len(points),41), dtype=np.float)
diameter_fepfish =np.zeros((len(points),41), dtype=np.float)
intensity_fepfish =np.zeros((len(points),41), dtype=np.float)
for j, point  in enumerate (points):
    # zernike = '14'
    nr = '07' # evaluate each image in 0 to account for previous zernike
    number = []
    for q in range(3,45):
        if q == 4:
            continue
        if q < 10:
            number += [str(q)]
        else:
            number += [str(q)]

    diameter_temp =np.zeros(np.shape(number))
    intensity_temp =np.zeros(np.shape(number))
    metric_1_temp=np.zeros(np.shape(number))

    i = 0
    for nn in number:


        image = np.array(Image.open(folder + 'im_interm' + point + '_' + nn + '_000' +nr+'.tif'))

        threshold = (np.amax(image)-np.amin(image))/np.exp(1)+np.amin(image)
        c_y,c_x = np.unravel_index(image.argmax(), image.shape)
        binary_image = np.ones(np.shape(image))
        binary_image[image<threshold] = 0
        # fig = plt.figure('check on diameter')
        # plt.imshow(binary_image)
        # plt.show
        diameter_temp[i] = 1/(np.max((np.sum(binary_image[c_y,:]),np.sum(binary_image[:,c_x]))))
        # intensity_temp[i] = np.amax(image) #max intensity
        # metric_1_temp[i]=intensity_temp[i]**2*diameter_temp[i] # metric M1 max
        intensity_temp[i] = np.sum(image)**2 # intensity metric exp
        metric_1_temp[i]=intensity_temp[i]*diameter_temp[i]# M_1 metric exp
        i += 1
    intensity_fepfish[j,:] = intensity_temp/np.amax(intensity_temp)
    diameter_fepfish[j,:]  = diameter_temp/np.amax(diameter_temp)
    metric_1_fepfish[j,:] = metric_1_temp/np.amax(metric_1_temp)
    x = np.linspace(3,44,41)

    # figs in for loop
#     fig_int=plt.figure('Intensity',  figsize=(15, 5))
#     plt.plot(x,intensity[j,:],'o-',color=color_i[j],label='ROI '+str(j+1))
#     plt.legend(loc='lower right',  fontsize=SMALL_SIZE)
#     plt.ylim([0,1.1])
#     # plt.xticks(np.linspace(3) )
#     plt.yticks(np.linspace(0,1,3))
#     plt.grid(True)
#
#     fig_d=plt.figure('Diameter',  figsize=(10, 5))
#     plt.plot(x,diameter[j,:],'*--', color=color_i[j],label='ROI '+str(j+1))
#     plt.legend(loc='lower right',  fontsize=SMALL_SIZE)
#     plt.ylim([0,1.1])
#     plt.yticks(np.linspace(0,1,3))
#     plt.ylabel(r'$Diameter  $',fontsize=MEDIUM_SIZE)
#     plt.xlabel(r'$Aberration \quad index $',fontsize=MEDIUM_SIZE)
#     plt.grid(True)
#
#
# plt.show()
# fig_d.savefig(save_path+'fep+fish_dimeter_improvement.png', dpi=300, bbox_inches='tight')
# fig_int.savefig(save_path+'fep+fish_intensity_improvement.png', dpi=300, bbox_inches='tight')

# fig_int_vs_index, axs = plt.subplots(2,1,figsize=(15, 7))
# fig_int_vs_index.subplots_adjust(hspace = .6, wspace=.3)
# axs = axs.ravel()
# x = np.linspace(3,44,41)
# for  j in range(len(points)):
#
#
#     # axs[0].plot(x,intensity_fep[j,:],'o-', color=color_i[j], label='ROI '+str(j+1))
#     axs[0].plot(x,intensity_fepfish[j,:],'o-', color=color_i[j],label='ROI '+str(j+1))
#
#
#     # axs[i].plot(results_FWHM_fit_raw[i][0],0.5*(np.amax(results_fit_raw[i])-np.amin(results_fit_raw[i]))*np.ones(np.shape(results_FWHM_fit_raw[i][0])),color='darkorange',linestyle='dashed')
#     # axs[i].plot(results_FWHM_fit_corr[i][0],0.5*(np.amax(results_fit_corr[i])-np.amin(results_fit_corr[i]))*np.ones(np.shape(results_FWHM_fit_corr[i][0])),color='darkblue',linestyle='dashed')
#
#     # axs[j_index].title.set_text(str(j))
#
#     # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
#     # plt.ylabel(r'$Normalized \quad intensity $')
#
#     # axs[j].legend(loc='upper left')
#     axs[0].grid()
#     axs[0].set_ylim([0,1.1])
#     axs[0].set_yticks(np.linspace(0,1,3))
#
#     # axs[1].grid()
#     # axs[1].set_ylim([0,1.1])
#     # axs[1].set_yticks(np.linspace(0,1,3))
# for ax in axs.flat:
#     ax.label_outer()
# fig_int_vs_index.text(0.5, 0.00, r'$Aberration \quad index $', ha='center', fontsize=MEDIUM_SIZE)
# fig_int_vs_index.text(0.04, 0.5, r'$Normalized \quad I^2 $', va='center', rotation='vertical',fontsize=MEDIUM_SIZE)
# # plt.legend(loc='lower right', mode="expand", ncol=4)
# axs[1].legend(loc='lower center', bbox_to_anchor=(0.3, -1),fancybox=False, shadow=False, ncol=4, fontsize=MEDIUM_SIZE)
# plt.show()
# fig_int_vs_index.savefig(save_path+'improvement_int_exp_total.png',dpi=300, bbox_inches='tight')


bars = ('3','4','5', '6', '7', '8', '9','10', '11', '12','13','14','15','16',\
'17','18','19','20','21','22','23', '24','25', '26', '27', '28', '29','30',\
'31', '32','33','34','35','36','37','38','39','40','41','42','43', '44')
fig_m1_vs_index = plt.figure(figsize=(25, 4),linewidth=2)
y_pos = np.arange(len(bars))
x = np.linspace(3,44,41)
for  j in range(len(points)):

    # axs[0].plot(x,metric_1_fep[j,:],'o-', color=color_i[j], label='ROI '+str(j))
    plt.plot(x,metric_1_fepfish[j,:],'o-', color=color_i[j],label='ROI '+str(j+1))

    plt.grid()
    plt.ylim([0,1.1])
    plt.yticks(np.linspace(0,1,3))
    plt.xticks(y_pos+3,bars)
    plt.xlabel(r'$Zernike \quad index $')
    plt.ylabel( r'$Normalized \quad M_1 $')
    plt.legend(loc='lower right')
plt.show()
fig_m1_vs_index.savefig(save_path+'improvement_metric_exp_total.png',dpi=300, bbox_inches='tight')

#===================================plot metric as function of measurement================================================



# ---fep + fish correction

# folder= 'M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\23_02\\fish_side\\20210223-180909_aniso\\'
# points = ['000','001','002']
# n_weights=15
# n_zernike=41
# n_meas=n_zernike*n_weights
# metric_1_fepfish=np.zeros((len(points),n_meas), dtype=np.float)
# diameter_fepfish =np.zeros((len(points),n_meas), dtype=np.float)
# intensity_fepfish =np.zeros((len(points),n_meas), dtype=np.float)
# for j, point  in enumerate (points):
#     # zernike = '14'
#     # nr = '07' # evaluate each image in 0 to account for previous zernike
#     number = []
#     for q in range(3,45):
#         if q == 4:
#             continue
#         if q < 10:
#             number += [str(q)]
#         else:
#             number += [str(q)]
#
#     diameter_temp =np.zeros(np.shape(number*n_weights))
#     intensity_temp =np.zeros(np.shape(number*n_weights))
#     metric_1_temp=np.zeros(np.shape(number*n_weights))
#     i = 0
#     for nn in number:
#         # evaluate each image in 0 to account for previous zernike
#         for nr in range (n_weights):
#             nr_string=str(nr)
#             if nr < 10:
#                 nr_string='0'+str(nr)
#             image = np.array(Image.open(folder + 'im_interm' + point + '_' + nn + '_000' +nr_string+'.tif'))
#
#             threshold = (np.amax(image)-np.amin(image))/np.exp(1)+np.amin(image)
#             c_y,c_x = np.unravel_index(image.argmax(), image.shape)
#             binary_image = np.ones(np.shape(image))
#             binary_image[image<threshold] = 0
#             # fig = plt.figure('check on diameter')
#             # plt.imshow(binary_image)
#             # plt.show
#             diameter_temp[i] = 1/(np.max((np.sum(binary_image[c_y,:]),np.sum(binary_image[:,c_x]))))
#             # intensity_temp[i] = np.amax(image) #max intensity
#             # metric_1_temp[i]=intensity_temp[i]**2*diameter_temp[i] # metric M1 max
#             intensity_temp[i] = np.sum(image)**2 # intensity metric exp
#             metric_1_temp[i]=intensity_temp[i]*diameter_temp[i]# M_1 metric exp
#             i += 1
#     intensity_fepfish[j,:] = intensity_temp/np.amax(intensity_temp)
#     diameter_fepfish[j,:]  = diameter_temp/np.amax(diameter_temp)
#     metric_1_fepfish[j,:] = metric_1_temp/np.amax(metric_1_temp)
#
#
#

# fig_m1_vs_meas= plt.figure(figsize=(15, 7), linewidth=2)
#
# x = np.linspace(1,616,615)
# x.astype('int')
# for  j in range(len(points)):
#
#
#     # axs[0].plot(x,metric_1_fep[j,:],'o-', color=color_i[j], label='ROI '+str(j))
#     plt.plot(x,metric_1_fepfish[j,:],'o-', color=color_i[j],label='ROI '+str(j+1))
#
#
#     plt.grid()
#     plt.ylim([0,1.1])
#     plt.yticks(np.linspace(0,1,3))
#     plt.xlabel(r'$Iterations $')
#     plt.ylabel( r'$Normalized \quad M_1 $')
#     plt.legend(loc='lower right')
#
# plt.show()
# fig_m1_vs_meas.savefig(save_path+'improvement_metric_over_meas_total.png',dpi=300, bbox_inches='tight')

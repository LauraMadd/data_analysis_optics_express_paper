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
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\experiments_AO\\16_02_21\\fish_1\\fep+fish\\'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_S5\\'
#
# #load images
im_aberrated=np.array(Image.open(semi_path+ 'before_correction.tif'))
im_aniso=np.array(Image.open(semi_path+ 'timelapse_20210216_122312\\00000001.tiff'))
im_preview=np.array(Image.open(semi_path+'cgh_3.2887199999999996.tif'))
# ## im aberrated
max_im_ab=np.amax(im_aberrated)
min_im_ab=np.amin(im_aberrated)
print('im_ab',max_im_ab,min_im_ab )
im_aberrated_norm= (im_aberrated-min_im_ab)/(max_im_ab-min_im_ab)
max_im_ab_norm=np.amax(im_aberrated_norm[1020:1156,833:1230])
min_im_ab_norm=np.amin(im_aberrated_norm[1020:1156,833:1230])
print('im_ab_norm',max_im_ab_norm,min_im_ab_norm )

fig_aberr=plt.figure('Aberrated image', figsize=(45, 30))
fig_aberr=plt.figure('Aberrated image', figsize=(15, 15))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
# plot_aberrated=plt.imshow(im_aberrated_norm[500:1658, 479:1670], vmin=np.amin(im_aberrated_norm[500:1658, 479:1670]), vmax=np.amax(im_aberrated_norm[500:1658, 479:1670]),cmap = 'jet' )
plot_aberrated=plt.imshow(im_aberrated_norm[1020:1156,833:1230], vmin=0, vmax=1, cmap = 'jet', )
# plot_aberrated=plt.imshow(im_aberrated_norm[600:1558, 579:1570], vmin=0, vmax=1, cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_aberrated, ticks=[0, max_im_ab_norm], orientation="horizontal", fraction=0.08, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_aberr.savefig(save_path+'im_ab.png',dpi=300, bbox_inches='tight')
#
#
# # im aniso
max_im_aniso=np.amax(im_aniso)
min_im_aniso=np.amin(im_aniso)
print('im_aniso',max_im_aniso,min_im_aniso )
im_aniso_norm=(im_aniso-min_im_aniso)/(max_im_aniso-min_im_aniso)
max_im_aniso_norm=np.amax(im_aniso_norm[1020:1156,833:1230])
min_im_aniso_norm=np.amin(im_aniso_norm[1020:1156,833:1230])
print('im_aniso_norm',max_im_aniso_norm,min_im_aniso_norm )

fig_aniso=plt.figure('Aniso image', figsize=(15, 15))
plot_aniso=plt.imshow(im_aniso_norm[1020:1156,833:1230],vmin=0, vmax=1, cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
plt.axis('off')
#fraction is 0.047 if we plot the whole fov
cbar_aniso = plt.colorbar(plot_aniso, ticks=[0, max_im_aniso_norm], orientation="horizontal", fraction=0.08, pad=0.01)
cbar_aniso.set_ticklabels([r'$0$', r'$1$'])
plt.show()
fig_aniso.savefig(save_path+'im_aniso.png',dpi=300, bbox_inches='tight')



fig_preview=plt.figure('Preview image', figsize=(15, 15))
plot_preview=plt.imshow(im_preview[210:308,110:406])
scalebar = ScaleBar(0.798, 'um', frameon=False,color='white',location='lower right')
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.show()
fig_preview.savefig(save_path+'im_preview_scale_bis.png',dpi=300, bbox_inches='tight')


# fig_preview=plt.figure('Preview image', figsize=(15, 15))
# plot_preview=plt.imshow(im_preview[210:308,110:406])
#
# plt.axis('off')
# plt.show()
# fig_preview.savefig(save_path+'im_preview.png',dpi=300, bbox_inches='tight')

# analysys roi--------------figure S6 panel c

sub_folder='data_analysis\\gauss_fit\\'
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
for i in range (len(folders)):


    # rois_files=fnmatch.filter(os.listdir('M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\experiments_AO\\02_02\\beads_0.1_um\\set_2\\stack_20210202-175600\\4analysis\\'), '*.tif')
    data_path=semi_path+sub_folder+folders[i]+'\\'
    files = [['original.tif','corrected.tif']]
    print('path:', data_path,'iteration:',i )

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
        im_corr_norm=(im_corr-min_corr)/(max_corr-min_corr)
        im_corr_norm_plot=im_corr_norm
        # im_corr_norm_plot=im_corr_norm[70:120,70:120]
        min_corr_norm=np.amin(im_corr_norm_plot)
        max_corr_norm=np.amax(im_corr_norm_plot)
        print(' min/max corr_norm', min_corr_norm, max_corr_norm)

        # radius_roi=5
        # x=np.linspace(0,radius_roi*2,radius_roi*2)*0.195
        index_im_corr_norm=np.unravel_index(im_corr_norm_plot.argmax(),im_corr_norm_plot.shape)
        print('index im_corr_norm', index_im_corr_norm)

        #------------------------Im original
        min_raw=np.amin(im_raw)
        max_raw=np.amax(im_raw)
        print('max/min raw',min_raw, max_raw)

        # --Normalization
        im_raw_norm=(im_raw-min_raw)/(max_raw-min_raw)
        im_raw_norm_plot=im_raw_norm
        # im_raw_norm_plot=im_raw_norm[70:120,70:120]
        min_raw_norm=np.amin(im_raw_norm_plot)
        max_raw_norm=np.amax(im_raw_norm_plot)
        print(' min/max raw_norm', min_raw_norm, max_raw_norm)

        radius_roi=8
        # x=np.linspace(0,radius_roi*2,radius_roi*2)*0.195
        index_im_raw_norm=np.unravel_index(im_raw_norm_plot.argmax(),im_raw_norm_plot.shape)
        print('index im_raw_norm', index_im_raw_norm)


        # ---------------------Plot xy focus---------------------------------------------
#     fig4, (ax1, ax2)= plt.subplots(1, 2,figsize = (5,5),linewidth=2.0)
#     plot_raw=ax1.imshow(im_raw_norm_plot, vmin=0, vmax=max_raw_norm, cmap = 'jet')
#     cbar_ax1 = plt.colorbar(plot_raw,ax=ax1,ticks=[0, max_raw_norm], orientation="horizontal", fraction=0.047, pad=0.01)
#     cbar_ax1.set_ticklabels([r'$0$', r'$1$'])
#     cbar_ax1.ax.tick_params(labelsize=15)
#     ax1.hlines(y=index_im_raw_norm[0], xmin=index_im_raw_norm[1]-radius_roi, xmax=index_im_raw_norm[1]+radius_roi, colors='black', linestyles='dashed', lw=1)
#     ax1.axis('off')
#     scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
#     # plt.gca().add_artist(scalebar)
#     ax1.add_artist(scalebar)
#
#     plot_corr=ax2.imshow(im_corr_norm_plot, vmin=0, vmax=1, cmap = 'jet')
#     cbar_ax2 = plt.colorbar(plot_corr,ax=ax2,ticks=[0, max_corr_norm], orientation="horizontal", fraction=0.047, pad=0.01)
#     cbar_ax2.ax.tick_params(labelsize=15)
#     cbar_ax2.set_ticklabels([r'$0$', r'$1$'])
#     # ax2.hlines(y=54, xmin=54-20, xmax=54+20, colors='gold', linestyles='dashed', lw=1)
#     ax2.hlines(y=index_im_corr_norm[0], xmin=index_im_corr_norm[1]-radius_roi, xmax=index_im_corr_norm[1]+radius_roi, colors='black', linestyles='dashed', lw=1)
#     ax2.axis('off')
#     scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
#     ax2.add_artist(scalebar)
#     plt.show()
#     fig4.savefig(save_path+'xy_image'+str(folders[i])+'.png',dpi=300, bbox_inches='tight')
# #
#
#     #--------------------------------Line profile ----------------------------------
    # --- Fit line prifile  + FWHM calculation -------------------------------------
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
# #     # #----------------------Figure panel g single lines----------------------------------------
#     # fig5=plt.figure('line profile aniso fit FWHM p0 aniso vs p0 raw',figsize = (11,11),linewidth=2.0)
#     # plt.plot(xfit,fit_corr,'-', color='darkblue',label = 'correction fit')
#     # plt.plot(xfit,fit_raw,'-', color='darkorange',label = 'aberration fit')
#     # plt.plot(x,im_raw_norm_plot[index_im_raw_norm[0], :], '.', color='darkorange',label='aberration')
#     # plt.plot(x,im_corr_norm_plot[index_im_corr_norm[0],:], '.', color='darkblue',label='correction')
#     #
#     #
#     # # plt.plot(FWHM_fit_raw,0.5*(np.amax(fit_raw)-np.amin(fit_raw))*np.ones(np.shape(FWHM_fit_raw)),color='darkorange',linestyle='dashed', label = 'FWHM=' + str(round(FWHM_fit_raw[-1]-FWHM_fit_raw[0],3)) + '$\mu m$')
#     # # plt.plot(FWHM_fit_corr,0.5*(np.amax(fit_corr)-np.amin(fit_corr))*np.ones(np.shape(FWHM_fit_corr)),color='darkblue',linestyle='dashed',label = 'FWHM=' + str(round(FWHM_fit_corr[-1]-FWHM_fit_corr[0],3)) + r'$\mu m$')
#     # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
#     # plt.ylabel(r'$Normalized \quad intensity $')
#     # plt.legend(loc='upper left')
#     # # plt.title('aniso fit FWHM corr vs raw')
#     # plt.show()
#     # fig5.savefig(save_path+'xy_line_profile'+str(folders[i])+ '.png',dpi=300, bbox_inches='tight')
#
fig_profiles, axs = plt.subplots(1,2,figsize=(15, 5))
fig_profiles.subplots_adjust(hspace = .3, wspace=.3)
axs = axs.ravel()
#
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
axs[1].legend(loc='lower center', bbox_to_anchor=(0.3, -0.6),fancybox=False, shadow=False, ncol=4, fontsize=MEDIUM_SIZE)
plt.show()
fig_profiles.savefig(save_path+'total_xy_line_profile_int.png',dpi=300, bbox_inches='tight')


#-------------------  plotting the intensity/ diameter progression through the zernikes

#---fep correction
#
folder= 'M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\experiments_AO\\16_02_21\\fish_1\\fep+fish\\20210216-120922_aniso\\'
points = ['000','001']
n_zerny=21
color_i=['blue', 'green', 'red']
x = np.linspace(3,n_zerny,17)
metric_1=np.zeros((len(points),17), dtype=np.float)
diameter =np.zeros((len(points),17), dtype=np.float)
intensity =np.zeros((len(points),17), dtype=np.float)
intensity_max =np.zeros((len(points),17), dtype=np.float)

for j, point  in enumerate (points):
    # zernike = '14'
    nr = '07' # evaluate each image in 0 to account for previous zernike
    number = []
    for q in range(3,n_zerny):
        if q == 4:
            continue
        if q < 10:
            number += [str(q)]
        else:
            number += [str(q)]

    diameter_temp =np.zeros(np.shape(number))
    intensity_temp =np.zeros(np.shape(number))
    intensity_max_temp =np.zeros(np.shape(number))
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
        intensity_max_temp[i] = np.amax(image)
        # metric_1_temp[i]=intensity_temp[i]**2*diameter_temp[i]
        intensity_temp[i]=np.sum(image)**2
        metric_1_temp[i]=intensity_temp[i]*diameter_temp[i]
        i += 1
    intensity[j,:] = intensity_temp/np.amax(intensity_temp)
    diameter[j,:]  = diameter_temp/np.amax(diameter_temp)
    metric_1[j,:] = metric_1_temp/np.amax(metric_1_temp)
    intensity_max[j,:] =intensity_max_temp/np.amax(intensity_max_temp)
#     # figs in for loop
#     # x = np.linspace(3,44,41)
# fig_int=plt.figure('Intensity',  figsize=(15, 5))
# for  j in range(len(points)):
#     plt.plot(x,intensity[j,:],'o-',color=color_i[j],label='ROI '+str(j))
#     plt.legend(loc='lower right',  fontsize=SMALL_SIZE)
#     plt.ylim([0,1.1])
#     # plt.xticks(np.linspace(3)
#     plt.ylabel(r'$Normalized \quad I^2 $',fontsize=MEDIUM_SIZE)
#     plt.xlabel(r'$Aberration \quad index $',fontsize=MEDIUM_SIZE)
#     plt.yticks(np.linspace(0,1,3))
#     plt.grid(True)
# plt.show()
# fig_int.savefig(save_path+'intensity2_vs_index.png',dpi=300, bbox_inches='tight')
#
#
fig_max_int=plt.figure('Intensity',  figsize=(10, 5))
for  j in range(len(points)):
    plt.plot(x,intensity_max[j,:],'o-',color=color_i[j],label='ROI '+str(j+1))
    plt.legend(loc='lower right',  fontsize=SMALL_SIZE)
    plt.ylim([0,1.1])
    # plt.xticks(np.linspace(3)
    plt.ylabel(r'$ Normalized \quad I \quad max $',fontsize=MEDIUM_SIZE)
    plt.xlabel(r'$Aberration \quad index $',fontsize=MEDIUM_SIZE)
    plt.yticks(np.linspace(0,1,3))
    plt.grid(True)
plt.show()
fig_max_int.savefig(save_path+'max_int_vs_index.png',dpi=300, bbox_inches='tight')
#
#
#
#
# fig_d=plt.figure('Diameter',  figsize=(10, 5))
# for  j in range(len(points)):
#     plt.plot(x,diameter[j,:],'*--', color=color_i[j],label='ROI '+str(j))
#     plt.legend(loc='lower right',  fontsize=SMALL_SIZE)
#     plt.ylim([0,1.1])
#     plt.yticks(np.linspace(0,1,3))
#     plt.ylabel(r'$Normalized \quad diameter  $',fontsize=MEDIUM_SIZE)
#     plt.xlabel(r'$Aberration \quad index $',fontsize=MEDIUM_SIZE)
#     plt.grid(True)
# plt.show()
# fig_d.savefig(save_path+'diameter_vs_index.png',dpi=300, bbox_inches='tight')
#
fig_M1=plt.figure('M1',  figsize=(10, 5))
for  j in range(len(points)):
    plt.plot(x,metric_1[j,:],'*--', color=color_i[j],label='ROI '+str(j+1))
    plt.legend(loc='lower right',  fontsize=SMALL_SIZE)
    plt.ylim([0,1.1])
    plt.yticks(np.linspace(0,1,3))
    plt.ylabel(r'$Normalized \quad M_1  $',fontsize=MEDIUM_SIZE)
    plt.xlabel(r'$Aberration \quad index $',fontsize=MEDIUM_SIZE)
    plt.grid(True)

plt.show()
fig_M1.savefig(save_path+'M1_vs_index.png',dpi=300, bbox_inches='tight')

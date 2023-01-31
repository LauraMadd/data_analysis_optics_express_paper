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
MEDIUM_SIZE = 25
BIGGER_SIZE=35

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=40)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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



#-------------------------Paths ---------------------------------
semi_path ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\experiments_AO\\30_03_21\\3D_holo\\'
semi_path_random =semi_path+'random\\'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_5\\'
#-----------------------------------------------------------------------------

#values z positions
coords_im=np.load('M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\experiments_AO\\30_03_21\\3D_holo\\random\\coords_20210330-184343\\coords_cameraholoName.npy')
z=np.zeros((coords_im.shape[0]),dtype=float )
z_index=np.load('M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\experiments_AO\\30_03_21\\3D_holo\\random\\coords_20210330-184343\\stack_index.npy')
z_min=39.7039
dz=0.49629

for i in range(coords_im.shape[0]):
    print(i, 'index')

    z[i]=round(coords_im[i,2],2)
    print(i, 'index', z[i],'z')

# ##-----------------------------------Total max projection images, figure 5 panel f ---------------
###Corrected
total_im_corr= np.array(Image.open(semi_path_random+'MAX_timelapse_20210330_184832.tif'))
#max and min corrected raw
min_total_corr=np.amin(total_im_corr)
max_total_corr=np.amax(total_im_corr)
print('max/min corr total',min_total_corr, max_total_corr)

# --Normalization
total_im_corr_norm=(total_im_corr-min_total_corr)/(max_total_corr-min_total_corr)
# total_im_corr_norm_plot=(total_im_corr_norm)
total_im_corr_norm_plot=total_im_corr_norm[750:1400,530:1480]
#max and min corrected normalized
min_total_im_corr_norm=np.amin(total_im_corr_norm_plot)
max_total_im_corr_norm=np.amax(total_im_corr_norm_plot)
print(' min/max corr total norm', min_total_im_corr_norm, max_total_im_corr_norm)

####Raw, panel f left
total_im_raw= np.array(Image.open(semi_path_random+'MAX_stack_20210330-184550.tif'))
#max and min corrected raw
min_total_raw=np.amin(total_im_raw)
max_total_raw=np.amax(total_im_raw)
print('max/min raw total',min_total_raw, max_total_raw)

# --Normalization
total_im_raw_norm=(total_im_raw-min_total_corr)/(max_total_corr-min_total_corr)
# total_im_raw_norm_plot=(total_im_raw_norm)
total_im_raw_norm_plot=total_im_raw_norm[750:1400,530:1480]
#max and min corrected normalized
# # im_corr_norm_plot=im_corr_norm[70:120,70:120]
min_total_im_raw_norm=np.amin(total_im_raw_norm_plot)
max_total_im_raw_norm=np.amax(total_im_raw_norm_plot)
print(' min/max raw total norm', min_total_im_raw_norm, max_total_im_raw_norm)
## figure aberrated image
fig_1=plt.figure('Max projection aberrated', figsize=(30, 15))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_raw=plt.imshow(total_im_raw_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_raw, ticks=[0, 1], orientation="vertical", fraction=0.05, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_1.savefig(save_path+'max_raw.png',dpi=300, bbox_inches='tight')
# #
# # #figure corr fep_correction, panel f left
fig_2=plt.figure('Max projection corrected', figsize=(30, 15))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_fep=plt.imshow(total_im_corr_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_fep, ticks=[0, 1], orientation="vertical", fraction=0.05, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_2.savefig(save_path+'max_corr.png',dpi=300, bbox_inches='tight')

#--------------------Plot coordinates random target
coords_target=np.load(semi_path+'coords_20210330-172600\\'+'coords_cameraholoName.npy')
coords_rand=np.load(semi_path_random+'coords_20210330-184343\\'+'coords_cameraholoName.npy')
x_ticks=np.linspace(-50,50,3)
y_ticks=x_ticks
z_ticks=[-10,0,30]
n_points=coords_target.shape[0]
dst_target_rand=np.zeros((n_points),dtype=np.float64)
dst_target_rand[:] =np.sqrt((coords_target[:,0]-coords_rand[:,0])**2+(coords_target[:,1]-\
            coords_rand[:,1])**2+(coords_target[:,2]-coords_rand[:,2])**2)

#=============================Plot, figure 5 panel e ==============================================
fig3 = plt.figure(figsize=(10,10), linewidth=2.0 )
ax = fig3.add_subplot(1,2,1, projection='3d')
ax.scatter(coords_target[:,0],coords_target[:,1],coords_target[:,2], s=100,  c='green', marker='^', label='Reference')
ax.scatter(coords_rand[:,0],coords_rand[:,1],coords_rand[:,2], s=100, c='red', marker='o', label= 'Random')
ax.set_xlabel(r'$x (\mu m)$',labelpad=30)
ax.set_ylabel(r'$y (\mu m)$', labelpad=30)
ax.set_zlabel(r'$z (\mu m)$',labelpad=5)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_zticks(z_ticks)
# ax.legend( loc='upper center', bbox_to_anchor=(0.5, -0.4),fancybox=False, shadow=False, ncol=2)
ax.legend( loc='best' )


ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.show()
fig3.savefig(save_path+'coords_triss.png',dpi=300, bbox_inches='tight')
##-----------------------------------Total max projection of reslices images---------------
# ### xz Corrected
# xz_corr= np.array(Image.open(semi_path_random+'MAX_Reslice left timelapse_20210330_184832.tif'))
# #max and min corrected raw
# min_xz_corr=np.amin(xz_corr)
# max_xz_corr=np.amax(xz_corr)
# print('max/min corr total',min_xz_corr, max_xz_corr)
#
# # --Normalization
# xz_corr_norm=(xz_corr-min_xz_corr)/(max_xz_corr-min_xz_corr)
# # total_im_corr_norm_plot=(total_im_corr_norm)
# xz_corr_norm_plot=xz_corr_norm[:,700:1400]
# #max and min corrected normalized
# min_xz_corr_norm=np.amin(xz_corr_norm_plot)
# max_xz_corr_norm=np.amax(xz_corr_norm_plot)
# print(' min/max corr total norm',min_xz_corr_norm, max_xz_corr_norm)
#
# ####Raw
# xz_raw= np.array(Image.open(semi_path_random+'MAX_Reslice left stack_20210330-184550.tif'))
# #max and min corrected raw
# min_xz_raw=np.amin(xz_raw)
# max_xz_raw=np.amax(xz_raw)
# print('max/min corr total',min_xz_raw, max_xz_raw)
#
# # --Normalization
# xz_raw_norm=(xz_raw-min_xz_corr)/(max_xz_corr-min_xz_corr)
# # total_im_corr_norm_plot=(total_im_corr_norm)
# xz_raw_norm_plot=xz_raw_norm[:,700:1400]
# #max and min corrected normalized
# min_xz_raw_norm=np.amin(xz_raw_norm_plot)
# max_xz_raw_norm=np.amax(xz_raw_norm_plot)
# print(' min/max corr total norm', min_xz_raw_norm, max_xz_raw_norm)
# ## figure aberrated image
# fig_3=plt.figure('xz aberrated',figsize=(30, 15))
# # plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
# plot_raw=plt.imshow(xz_raw_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
# scalebar = ScaleBar(0.195, 'um',location='lower right',font_properties={'family':'calibri', 'size': 40})
# plt.gca().add_artist(scalebar)
# cbar_ab = plt.colorbar(plot_raw, ticks=[0, 1], orientation="vertical", fraction=0.0055, pad=0.01)
# #cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
# cbar_ab.set_ticklabels([r'$0$', r'$1$'])
# plt.axis('off')
# plt.show()
# fig_3.savefig(save_path+'xz_raw.png',dpi=300, bbox_inches='tight')
# # #
# # # #figure corr fep_correction
# fig_4=plt.figure('xz corrected', figsize=(30, 15))
# # plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
# plot_fep=plt.imshow(xz_corr_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
# scalebar = ScaleBar(0.195, 'um',location='lower right',font_properties={'family':'calibri', 'size': 40})
# plt.gca().add_artist(scalebar)
# cbar_ab = plt.colorbar(plot_fep, ticks=[0, 1], orientation="vertical", fraction=0.0055, pad=0.01)
# #cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
# cbar_ab.set_ticklabels([r'$0$', r'$1$'])
# plt.axis('off')
# plt.show()
# fig_4.savefig(save_path+'xz_corr.png',dpi=300, bbox_inches='tight')
#
# ###- yz Corrected
# yz_corr= np.array(Image.open(semi_path_random+'MAX_Reslice_top_timelapse_20210330_184832.tif'))
# #may and min corrected raw
# min_yz_corr=np.amin(yz_corr)
# max_yz_corr=np.amax(yz_corr)
# print('max/min corr total',min_yz_corr, max_yz_corr)
#
# # --Normalization
# yz_corr_norm=(yz_corr-min_yz_corr)/(max_yz_corr-min_yz_corr)
# # total_im_corr_norm_plot=(total_im_corr_norm)
# yz_corr_norm_plot=yz_corr_norm[:,700:1400]
# #max and min corrected normalized
# min_yz_corr_norm=np.amin(yz_corr_norm_plot)
# max_yz_corr_norm=np.amax(yz_corr_norm_plot)
# print(' min/max corr total norm',min_yz_corr_norm, max_yz_corr_norm)
#
# ####Raw
# yz_raw= np.array(Image.open(semi_path_random+'MAX_Reslice top stack_20210330-184550.tif'))
# #may and min corrected raw
# min_yz_raw=np.amin(yz_raw)
# may_yz_raw=np.amax(yz_raw)
# print('max/min corr total',min_yz_raw, may_yz_raw)
#
# # --Normalization
# yz_raw_norm=(yz_raw-min_yz_corr)/(max_yz_corr-min_yz_corr)
# # total_im_corr_norm_plot=(total_im_corr_norm)
# yz_raw_norm_plot=yz_raw_norm[:,700:1400]
# #max and min corrected normalized
# min_yz_raw_norm=np.amin(yz_raw_norm_plot)
# max_yz_raw_norm=np.amax(yz_raw_norm_plot)
# print(' min/max corr total norm', min_yz_raw_norm, max_yz_raw_norm)
# ## figure aberrated image
# fig_5=plt.figure('yz aberrated',figsize=(30, 15))
# # plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
# plot_raw=plt.imshow(yz_raw_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
# scalebar = ScaleBar(0.195, 'um',font_properties={'family':'calibri', 'size': 40})
# plt.gca().add_artist(scalebar)
# cbar_ab = plt.colorbar(plot_raw, ticks=[0, 1], orientation="vertical", fraction=0.0055, pad=0.01)
# #cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
# cbar_ab.set_ticklabels([r'$0$', r'$1$'])
# plt.axis('off')
# plt.show()
# fig_5.savefig(save_path+'yz_raw.png',dpi=300, bbox_inches='tight')
# # #
# # # #figure corr fep_correction
# fig_6=plt.figure('yz corrected', figsize=(30, 15))
# # plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
# plot_fep=plt.imshow(yz_corr_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
# scalebar = ScaleBar(0.195, 'um',font_properties={'family':'calibri', 'size': 40})
# plt.gca().add_artist(scalebar)
# cbar_ab = plt.colorbar(plot_fep, ticks=[0, 1], orientation="vertical", fraction=0.0055, pad=0.01)
# #cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
# cbar_ab.set_ticklabels([r'$0$', r'$1$'])
# plt.axis('off')
# plt.show()
# fig_6.savefig(save_path+'yz_corr.png',dpi=300, bbox_inches='tight')

#===============================================================================
# # #----------------------------------Roi xy planes -------------------------------
# #
sub_folder='data_analysis\\best_best_gauss_fit\\'
# sub_folder='data_analysis\\best_gauss_fit\\'

folders=os.listdir(semi_path_random+sub_folder)
#variables to save all data
results_fit_corr=np.zeros((len(folders), 2000), dtype=float)
results_fit_raw=np.zeros((len(folders), 2000), dtype=float)
results_raw=np.zeros((len(folders), 40), dtype=float)
results_corr=np.zeros((len(folders), 40), dtype=float)
results_max_corr=np.zeros((len(folders), 1), dtype=float)
results_max_raw=np.zeros((len(folders), 1), dtype=float)
results_max_corr_fit=np.zeros((len(folders), 1), dtype=float)
results_max_raw_fit=np.zeros((len(folders), 1), dtype=float)
results_i_corr=np.zeros((len(folders), 1), dtype=float)
results_i_raw=np.zeros((len(folders), 1), dtype=float)

for i in range (len(folders)):


    data_path=semi_path_random+sub_folder+folders[i]+'\\'
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


#     #---------------------Plot xy focus---------------------------------------------
#     # fig4, (ax1, ax2)= plt.subplots(1, 2,figsize = (5,5),linewidth=2.0)
#     # plot_raw=ax1.imshow(im_raw_norm_plot, vmin=0, vmax=max_corr_norm, cmap = 'jet')
#     # cbar_ax1 = plt.colorbar(plot_raw,ax=ax1,ticks=[0, max_corr_norm], orientation="horizontal", fraction=0.047, pad=0.01)
#     # cbar_ax1.set_ticklabels([r'$0$', r'$1$'])
#     # cbar_ax1.ax.tick_params(labelsize=15)
#     # ax1.hlines(y=index_im_raw_norm[0], xmin=index_im_raw_norm[1]-radius_roi, xmax=index_im_raw_norm[1]+radius_roi, colors='black', linestyles='dashed', lw=1)
#     # ax1.axis('off')
#     # scalebar = ScaleBar(0.195, 'um')
#     # ax1.add_artist(scalebar)
#     #
#     # plot_corr=ax2.imshow(im_corr_norm_plot, vmin=0, vmax=1, cmap = 'jet')
#     # cbar_ax2 = plt.colorbar(plot_corr,ax=ax2,ticks=[0, max_corr_norm], orientation="horizontal", fraction=0.047, pad=0.01)
#     # cbar_ax2.ax.tick_params(labelsize=15)
#     # cbar_ax2.set_ticklabels([r'$0$', r'$1$'])
#     # ax2.hlines(y=index_im_corr_norm[0], xmin=index_im_corr_norm[1]-radius_roi, xmax=index_im_corr_norm[1]+radius_roi, colors='black', linestyles='dashed', lw=1)
#     # ax2.axis('off')
#     # scalebar = ScaleBar(0.195, 'um')
#     # ax2.add_artist(scalebar)
#     # fig4.subplots_adjust( bottom=0.45)
#     # plt.suptitle(r'$ Z=$'+str(z[i])+r'$ \mu m$')
#     # plt.show()
#     # fig4.savefig(save_path+'xy_image'+str(folders[i])+'.png',dpi=300, bbox_inches='tight')
#
# #
# # #     # #--------------------------------Line profile ----------------------------------
# # #     #
# # #     # #-------------------------------------------------------------------------------
#     # --- Fit line prifile  + FWHM calculation -------------------------------------
    def lor(x, amp1, cen1, wid1,b,off):
        return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + b*x + off
    def gauss(x, amp1,cen1,sigma1,b):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + b*x

    #--- weights, saved_ints, and xfit2 have to be altered
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

    FWHM_fit_raw = xfit[fit_raw >= 0.5*(np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw)]
    FWHM_fit_corr = xfit[fit_corr >= 0.5*(np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr)]

    # save results for each point in global arrays.
    results_fit_corr[i, :]=fit_corr
    results_fit_raw[i, :]=fit_raw
    results_raw[i, :]=im_raw_norm_plot[ :,index_im_raw_norm[0]]
    results_corr[i, :]=im_corr_norm_plot[:, index_im_raw_norm[0]]

    results_i_corr[i, :]=np.sum(im_corr)
    results_i_raw[i, :]=np.sum(im_raw)

    results_max_corr[i, :]=np.amax(im_corr_norm_plot[ :,index_im_raw_norm[0]])
    results_max_raw[i, :]=np.amax(im_raw_norm_plot[ :,index_im_raw_norm[0]])

    results_max_corr_fit[i, :]=np.amax(fit_corr)
    results_max_raw_fit[i, :]=np.amax(fit_raw)


# # ##----------------------Quantifications ----------------------------------
I_improvement_perc=(results_max_corr-results_max_raw)*100/results_max_corr
av_I_lateral_perc=np.average(I_improvement_perc)
std_I_lateral_perc=np.std(I_improvement_perc)
I_improvement_factor=(results_max_corr/results_max_raw)
av_I_lateral_factor=np.average(I_improvement_factor)
std_I_lateral_factor=np.std(I_improvement_factor)
table_results=tabulate([['av_I_Improvement percentage ',av_I_lateral_perc ],['std_I_Improvement percentage ',std_I_lateral_perc],['av_I_Improvement factor ',av_I_lateral_factor ],['std_I_Improvement factor ',std_I_lateral_factor]],headers=['Correction improvement'])

f = open(save_path+'Corrected-improvement-xy_3D.txt', 'w')
f.write(table_results)
f.close()
#----------------------plot intensities, figure 5 panel g----------------------------------------
fig_int_max =plt.figure('intensities', figsize=(8, 2), linewidth=2.0)
x_p = np.arange(0,len(folders), 1)
plt.plot(x_p, results_max_raw,'o', color='darkorange',  label = 'raw')
plt.plot(x_p, results_max_corr,'o', color='darkblue',label = 'corrected',mfc='none')
plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE),
plt.ylabel(r'$ I \quad  max. \quad norm.$',fontsize= SMALL_SIZE)
plt.title('Lateral',fontsize= SMALL_SIZE)
plt.xticks(x_p, x_p+1)
plt.ylim([0,1.2])
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
plt.show()
fig_int_max.savefig(save_path+'max_intensities_norm_3D.png',dpi=300, bbox_inches='tight')

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

# #----------------------plot fidelity ----------------------------------------
#===================== reference coords: SLM  coords
coords_ref_temp=coords_rand
coords_ref_temp=np.transpose(coords_ref_temp)
n_points=coords_ref_temp.shape[1]

coords_ref=np.zeros((3,n_points))


coords_ref=coords_ref_temp
# =================corr_coords : coordinates adter correction.
# coords from image
# -----system parameters
cam_h=2048
off_y=cam_h/2
off_x=cam_h/2
pix_size=6.5
M_2=9./300
n_points=coords_ref.shape[1]
defocus_voltage=np.zeros((n_points), dtype=np.float)
for i in range(n_points):
    defocus_voltage[i]= -0.0153835 *coords_ref[2,i]+2.65808
coords_corr_im=np.load(semi_path_random+"centers_corr_cgh.npy")
coords_raw_im=np.load(semi_path_random+"centers_raw_cgh.npy")

coords_corr_im_bis=np.load(semi_path_random+"centers_corr_cgh_tris.npy",allow_pickle=True)
coords_raw_im_bis=np.load(semi_path_random+"centers_raw_cgh_tris.npy",allow_pickle=True)

coords_corr_im[1]=coords_corr_im_bis[1][3]
coords_corr_im[2]=coords_corr_im_bis[2][3]
coords_corr_im[3]=coords_corr_im_bis[3][3]
coords_corr_im[4]=coords_corr_im_bis[4][2]
coords_corr_im[5]=coords_corr_im_bis[5][2]

coords_raw_im[1]=coords_raw_im_bis[1][2]
coords_raw_im[2]=coords_raw_im_bis[2][2]
coords_raw_im[3]=coords_raw_im_bis[3][2]
coords_raw_im[6]=coords_raw_im_bis[6][0]
coords_raw_im[7]=coords_raw_im_bis[7][1]
coords_raw_im[8]=coords_raw_im_bis[8][0]
# coords_corr_im_bis=np.load(semi_path+"centers_corr.npy")
# coords_raw_im_bis=np.load(semi_path+"centers_raw.npy")
centers=np.zeros((2,n_points))
#swap for carthesian coordinates
centers[0,:]=coords_corr_im[:,1]
centers[1,:]=coords_corr_im[:,0]
# # transformation in samples
coords_corr_sample=np.ones((4,n_points), dtype=np.float64)
xc_im=centers[0,:]-off_x
yc_im=off_y-centers[1,:]
coords_corr_sample[0,:]=xc_im*M_2*pix_size
coords_corr_sample[1,:]=yc_im*M_2*pix_size
coords_corr_sample[2,:]= 136.906*(0.608306*defocus_voltage[:]-1.5091)
# #transformation to SLM
camera_to_slm_matrix=np.load('M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\experiments_AO\\30_03\\t_aff_rand_bigrange_20_fov130\\T_affine.npy')
coords_corr_slm=np.zeros((3,n_points))
coords_corr_slm=np.dot(camera_to_slm_matrix,coords_corr_sample)

coords_flip=np.copy(coords_corr_slm)
coords_flip[1,:]=coords_corr_slm[1,:]*(-1.0) # flipy


#Apply rot matrix -90
coords_rot=np.copy(coords_corr_slm)
rot_matrix=np.zeros((2,2))
rot_matrix[0,1]=1
rot_matrix[1,0]=-1
coords_rot[:2,:]=np.dot(rot_matrix,coords_flip[:2,:])

coords_corr=coords_rot
coords_corr=coords_corr_slm

# calculation accuracy
dst=np.zeros((n_points),dtype=np.float64)
dst[:] =np.sqrt((coords_ref[0,:]-coords_corr[0,:])**2+(coords_ref[1,:]-\
            coords_corr[1,:])**2+(coords_ref[2,:]-coords_corr[2,:])**2)
average_dst=np.average(dst)
sigma_dst=np.std(dst)

#calculation difference between corrected and raw spots in theimage
dst_corr_raw=np.zeros((n_points),dtype=np.float64)
dst_corr_raw[:] =np.sqrt((coords_raw_im[:,0]-coords_corr_im[:,0])**2+(coords_raw_im[:,1]-\
            coords_corr_im[:,1])**2)*0.195
z_input=[-28.785327,-23.822340, -15.881560,-7.940780 , -2.977793, 2.977793,7.940780, 13.896360, 22.829742,26.800132]
z_input=np.asarray(z_input,dtype=np.float)
dz=0.992597
zstep_corr=[4,2,2,2,1,1,0,-1,-1,4]
zstep_corr=np.asarray(zstep_corr,dtype=np.float)
zstep_raw=[2,1,1,1,0,0,-1,-1,-2,4]
zstep_raw=np.asarray(zstep_raw,dtype=np.float)
z_corr=0.992597
z_corr=z_input+dz*zstep_corr
z_raw=z_input+dz*zstep_raw
dst_corr_raw_z=np.sqrt((z_raw-z_corr)**2)
average_dst_corr_raw_z=np.average(dst_corr_raw_z)

average_dst_corr_raw=np.average(dst_corr_raw)
sigma_dst_corr_raw=np.std(dst_corr_raw)


# np.save(semi_path_random+'dst_raw',dst)

dst_raw=np.load(semi_path_random+'dst_raw.npy')
average_dst_raw=np.average(dst_raw)
sigma_dst_raw=np.std(dst_raw)
#
# fig3 = plt.figure(figsize=(10,3), linewidth=2.0)
# x= np.linspace(1,n_points, n_points)
# x_ticks=[]
# x_ticks=x
# plt.plot(x,dst,'o',color='darkblue')
# plt.axhline(y=average_dst,color='darkblue', alpha=.5, linestyle='--')
# plt.xticks(x_ticks)
# plt.xlabel(r'$ ROIs $',fontsize= SMALL_SIZE)
# plt.tick_params( bottom = False)
# plt.ylabel(r'$ Fidelity\quad (\mu m)$',fontsize= SMALL_SIZE)
# plt.ylim([0,1.6])
# plt.show()
# fig3.savefig(save_path+'fidelity_bis.png',dpi=300, bbox_inches='tight')
#
# #plot comparison fidelity
# fig4 = plt.figure(figsize=(10,3), linewidth=2.0)
# x= np.linspace(1,n_points, n_points)
# x_ticks=[]
# x_ticks=x
# y_ticks=np.linspace(0,1.5,4)
# plt.plot(x,dst, 'o',color='darkblue',label= 'corrected')
# plt.axhline(y=average_dst,color='darkblue', alpha=.5, linestyle='--')
# plt.plot(x,dst_raw, 'o',color='darkorange',label= 'raw')
# plt.axhline(y=average_dst_raw,color='darkorange', alpha=.5, linestyle='--')
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)
# plt.xlabel(r'$ ROIs $',fontsize= SMALL_SIZE)
# plt.ylabel(r'$ Fidelity\quad (\mu m)$',fontsize= SMALL_SIZE)
# plt.ylim([0,1.6])
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2)
#
# plt.show()
# fig4.savefig(save_path+'fidelity_comparison.png',dpi=300, bbox_inches='tight')
#---Figure 5 panel h
fig5 = plt.figure(figsize=(8,2), linewidth=2.0)
x= np.linspace(1,n_points, n_points)
x_ticks=[]
x_ticks=x
y_ticks=np.linspace(0,1,3)
plt.axhline(y=average_dst_corr_raw, color='darkblue', alpha=.5, linestyle='--')
plt.plot(x,dst_corr_raw,'o',color='darkblue')
plt.xticks(x_ticks)
plt.xlabel(r'$ ROIs $')
plt.ylabel(r'$ Distance \quad (\mu m)$')
plt.yticks(y_ticks)
plt.ylim([0,1.5])
plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2)
plt.show()
fig5.savefig(save_path+'distance_xy.png',dpi=300, bbox_inches='tight')

fig5bis= plt.figure(figsize=(8,2), linewidth=2.0)
x= np.linspace(1,n_points, n_points)

x_ticks=[]
x_ticks=x
plt.axhline(y=average_dst_corr_raw_z, color='darkblue', alpha=.5, linestyle='--')
plt.plot(x,dst_corr_raw_z,"v",color='darkblue')
plt.xticks(x_ticks)
plt.xlabel(r'$ ROIs $')
plt.ylabel(r'$ Distance \quad (\mu m)$')
plt.ylim([-0.5,1.5])
plt.yticks(y_ticks)
plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2)
plt.show()
fig5bis.savefig(save_path+'distance_z.png',dpi=300, bbox_inches='tight')


# #------------------------------------------------------------------------------
# # #-----------------------Roi z planes--------------------------------------------
# sub_folder='data_analysis\\gauss_fit_z\\'
# folders=os.listdir(semi_path+sub_folder)
# #variables to save all data
# results_fit_corr=np.zeros((len(folders), 2000), dtype=float)
# results_fit_raw=np.zeros((len(folders), 2000), dtype=float)
# results_raw=np.zeros((len(folders), 39), dtype=float)
# results_corr=np.zeros((len(folders), 39), dtype=float)
# im_raw_norm_plot_array=np.zeros((len(folders), 39,40),dtype=np.float64)
# im_corr_norm_plot_array=np.zeros((len(folders), 39,40),dtype=np.float64)
# max_index_corr_array=np.zeros((len(folders), 1),dtype=int)
# max_index_raw_array=np.zeros((len(folders), 1),dtype=int)
#
# resultsz_max_corr_fit=np.zeros((len(folders), 1), dtype=float)
# resultsz_max_raw_fit=np.zeros((len(folders), 1), dtype=float)
# resultsz_i_corr=np.zeros((len(folders), 1), dtype=float)
# resultsz_i_raw=np.zeros((len(folders), 1), dtype=float)
# for i in range (len(folders)):
#
# #
#     data_path=semi_path+sub_folder+folders[i]+'\\'
#     files = [['z_original.tif','z_corrected.tif']]
#     print(folders[i])
#     for name in files[0]:
#         print(name)
#         image = Image.open(data_path+name)
#         if (name == 'z_original.tif'):
#             im_raw=np.array(image)
#             print(name)
#         im_corr=np.array(image)
#
#     #---------------Im corr
#
#     min_corr=np.amin(im_corr)
#     max_corr=np.amax(im_corr)
#     print('max/min corr',min_corr, max_corr)
#
#     # --Normalization
#     im_corr_norm=(im_corr)/(max_corr)
#     im_corr_norm_plot=im_corr_norm
#     # im_corr_norm_plot=im_corr # to plot without norm
#     min_corr_norm=np.amin(im_corr_norm_plot)
#     max_corr_norm=np.amax(im_corr_norm_plot)
#     print(' min/max corr_norm', min_corr_norm, max_corr_norm)
#     im_corr_norm_plot_array[i,:,:]=im_corr_norm_plot
#
#     index_im_corr_norm=np.unravel_index(im_corr_norm_plot.argmax(),im_corr_norm_plot.shape)
#     print('index im_corr_norm', index_im_corr_norm)
#     max_index_raw_array[i,:]=index_im_corr_norm[0]
#     #------------------------Im original
#     min_raw=np.amin(im_raw)
#     max_raw=np.amax(im_raw)
#     print('max/min raw',min_raw, max_raw)
#
#     # --Normalization
#     im_raw_norm=(im_raw)/(max_corr)
#     im_raw_norm_plot=im_raw_norm
#     # im_raw_norm_plot=im_raw # to plot without norm
#
#     min_raw_norm=np.amin(im_raw_norm_plot)
#     max_raw_norm=np.amax(im_raw_norm_plot)
#     print(' min/max raw_norm', min_raw_norm, max_raw_norm)
#     im_raw_norm_plot_array[i,:,:]=im_raw_norm_plot
#     radius_roi=0
#     # x=np.linspace(0,radius_roi*2,radius_roi*2)*0.195
#     index_im_raw_norm=np.unravel_index(im_raw_norm_plot.argmax(),im_raw_norm_plot.shape)
#     print('index im_raw_norm', index_im_raw_norm)
#     max_index_raw_array[i,:]=index_im_raw_norm[0]
#     print('shapes',im_raw_norm_plot.shape, im_corr_norm_plot.shape, type(im_corr_norm_plot),type(im_corr_norm_plot[0,2])  )
#
# # #==========================================Fit z profile========================
#     #weights, saved_ints, and xfit2 have to be altered
#     x=np.linspace(-19.5,19.5,39)*0.195
#     # x=np.linspace(0,40,40)*0.195
#     meancorr = sum(x * im_corr_norm_plot[:,index_im_corr_norm[1]]) / sum(im_corr_norm_plot[:,index_im_corr_norm[1]])
#     sigmacorr = np.sqrt(sum(im_corr_norm_plot[:,index_im_corr_norm[1]] * (x - meancorr)**2) / sum(im_corr_norm_plot[:,index_im_corr_norm[1]]))
#
#     parscorr, covcorr = curve_fit(f=lor,xdata=x,ydata=im_corr_norm_plot[:,index_im_corr_norm[1]],p0=[max(im_corr_norm_plot[:,index_im_corr_norm[1]]), meancorr, sigmacorr,1,0])
#
#
#     meanraw = sum(x * im_raw_norm_plot[:,index_im_raw_norm[1]]) / sum(im_raw_norm_plot[:,index_im_raw_norm[1]])
#     sigmaraw = np.sqrt(sum(im_raw_norm_plot[:,index_im_raw_norm[1]] * (x - meanraw)**2) / sum(im_raw_norm_plot[:,index_im_raw_norm[1]]))
#
#     parsraw, covraw = curve_fit(f=lor,xdata=x,ydata=im_raw_norm_plot[:,index_im_raw_norm[1]],p0=[max(im_raw_norm_plot[:,index_im_raw_norm[1]]), meanraw, sigmaraw,1,0])
#
#     xfit = np.linspace(-19.5,19.5,2000)*0.195
#     fit_raw = lor(xfit,*parsraw.tolist())
#     fit_corr = lor(xfit,*parscorr.tolist())
#     FWHM_fit_raw = xfit[fit_raw >= 0.5*(np.amax(fit_raw)-np.amin(fit_raw))+np.amin(fit_raw)]
#     FWHM_fit_corr = xfit[fit_corr >= 0.5*(np.amax(fit_corr)-np.amin(fit_corr))+np.amin(fit_corr)]
#
#     # save results for each point in global arrays.
#     results_fit_corr[i, :]=fit_corr
#     results_fit_raw[i, :]=fit_raw
#     results_raw[i, :]=im_raw_norm_plot[:,index_im_raw_norm[1]]
#     results_corr[i, :]=im_corr_norm_plot[:, index_im_corr_norm[1]]
#
#     resultsz_i_corr[i, :]=np.sum(im_corr)
#     resultsz_i_raw[i, :]=np.sum(im_raw)
#
#     resultsz_max_corr_fit[i, :]=np.amax(fit_corr)
#     resultsz_max_raw_fit[i, :]=np.amax(fit_raw)
# # =======================Lines profiles==========================================
# # fig_profiles_z, axs = plt.subplots(2,5,figsize=(20, 20))
# # fig_profiles_z.subplots_adjust(hspace = .5, wspace=.5)
# # axs = axs.ravel()
# #
# # for j in range(len(folders)):
# #
# #
# #     axs[j].plot(xfit,results_fit_raw[j, :],'-', color='darkorange',label = 'aberration fit')
# #     axs[j].plot(xfit,results_fit_corr[j,:],'-', color='darkblue',label = 'correction fit')
# #
# #     axs[j].plot(x,results_raw[j, :],'.', color='darkorange',label = 'aberration')
# #     axs[j].plot(x,results_corr[j,:],'.', color='darkblue',label = 'correction')
# #
# #     # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
# #     # plt.ylabel(r'$Normalized \quad intensity $')
# #
# #     # axs[j].legend(loc='upper left')
# #     axs[j].grid()
# #     axs[j].set_title('z='+str(z[j])+r'$ \mu m$')
# #     # axs[j].set_xticks(range(0, int(np.max(x))+1,1))
# #     axs[j].set_xticks(np.linspace(-4,4,5))
# #     # axs[j].set_yticks(np.linspace(0,1,3))
# #     axs[j].set_yticks(np.linspace(0,1,3))
# #
# #     # axs[j].title.set_text(r'$ z$'+str(z[j])+r'$ \mu m$')
# # # for ax in axs.flat:
# # #     ax.set(xlabel=r'$ Lateral \quad distance \quad (\mu m)$', ylabel=r'$Norm. \quad int. $')
# #
# # # Hide x labels and tick labels for top plots and y ticks for right plots.
# # for ax in axs.flat:
# #     ax.label_outer()
# # fig_profiles_z.text(0.5, 0.01, r'$ Axial \quad distance \quad (\mu m)$', ha='center', fontsize=BIGGER_SIZE)
# # fig_profiles_z.text(0.04, 0.5, r'$Normalized \quad intensity $', va='center', rotation='vertical',fontsize=BIGGER_SIZE)
# # # plt.legend(loc='lower right', mode="expand", ncol=4)
# # axs[6].legend(loc='upper center', bbox_to_anchor=(1, -0.4),fancybox=False, shadow=False, ncol=4)
# # plt.show()
# # fig_profiles_z.savefig(save_path+'total_z_line_profile'+str(folders[i])+ '.png',dpi=300, bbox_inches='tight')
#
#
# # ----------------------plot intensities----------------------------------------
# fig_int_max =plt.figure('intensities', figsize=(10, 3), linewidth=2.0)
# x_p = np.arange(0,len(folders), 1)
# plt.plot(x_p, resultsz_max_raw_fit,'o', color='darkorange',  label = 'raw')
# plt.plot(x_p, resultsz_max_corr_fit,'o', color='darkblue',label = 'corrected')
# plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE),
# plt.ylabel(r'$ I \quad max.\quad norm.$',fontsize= SMALL_SIZE)
# plt.title('Axial',fontsize= SMALL_SIZE)
# plt.xticks(x_p, x_p+1)
# plt.ylim([0,1.2])
# # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
# plt.show()
# fig_int_max.savefig(save_path+'axial_intensities_norm.png',dpi=300, bbox_inches='tight')
#
# fig_int =plt.figure('intensities', figsize=(10, 3), linewidth=2.0)
# x_p = np.arange(0,len(folders), 1)
# plt.plot(x_p, resultsz_i_raw,'o',color='darkorange',  label = 'raw')
# plt.plot(x_p, resultsz_i_corr, 'o', color='darkblue',label = 'corrected')
# plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE)
# plt.title('Axial',fontsize= SMALL_SIZE)
# plt.ylabel(r'$ I \quad (arb. unit)$',fontsize= SMALL_SIZE)
# plt.xticks(x_p, x_p+1)
# plt.ylim([0,np.amax(results_i_corr)+100000])
# # plt.yticks(np.linspace(0,9,4)*100000)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
# plt.show()
# fig_int.savefig(save_path+'axial_intensities.png',dpi=300, bbox_inches='tight')
#
#
#
# fig_int_max_total, axs =plt.subplots(2,1,figsize=(10, 6))
# fig_int_max_total.subplots_adjust(hspace = .2, wspace=.2)
# axs = axs.ravel()
# x_p = np.arange(0,len(folders), 1)
# labels_x=['1','2','3','4','5', '6', '7', '8', '9','10']
# axs[0].plot(x_p, results_max_raw_fit,'o', color='darkorange',  label = 'raw')
# axs[0].plot(x_p, results_max_corr_fit,'o', color='darkblue',label = 'corrected')
# axs[0].set_title('Lateral',fontsize= SMALL_SIZE)
# # axs[0].set_xticks(x_p,labels_x)
# axs[0].set_ylim([0,1.2])
#
# axs[1].plot(x_p, resultsz_max_raw_fit,'o', color='darkorange',  label = 'raw')
# axs[1].plot(x_p, resultsz_max_corr_fit,'o', color='darkblue',label = 'corrected')
# axs[1].set_title('Axial',fontsize= SMALL_SIZE)
# axs[1].set_xticks(x_p,labels_x)
# axs[1].set_ylim([0,1.2])
# # axs[1].legend(loc='lower right', ncol=2, fontsize= SMALL_SIZE)
#
# for ax in axs.flat:
#     ax.label_outer()
# # axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# fig_int_max_total.text(0.5, -0.05, r'$ROIs$', ha='center', fontsize= BIGGER_SIZE)
# fig_int_max_total.text(-0.01, 0.5, r'$ I \quad  max. \quad norm.$', va='center', rotation='vertical', fontsize=BIGGER_SIZE)
# axs[1].legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= BIGGER_SIZE)
# plt.show()
#
# fig_int_max_total.savefig(save_path+'total_intensities_norm.png',dpi=300, bbox_inches='tight')

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
import fnmatch
import os


#-----------------plots_style-------------------------------------------------

plt.style.use('seaborn-white')
#plt.xkcd()
plt.rcParams['image.cmap'] = 'gray'
font = {'family' : 'calibri',
        'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE=40

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#-------------------------Bar plot best weights---------------------------------

#semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Hidde\\'


# save_path=semi_path+'figures\\'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_2\\new\\'
semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\method_validation\\27_01\\analysis\\method\\march_21\\'



# # #-------------------------------------------------------------------------------
# # # ##--------------------------------ROI intensity measurement---------------------
im_ROI_small= np.array(Image.open(semi_path+'ROI_interm005_12_00009.tif'))
min_ROI_small=np.amin(im_ROI_small)
max_ROI_small=np.amax(im_ROI_small)
print('min/max ROI small', min_ROI_small, max_ROI_small, 'shape', im_ROI_small.shape)
# #
# # #------------------Plot igure 2 panel a middle---------------------------------------------------------
fig_1=plt.figure('ROI small', figsize=(30, 30))
plot_roi_small=plt.imshow(im_ROI_small,vmin=min_ROI_small, vmax= max_ROI_small,cmap = 'gray', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_roi_small, ticks=[min_ROI_small, max_ROI_small], orientation="vertical", fraction=0.047, pad=0.01)
# cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
# cbar_ab.set_ticklabels([r'$688$', r'$5894$'])
plt.axis('off')
# plt.show()
fig_1.savefig(save_path+'roi_small.png',dpi=300, bbox_inches='tight')
# # #-------------------------------------------------------------------------------
# # ##--------------------------------ROI diameter/area measurement-----------------
im_ROI_big= np.array(Image.open(semi_path+'im_interm005_12_00009.tif'))
min_ROI_big=np.amin(im_ROI_big)
max_ROI_big=np.amax(im_ROI_big)
center_roi_big=np.unravel_index(np.argmax(im_ROI_big),im_ROI_big.shape)
print('min/max ROI big', min_ROI_big, max_ROI_big,'shape', im_ROI_big.shape)
# radius_roi=20
# im_ROI_big_plot=im_ROI_big[center_roi_big[0]-radius_roi:center_roi_big[0]+radius_roi,center_roi_big[1]-radius_roi:center_roi_big[1]+radius_roi]
# min_ROI_big=np.amin(im_ROI_big_plot)
# max_ROI_big=np.amax(im_ROI_big_plot)
# # #---------------------Plot figure 2 panel a left------------------------------------------------------
fig_2=plt.figure('ROI big', figsize=(30, 30))
plot_roi_big=plt.imshow(im_ROI_big,vmin=min_ROI_big, vmax= max_ROI_big,cmap = 'gray', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_roi_big, ticks=[min_ROI_big, max_ROI_small], orientation="vertical", fraction=0.047, pad=0.01)

# cbar_ab.set_ticklabels([r'$194$', r'$5894$'])
plt.axis('off')
plt.show()
fig_2.savefig(save_path+'roi_big_40.png',dpi=300, bbox_inches='tight')
# # #
# # # #-------------------------------------------------------------------------------
# # # #----------------Roi diameter measurement + threshold---------------------------
im_inter=np.max(im_ROI_big)
#threshold = np.exp(-1)*(im_inter) # old thesh
threshold = np.exp(-1)*(im_inter-min_ROI_big)+min_ROI_big
binary_image = np.ones(np.shape(im_ROI_big))
binary_image[im_ROI_big<threshold] = 0
#diameter
c_y,c_x = np.where(im_ROI_big==im_inter)
diameter = 1/(np.max((np.sum(binary_image[c_y,:]),np.sum(binary_image[:,c_x]))))
#area
area=1/np.sum(binary_image)
# # # #----------------Plot figure 2 panel a right---------------------------
fig_3=plt.figure('ROI big theshold', figsize=(30, 30))
plot_threshold=plt.imshow(binary_image,vmin=0, vmax= 1,cmap = 'gray')
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_threshold, ticks=[0,1], orientation="vertical", fraction=0.047, pad=0.01)

cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_3.savefig(save_path+'roi_big_threshold_40.png',dpi=300, bbox_inches='tight')
# #-------------------------------------------------------------------------------

# # # ##--------------------------------Plot montage ROI -----------------
montage_ROI= np.array(Image.open(semi_path+'montage_ROI.tif'))
min_montage_ROI=np.amin(montage_ROI)
max_montage_ROI=np.amax(montage_ROI)
#
# # # #---------------------Plot montage ROI, figure 2 panel b, bottom ------------------------------------------------------
fig_4=plt.figure('montage ROI', figsize=(30, 30))
plot_montage_ROI=plt.imshow(montage_ROI,vmin=0, vmax= max_montage_ROI,cmap = 'gray')
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower left',width_fraction=0.04,scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
# cbar_ab = plt.colorbar(plot_montage_ROI, ticks=[0, max_montage_ROI], orientation="horizontal", fraction=0.047, pad=0.01)

# cbar_ab.set_ticklabels([r'$0$', r'$5894$'])
plt.axis('off')
plt.show()
fig_4.savefig(save_path+'plot_montage_ROI_final.png',dpi=300, bbox_inches='tight')
#
# # #-------------------------------------------------------------------------------
montage_im_interm= np.array(Image.open(semi_path+'Montage_im_interm.tif'))
min_montage_im_interm=np.amin(montage_im_interm)
max_montage_im_interm=np.amax(montage_im_interm)

# # #---------------------Plot, figure 2 panel b, top ------------------------------------------------------
fig_5=plt.figure('montage binary', figsize=(30, 30))
plot_montage_im_interm=plt.imshow(montage_im_interm,vmin=0, vmax= max_montage_im_interm,cmap = 'gray', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',width_fraction=0.04,scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
# cbar_ab = plt.colorbar(plot_montage_binary, ticks=[0, max_montage_binary], orientation="vertical", fraction=0.047, pad=0.01)

# cbar_ab.set_ticklabels([r'$0$', r'$5894$'])
plt.axis('off')
plt.show()
fig_5.savefig(save_path+'plot_montage_im_interm_final_trial.png',dpi=300, bbox_inches='tight')
#
#
# # #-----------------------plot montage binary  -------------------------------------------------
#
#
montage_binary= np.array(Image.open(semi_path+'Montage_binary.tif'))
min_montage_binary=np.amin(montage_binary)
max_montage_binary=np.amax(montage_binary)

# #---------------------Plot figure 2 panel b, middle ------------------------------------------------------
fig_6=plt.figure('montage binary', figsize=(30, 30))
plot_montage_binary=plt.imshow(montage_binary,vmin=0, vmax= max_montage_binary,cmap = 'gray', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower left',width_fraction=0.04,scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
# cbar_ab = plt.colorbar(plot_montage_binary, ticks=[0, max_montage_binary], orientation="vertical", fraction=0.047, pad=0.01)

# cbar_ab.set_ticklabels([r'$0$', r'$5894$'])
plt.axis('off')
plt.show()
fig_6.savefig(save_path+'plot_montage_binary_final.png',dpi=300, bbox_inches='tight')

# #------------------------import data images
semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\test_gui_AO\\30_03_21\\fep_repetition\\method\\'
best_weights=pickle.load(open(semi_path+'_best_weights.p','rb'))

path_stack_interm= semi_path+'stack_interm\\'
path_stack_roi= semi_path+'stack_roi\\'

im_interm = fnmatch.filter(os.listdir(path_stack_interm), '*.tif')
im_roi = fnmatch.filter(os.listdir(path_stack_roi), '*.tif')

data_im_interm=np.zeros((30,30,len(im_interm)),dtype=np.uint16)
data_im_roi=np.zeros((10,10,len(im_roi)), dtype=np.uint16)
for j in range (len(im_interm)):

    data_im_interm[:,:,j]=np.array(Image.open(path_stack_interm+ im_interm[j]))
    data_im_roi[:,:,j]=np.array(Image.open(path_stack_roi+ im_roi[j]))
#---------------metrics calculations

diameter_metric=np.zeros((len(im_interm)),dtype=np.float64)
comb_ia=np.zeros((len(im_interm)),dtype=np.float64)
comb_id=np.zeros((len(im_interm)),dtype=np.float64)

aniso_ints=pickle.load(open(semi_path+'_ints.p','rb'))
aniso_areas= pickle.load(open(semi_path+'_areas.p','rb'))
# print(aniso_ints.shape, aniso_areas.shape)
area_metric=np.zeros((len(im_interm)),dtype=np.float64)
intensity_metric=np.zeros((len(im_interm)),dtype=np.uint64)
area_metric=aniso_areas[9,:,2]
intensity_metric=aniso_ints[9,:,2]
for j in range (len(im_interm)):
        #max and min calculation intermediate
        min_im_interm = np.min(data_im_interm[:,:,j])
        max_im_interm = np.max(data_im_interm[:,:,j])
        # calculation 2 maxima
        c_y,c_x = np.where(data_im_interm[:,:,j]==max_im_interm)

        threshold = np.exp(-1)*(max_im_interm-min_im_interm)+min_im_interm
        binary_image = np.ones(np.shape(data_im_interm[:,:,j]))
        binary_image[data_im_interm[:,:,j]<threshold] = 0


        im_binary = Image.fromarray(binary_image)
        # im_binary.save(semi_path + 'stack_binary\\'+str(j).zfill(6)+'_binary.tif')

        # metric calculations as function of weights
        # area_metric[j]=1/np.sum(binary_image)
        # intensity_metric[j]=np.sum((data_im_roi[:,:,j])**2)
        diameter_metric[j]=1/(np.max((np.sum(binary_image[c_y,:]),np.sum(binary_image[:,c_x]))))

        comb_ia[j]=intensity_metric[j]*area_metric[j]
        comb_id[j]=intensity_metric[j]*diameter_metric[j]
# #--------------metrics plot + fit --------------------
#
#-------------------------PExperiment parameters  ------------------------------

weights=np.linspace(-3,3,15)

# --------------------Functions to fit------------------------------------------
def lor(x, amp1, cen1, wid1,b):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2))+b*x
def gauss(x, amp1,cen1,sigma1,b):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))+b*x


#Intensity fit
xfit2 = np.linspace(weights[0],weights[-1],1000)
meani = sum(weights * intensity_metric) / sum(intensity_metric)
sigmai = np.sqrt(sum(intensity_metric * (weights - meani)**2) / sum(intensity_metric))
parsi, covi = curve_fit(f=lor,xdata=weights,ydata=intensity_metric,p0=[max(intensity_metric), meani, sigmai,1])
#=============================================================================


#-----------------------------  fit area metric ---------------------
mean_area = sum(weights* area_metric) / sum(area_metric)
sigma_area = np.sqrt(sum(area_metric * (weights - mean_area)**2) / sum(area_metric))
pars_area, cov_area = curve_fit(f=lor,xdata=weights,ydata=area_metric,p0=[max(area_metric), mean_area, sigma_area ,1])


#----------------------------- combined fit intensity area---------------------
meanc_a = sum(weights* comb_ia) / sum(comb_ia)
sigmac_a = np.sqrt(sum(comb_ia* (weights - meanc_a)**2) / sum(comb_ia))
parsc_a, covc_a = curve_fit(f=lor,xdata=weights,ydata=comb_ia,p0=[max(comb_ia), meanc_a, sigmac_a,1])

#----------------------------- combined fit intensity diameter---------------------
meanc_d = sum(weights* comb_id) / sum(comb_id)
sigmac_d = np.sqrt(sum(comb_id* (weights - meanc_d)**2) / sum(comb_id))
parsc_d, covc_d = curve_fit(f=lor,xdata=weights,ydata=comb_id,p0=[max(comb_id), meanc_d, sigmac_d,1])

# =============================================================================
#----------------------------- Plots figure 2 panel c---------------------
fig_metric_ia=plt.figure('Metric  intensity_area', figsize = (10,7),linewidth=2.0)
plt.plot(weights,intensity_metric/np.amax(intensity_metric[-1]),'og')
plt.plot(xfit2,lor(xfit2,*parsi.tolist())/np.max(intensity_metric[-1]),'--g',  label=r'$I^2$')

plt.plot(weights,area_metric/np.max(area_metric[-1]),'or',mfc='none')
plt.plot(xfit2,lor(xfit2,*pars_area.tolist())/np.max(area_metric[-1]),'--r',  label=r'$ A$' )

plt.plot(weights,comb_ia/np.max(comb_ia[-1]), 'sb')
plt.plot(xfit2,lor(xfit2,*parsc_a.tolist())/np.max(comb_ia[-1]),'--b', label=r'$M_2$')


plt.ylabel(r'$Metric \quad (arb. units) $')
plt.xlabel(r'$ Weights \quad (\mu m)$')
plt.title(r'$Z_{12} $')#str(i+3))
plt.grid(True)
# plt.legend()
plt.xlim([-3.2,3.2])
plt.xticks([-3,-2,-1,0,1,2,3])
plt.show()
fig_metric_ia.savefig(save_path+'metric_intensity_area.png',dpi=300, bbox_inches='tight')

#for legend
fig_metric_ia_legend=plt.figure('Metric  intensity_area', figsize = (10,7),linewidth=2.0)
plt.plot(weights,intensity_metric/np.amax(intensity_metric[-1]),'og')
plt.plot(xfit2,lor(xfit2,*parsi.tolist())/np.max(intensity_metric[-1]),'--og',  label=r'$I^2$')

plt.plot(weights,area_metric/np.max(area_metric[-1]),'or')
plt.plot(xfit2,lor(xfit2,*pars_area.tolist())/np.max(area_metric[-1]),'--or',  label=r'$ A$', mfc='none')

plt.plot(weights,comb_ia/np.max(comb_ia[-1]), 'b')
plt.plot(xfit2,lor(xfit2,*parsc_a.tolist())/np.max(comb_ia[-1]),'--sb', label=r'$M_2$')


plt.ylabel(r'$Metric \quad (arb. units) $')
plt.xlabel(r'$ Weights \quad (\mu m)$')
plt.title(r'$Zernike \quad 12 $')#str(i+3))
plt.grid(True)
plt.legend()
plt.xlim([-3.2,3.2])
# plt.xticks([-3,-2,-1,0,1,2,3])
plt.show()
fig_metric_ia_legend.savefig(save_path+'metric_intensity_area_4leg.png',dpi=300, bbox_inches='tight')
#
fig_metric_id=plt.figure('Metric  intensity_diameter', figsize = (10,7),linewidth=2.0)
plt.plot(weights,intensity_metric/np.amax(intensity_metric[-1]),'og',label = r'$I^2$')
plt.plot(xfit2,lor(xfit2,*parsi.tolist())/np.max(intensity_metric[-1]),'--g', label=r'$ fit \quad I^2$')

plt.plot(weights,diameter_metric/np.max(diameter_metric[-1]),'or',label = 'D')


plt.plot(weights,comb_id/np.max(comb_id[-1]), 'ob', label=r'$M_1$')
plt.plot(xfit2,lor(xfit2,*parsc_d.tolist())/np.max(comb_id[-1]),'--b', label=r'$ fit \quad M_1$')

plt.ylabel(r'$ Normalized \quad metric $')
plt.xlabel(r'$ Weights \quad (\mu m)$')
plt.title(r'$Zernike \quad 12 $')#str(i+3))
plt.grid(True)
plt.legend()
plt.xlim([-3.2,3.2])
plt.xticks([-3,-2,-1,0,1,2,3])
plt.show()
fig_metric_id.savefig(save_path+'metric_intensity_diameter.png',dpi=300, bbox_inches='tight')

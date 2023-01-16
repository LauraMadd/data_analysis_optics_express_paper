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
# SMALL_SIZE = 20 #20
# MEDIUM_SIZE = 25
# BIGGER_SIZE=30
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#for totol plot line profiles
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

#-------------------------Paths ---------------------------------

# semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\23_02\\fep\\'
semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\test_gui_AO\\02_03\\fep_fov100\\weighted_nearest\\analysis_w\\'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_5\\'

##-----------------------------------Total images random -------------------------------
###Corrected
total_im_corr= np.array(Image.open(semi_path+'after_correction.tif'))
#max and min corrected raw
min_total_corr=np.amin(total_im_corr)
max_total_corr=np.amax(total_im_corr)
print('max/min corr total',min_total_corr, max_total_corr)

# --Normalization
total_im_corr_norm=(total_im_corr-min_total_corr)/(max_total_corr-min_total_corr)
total_im_corr_norm_plot=total_im_corr_norm[670:1390,670:1390]
min_total_im_corr_norm=np.amin(total_im_corr_norm_plot)
max_total_im_corr_norm=np.amax(total_im_corr_norm_plot)
print(' min/max corr total norm', min_total_im_corr_norm, max_total_im_corr_norm)
####Raw
total_im_raw= np.array(Image.open(semi_path+'before_correction.tif'))
#max and min corrected raw
min_total_raw=np.amin(total_im_raw)
max_total_raw=np.amax(total_im_raw)
print(' min/max  raw total',min_total_raw, max_total_raw)
# --Normalization
total_im_raw_norm=(total_im_raw-min_total_corr)/(max_total_corr-min_total_corr)
total_im_raw_norm_plot=total_im_raw_norm[670:1390,670:1390]
min_total_im_raw_norm=np.amin(total_im_raw_norm_plot)
max_total_im_raw_norm=np.amax(total_im_raw_norm_plot)
print(' min/max raw total norm', min_total_im_raw_norm, max_total_im_raw_norm)

#-----------------------Figure 5 panel b left
fig_1=plt.figure('Aberrated image rand', figsize=(30, 30))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_raw=plt.imshow(total_im_raw_norm_plot, vmin=0, vmax=1,cmap = 'jet' )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_raw, ticks=[0, 1], orientation="vertical", fraction=0.047, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_1.savefig(save_path+'im_raw.png',dpi=300, bbox_inches='tight')
# # # #
# # #figure corr-Figure 5 panel b right
fig_2=plt.figure('Corr  image rand', figsize=(30, 30))
# plot_aberrated=plt.imshow(im_aberrated,cmap = 'jet', )
plot_fep=plt.imshow(total_im_corr_norm_plot, vmin=0, vmax=1,cmap = 'jet', )
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_fep, ticks=[0, 1], orientation="vertical", fraction=0.047, pad=0.01)
#cbar_ab.set_ticklabels([r'$161$', r'$13559$'])
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig_2.savefig(save_path+'im_corr.png',dpi=300, bbox_inches='tight')



# #----------------------------------Roi xy planes -------------------------------

sub_folder='gauss_fit_cgh\\'
folders=os.listdir(semi_path+sub_folder)
# variables to save all data
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

# rois_files=fnmatch.filter(os.listdir('M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\02_02\\beads_0.1_um\\set_2\\stack_20210202-175600\\4analysis\\'), '*.tif')
    data_path=semi_path+sub_folder+folders[i]+'\\'
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
    im_corr_norm=(im_corr-min_corr)/(max_corr-min_corr)
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
    im_raw_norm=(im_raw-min_corr)/(max_corr-min_corr)
    im_raw_norm_plot=im_raw_norm
    # im_raw_norm_plot=im_raw_norm[70:120,70:120]
    min_raw_norm=np.amin(im_raw_norm_plot)
    max_raw_norm=np.amax(im_raw_norm_plot)
    print(' min/max raw_norm', min_raw_norm, max_raw_norm)

    radius_roi=8
    # x=np.linspace(0,radius_roi*2,radius_roi*2)*0.195
    index_im_raw_norm=np.unravel_index(im_raw_norm_plot.argmax(),im_raw_norm_plot.shape)
    print('index im_raw_norm', index_im_raw_norm)

#--- Fit line prifile  + FWHM calculation -------------------------------------
    def lor(x, amp1, cen1, wid1,b,off):
        return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + b*x + off
    def gauss(x, amp1,cen1,sigma1,b):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + b*x

    #--- weights, saved_ints, and xfit2 have to be altered
    # x=np.linspace(0,110,110)*0.195:
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
#lines profiles plot
fig_profiles, axs = plt.subplots(3,4,figsize=(20, 20))
fig_profiles.subplots_adjust(hspace = .1, wspace=.1)
axs = axs.ravel()

for j in range(len(folders)):


    axs[j].plot(xfit,results_fit_raw[j, :],'-', color='darkorange',label = 'aberration fit')
    axs[j].plot(xfit,results_fit_corr[j,:],'-', color='darkblue',label = 'correction fit')

    axs[j].plot(x,results_raw[j, :],'.', color='darkorange',label = 'aberration')
    axs[j].plot(x,results_corr[j,:],'.', color='darkblue',label = 'correction')

    # plt.xlabel(r'$ Lateral \quad distance \quad (\mu m)$')
    # plt.ylabel(r'$Normalized \quad intensity $')

    # axs[j].legend(loc='upper left')
    axs[j].grid()
    axs[j].set_xticks(np.linspace(-4, 4,5))
    axs[j].set_yticks(np.linspace(0,1,3))

# for ax in axs.flat:#     ax.set(xlabel=r'$ Lateral \quad distance \quad (\mu m)$', ylabel=r'$Normalized \quad intensity $')
#
# # Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
fig_profiles.text(0.5, 0.01, r'$ Lateral \quad distance \quad (\mu m)$', ha='center', fontsize= BIGGER_SIZE)
fig_profiles.text(0.04, 0.5, r'$Normalized \quad intensity $', va='center', rotation='vertical', fontsize=BIGGER_SIZE)

# plt.legend(loc='lower right', mode="expand", ncol=4)
axs[9].legend(loc='upper center', bbox_to_anchor=(1, -0.4),fancybox=False, shadow=False, ncol=4)
plt.show()
fig_profiles.savefig(save_path+'total_xy_line_profile_w_bis'+str(folders[i])+ '.png',dpi=300, bbox_inches='tight')
#========================Quantifications
I_improvement_perc=(results_max_corr-results_max_raw)*100/results_max_corr
av_I_lateral_perc=np.average(I_improvement_perc)
std_I_lateral_perc=np.std(I_improvement_perc)
I_improvement_factor=(results_max_corr/results_max_raw)
av_I_lateral_factor=np.average(I_improvement_factor)
std_I_lateral_factor=np.std(I_improvement_factor)
table_results=tabulate([['av_I_Improvement percentage ',av_I_lateral_perc ],['std_I_Improvement percentage ',std_I_lateral_perc],['av_I_Improvement factor ',av_I_lateral_factor ],['std_I_Improvement factor ',std_I_lateral_factor]],headers=['Correction improvement'])

f = open(save_path+'Corrected-improvement-xy.txt', 'w')
f.write(table_results)
f.close()

#----------------------plot intensities, figure 5 panel d ----------------------------------------
fig_int_max =plt.figure('intensities', figsize=(8, 3), linewidth=2.0)
x_p = np.arange(0,len(folders), 1)
plt.plot(x_p, results_max_raw,'o', color='darkorange',  label = 'raw')
plt.plot(x_p, results_max_corr,'o', color='darkblue',label = 'corrected',mfc='none')
plt.xlabel(r'$ROIs$',fontsize= SMALL_SIZE),
plt.ylabel(r'$ I \quad  max. \quad norm.$',fontsize= SMALL_SIZE)
plt.xticks(x_p, x_p+1)
plt.ylim([0,1.2])
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2,fontsize= SMALL_SIZE)
plt.show()
fig_int_max.savefig(save_path+'max_intensities_norm_2D.png',dpi=300, bbox_inches='tight')
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

# #----------------------plot fidelity ----------------------------------------
#===================== reference coords: SLM  coords
coords_ref_temp=np.load(semi_path+"\\Final_coords_holoName.npy")
coords_ref_temp=np.transpose(coords_ref_temp)
n_points=coords_ref_temp.shape[1]

coords_ref=np.zeros((3,n_points))


coords_ref=coords_ref_temp
#=================corr_coords : coordinates adter correction.
coords from image
-----system parameters
cam_h=2048
off_y=cam_h/2
off_x=cam_h/2
pix_size=6.5
M_2=9./300
n_points=coords_ref.shape[1]
defocus_voltage= -0.0154474 *coords_ref[2,0]+ 2.65526
coords_corr_im=np.load(semi_path+"centers_corr_cgh.npy")
coords_raw_im=np.load(semi_path+"centers_raw_cgh.npy")

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
coords_corr_sample[2,:]= 136.684 *(0.609355*defocus_voltage-1.51339)
# #transformation to SLM
camera_to_slm_matrix=np.load('M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\test_gui_AO\\02_03\\t_aff_rand_bigrange_20_fov130\\T_affine.npy')
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
# coords_corr=coords_corr_slm

#calculation accuracy
dst=np.zeros((n_points),dtype=np.float64)
dst[:] =np.sqrt((coords_ref[0,:]-coords_corr[0,:])**2+(coords_ref[1,:]-\
            coords_corr[1,:])**2+(coords_ref[2,:]-coords_corr[2,:])**2)


average_dst=np.average(dst)
sigma_dst=np.std(dst)
# np.save(semi_path+'dst_raw',dst)
dst_raw=np.load(semi_path+'dst_raw.npy')
average_dst_raw=np.average(dst_raw)
sigma_dst_raw=np.std(dst_raw)

fig3 = plt.figure(figsize=(10,3), linewidth=2.0)
x= np.linspace(1,n_points, n_points)
x_ticks=[]
x_ticks=x
plt.plot(x,dst,'o',color='darkblue')
plt.axhline(y=average_dst,color='darkblue', alpha=.5, linestyle='--')
# plt.xticks(x_ticks)
# plt.xlabel(r'$ ROIs $',fontsize= SMALL_SIZE)
plt.tick_params( bottom = False)
plt.ylabel(r'$ Fidelity\quad (\mu m)$',fontsize= SMALL_SIZE)
plt.ylim([0,1.0])
plt.show()
fig3.savefig(save_path+'fidelity_bis.png',dpi=300, bbox_inches='tight')

#plot comparison fidelity
fig4 = plt.figure(figsize=(10,3), linewidth=2.0)
x= np.linspace(1,n_points, n_points)
x_ticks=[]
x_ticks=x
plt.plot(x,dst, 'o',color='darkblue',label= 'corrected')
plt.axhline(y=average_dst,color='darkblue', alpha=.5, linestyle='--')
plt.plot(x,dst_raw, 'o',color='darkorange',label= 'raw')
plt.axhline(y=average_dst_raw,color='darkorange', alpha=.5, linestyle='--')
plt.xticks(x_ticks)
plt.xlabel(r'$ ROIs $',fontsize= SMALL_SIZE)
plt.ylabel(r'$ Fidelity\quad (\mu m)$',fontsize= SMALL_SIZE)
plt.ylim([0,1.0])
plt.legend(loc='upper center', bbox_to_anchor=(0, -0.5),fancybox=False, shadow=False, ncol=2)

plt.show()
fig4.savefig(save_path+'fidelity_comparison.png',dpi=300, bbox_inches='tight')



# -----------------------------------------Phase  maps-----------------------------
# semi_path_zernikes_grid='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\test_gui_AO\\02_03\\fep_fov100\\20210302-135835_aniso\\'
# best_weights_grid=pickle.load(open(semi_path_zernikes_grid+'_best_weights.p','rb'))


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

#------------------Phase maps weighted, figure 5 panel c
semi_path_zernikes='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\test_gui_AO\\02_03\\fep_fov100\\weighted_nearest\\'
best_weights=np.load(semi_path_zernikes+'zernikes_w.npy')

fig_aniso, axs =plt.subplots(3,4,figsize=(30, 30))
fig_aniso.subplots_adjust(hspace = .002, wspace=.002)
axs = axs.ravel()
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

fig_aniso.subplots_adjust(right=0.5)
cbar = plt.colorbar(im, ticks=[-np.pi, np.pi], ax=axs,orientation='vertical',fraction=0.047, pad=0.01 )
cbar.set_ticklabels([r'$- \pi$', r'$\pi$'])
plt.show()
fig_aniso.savefig(save_path+'aniso_phase_weighted.png',dpi=300, bbox_inches='tight' )

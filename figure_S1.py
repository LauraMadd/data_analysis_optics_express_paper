from importlib import reload
import sys
sys.path.append('C:\\Python_scripts\\codes_building_blocks\\building_blocks\\')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
# from mayavi import mlab
from PIL import Image
# import math_functions as mtm
# reload(mtm)
from scipy.integrate import quad
from scipy.spatial import distance
import os

#-------------------------------------------------------------------------------
# matplotlib parameters
plt.style.use('seaborn-white')
#plt.xkcd()
plt.rcParams['image.cmap'] = 'gray'
font = {'family' : 'calibri',
        'weight' : 'normal',
        'size'   : 15}
plt.rc('font', **font)
SMALL_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 40

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

data_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\neuromethods_book_chapter\\aliasing\\experimental\\23_01_20\\luckosz\\grid_21_coma\\3101\\'
save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_S1\\'
#--------------Figure S1, panel b ---------------------------------
####-----------Aberrated image
# image = Image.open(data_path+'grid21_100coma_exp0.5ms_better_2401.tif')
im_aberrated= np.array(Image.open(data_path+'z7_100.tif'))
##---normalization im aberrated
max_im_ab=np.amax(im_aberrated)
min_im_ab=np.amin(im_aberrated)
print('im_ab',max_im_ab,min_im_ab )
im_aberrated_norm= (im_aberrated-min_im_ab)/(max_im_ab-min_im_ab)
im_aberrated_norm_plot=im_aberrated_norm[400:1789,193:1699]
max_im_ab_norm=np.amax(im_aberrated_norm)
min_im_ab_norm=np.amin(im_aberrated_norm)
print('im_ab_norm',max_im_ab_norm,min_im_ab_norm )

#### Original image
im_original= np.array(Image.open(data_path+'no_ab.tif'))
##---normalization im aberrated
max_im_original=np.amax(im_original)
min_im_original=np.amin(im_original)
print('im_ab',max_im_ab,min_im_ab )
im_original_norm= (im_original-min_im_original)/(max_im_original-min_im_original)
im_original_norm_plot=im_original_norm[400:1789,193:1699]
max_im_original_norm_plot=np.amax(im_original_norm_plot)
min_im_original_norm_plot=np.amin(im_original_norm_plot)
print('im_ab_norm',max_im_original_norm_plot,min_im_original_norm_plot )

# Aberrated point
# matrix_roi=np.zeros((300,400),dtype=np.uint16)
# matrix_roi[:,:]=matrix_image[800:1100,400:800]*100
matrix_roi=np.zeros((75,75),dtype=np.uint16)
matrix_roi[:,:]=im_aberrated[925:1000,500:575]*100
# image_roi=Image.fromarray(matrix_roi)
# image_roi.save(save_path+'zoom_roi.tif')



fig1=plt.figure('Grid image aberrated', figsize=(45, 30))
# plt.title('Grid image aberrated ')
plot_aberrated=plt.imshow(im_aberrated_norm_plot, cmap='jet')
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right')
# scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')

plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_aberrated, ticks=[0, max_im_ab_norm], orientation="horizontal", fraction=0.051, pad=0.01)
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig1.savefig(save_path+'grid_aberrated_scale.png',dpi=300, bbox_inches='tight')


fig2=plt.figure('Grid image original ',figsize=(45, 30))
# plt.title('Grid image original ')
plot_original=plt.imshow(im_original_norm_plot, cmap='jet')
# scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right')
plt.gca().add_artist(scalebar)
cbar_ab = plt.colorbar(plot_original, ticks=[min_im_original_norm_plot, max_im_original_norm_plot], orientation="horizontal", fraction=0.051, pad=0.01)
cbar_ab.set_ticklabels([r'$0$', r'$1$'])
plt.axis('off')
plt.show()
fig2.savefig(save_path+'grid_original_scale.png',dpi=300, bbox_inches='tight')



# fig3=plt.figure('ROI aberrated ', figsize=(30,30))
# # plt.title('Zoom in aberrated point')
# plt.imshow(matrix_roi, cmap='jet')
# plt.axis('off')
# plt.show()
# fig3.savefig(save_path+'roi_aberrated.png',dpi=300, bbox_inches='tight')



#-------------------Figure S1, panel a
data_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\neuromethods_book_chapter\\aliasing\\experimental\\23_01_20\\luckosz\\row_of_points\\post\\'
r= 0.195
image = np.array(Image.open(data_path+'Composite0_100_22.png'))
image_plot=image[727:1211,10:2035]
fig4=plt.figure('Row row_of_points')
ax = plt.gca()
scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right')
# scalebar = ScaleBar(0.195, 'um', frameon=False,color='white',location='lower right',scale_formatter=lambda value, unit: f'')
plt.gca().add_artist(scalebar)
# im=ax.imshow(image_plot)
im=ax.imshow(image_plot, extent=(-image_plot.shape[1]/2*r,image_plot.shape[1]/2*r, -image_plot.shape[0]/2*r,image_plot.shape[0]/2*r) )
plt.xlabel(r'$Distance\quad(\mu m)$')
plt.ylabel(r'$Distance\quad(\mu m)$')
# ax.set_xlim(left=-50,right=50)
# ax.set_ylim(bottom=-15,top=5)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
#
# plt.colorbar(im, cax=cax)


plt.show()
fig4.savefig(save_path+'points_row_0_100_coma_scale.png',dpi=300, bbox_inches='tight')

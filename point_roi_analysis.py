# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 13:45:54 2020

@author: Hidde
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import operator
import pandas as pd
import shutil
import os
import fnmatch
# import skimage.external.tifffile as scitif
#-------------------------------------------------------------------------------
'''Variables to change :
 width,type, date, points, ROI_size, add forlder 'data_analysis\\gauss_fit\\'
 to save data manually to the general folder.

'''
#-------------------------------------------------------------------------------
# semi_path = 'M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Hidde\\'
#Method_validation
# semi_path = 'W:\\staff-groups\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Hidde\\'
## NOTE:
# type = 'aniso_sph'
# date = 'precise_22_12'
# general_folder=semi_path+date+'\\'+type+'\\'
# fixed fish 13_01
# general_folder = 'M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\23_02\\fish_side+fep\\comparison_raw_fep_corr_total_corr\\'
# general_folder = 'M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\23_02\\fish_side+fep\\'
# general_folder='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\23_02\\fish_side+fep\\comparison_raw_fep_corr_total_corr\\'
# general_folder='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Hidde\\precise_22_12\\stack\\analysis_stack_0.5\\'
# general_folder = 'M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\02_03\\fep_fov100\\weighted_nearest\\analysis_w\\'
# general_folder ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\Hidde\\03_03\\fep_spatial_baseline\\'
# general_folder ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\16_02\\fish_1\\fep+fish\\'
general_folder ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\02_03\\fep_fov100\\'
# general_folder ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\17_03\\3d_aniso\\random\\'
# general_folder ='W:\\staff-groups\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\17_03\\3d_aniso\\'
# general_folder ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\30_03\\3D_holo\\random\\'
z_min=39.7039
dz_volume=0.49629
dz_xy=0.992597
coords_folder='coords_20210330-184343'
# # z slices
# filen_raw='timelapse_20210330_181930_raw_big\\'
# filen_corr='timelapse_20210330_182514_corr_big\\'
# filen_raw='timelapse_20210330_185506_big_raw\\'
# filen_corr='timelapse_20210330_185326_big_corr\\'

#xy
filen_raw='stack_20210330-184550_raw\\'
filen_corr='timelapse_20210330_184832_corr\\'
# general_folder ='W:\\staff-groups\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\17_03\\3d_aniso\\random\\'

# general_folder='W:\\staff-groups\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\30_03\\fep_repetition\\fep_spatial_base\\'
def find_centers(data):
    '''
    Parameters
    ----------
    data : TYPE
        DESCRIPTION

    Returns
    -------
    centers : ordered horizontally first left -> right
    note: code works because there is a slight tilt in the grid!!!

    '''

    points = 3
    ROI_size =200
    maxindex = np.zeros((points,2))

    #finding centers
    for i in range(points):
        #initial center spot, incase of 0th order nearby
        maxindex[i,0], maxindex[i,1] = np.unravel_index(data.argmax(), data.shape)

        data[int(maxindex[i,0]-ROI_size):int(maxindex[i,0]+ROI_size),int(maxindex[i,1]-ROI_size):int(maxindex[i,1]+ROI_size)] = 0

    #ordering the centers
    centers = np.array(sorted(maxindex, key=operator.itemgetter(0, 1)))
    return centers

def find_centers_vert(data):
    '''
    centers: ordered horizontally first and then  top -> bottom
    note: code only works for a 5x5 grid, since it works around the tilt in the grid in a
    brute force way

    '''
    points =16 # grid spatial frp 17/03
    # points =6 # grid spatial frp 03/03
    # points =3
    # points=1 # 3D dataset
    ROI_size =40
    maxindex = np.zeros((points,2))

    #finding centers
    for i in range(points):
        #initial center spot, incase of 0th order nearby
        maxindex[i,0], maxindex[i,1] = np.unravel_index(data.argmax(), data.shape)

        data[int(maxindex[i,0]-ROI_size):int(maxindex[i,0]+ROI_size),int(maxindex[i,1]-ROI_size):int(maxindex[i,1]+ROI_size)] = 0
        # plt.figure('display mask')
        # plt.imshow(data)
        # plt.show()
    print('max index', maxindex)
    centers = sorted(maxindex, key=operator.itemgetter(0)) # orders centers with respect to coord 0 namely y image
    # centers = sorted(maxindex, key=operator.itemgetter(1)) # orders centers with respect to coord 1 namely x image
    # print('centers', centers)
    ## line of 1 point
    # c_0 = np.array(centers) # sorts along y
    # print('c0' , c_0)
    # centers_final =c_0

    ## line of 2 points
    # c_0 = np.array(sorted(centers[0:2],key=operator.itemgetter(1))) # sorts along y
    # print('c0' , c_0)
    # centers_final =c_0
    # print('centers_final

    ## line of 3 points
    # c_0 = np.array(sorted(centers[0:3],key=operator.itemgetter(1))) # sorts along y
    # print('c0' , c_0)
    # centers_final =c_0
    # # print('centers_final ',centers_final)

    ##grid  16 points
    c_0 = np.array(sorted(centers[0:4],key=operator.itemgetter(1))) # sorts along coord 0 namely y image, left to right
    print('c0', c_0)
    c_1 = np.array(sorted(centers[4:8],key=operator.itemgetter(1)))
    c_2 = np.array(sorted(centers[8:12],key=operator.itemgetter(1)))
    c_3 = np.array(sorted(centers[12:16],key=operator.itemgetter(1)))
    centers_final = np.concatenate((c_0,c_1,c_2,c_3),axis=0)

    ##grid  12 points
    # c_0 = np.array(sorted(centers[0:4],key=operator.itemgetter(1)))
    # c_1 = np.array(sorted(centers[4:8],key=operator.itemgetter(1)))
    # c_2 = np.array(sorted(centers[8:12],key=operator.itemgetter(1)))
    #
    # centers_final = np.concatenate((c_0,c_1,c_2),axis=0)
    ##grid  6 points
    # c_0 = np.array(sorted(centers[0:2],key=operator.itemgetter(1)))
    # c_1 = np.array(sorted(centers[2:4],key=operator.itemgetter(1)))
    # c_2 = np.array(sorted(centers[4:6],key=operator.itemgetter(1)))
    #
    # centers_final = np.concatenate((c_0,c_1,c_2),axis=0)

    ##grid  9 points
    # c_0 = np.array(sorted(centers[0:3],key=operator.itemgetter(1)))
    # c_1 = np.array(sorted(centers[3:6],key=operator.itemgetter(1)))
    # c_2 = np.array(sorted(centers[6:9],key=operator.itemgetter(1)))
    #
    # centers_final = np.concatenate((c_0,c_1,c_2),axis=0)
    return centers_final
def find_centers_vert_bis(data,centers_im):
    '''
    centers: ordered vertically first top -> bottom
    note: code only works for a 5x5 grid, since it works around the tilt in the grid in a
    brute force way

    '''
    # points =9 # grid spatial frp 17/03
    # points =6 # grid spatial frp 03/03
    # points =3
    points=1 # 3D dataset
    ROI_size =30
    maxindex = np.zeros((points,2))
    data_temp=data
    #finding centers
    for i in range(points):
        # print(centers_im)
        #initial center spot, incase of 0th order nearby
        mask = np.zeros(data_temp.shape,np.uint16)
        mask[int(centers_im[1]-ROI_size):int(centers_im[1]+ROI_size),int(centers_im[0]-ROI_size):int(centers_im[0]+ROI_size)] = data_temp[int(centers_im[1]-ROI_size):int(centers_im[1]+ROI_size),int(centers_im[0]-ROI_size):int(centers_im[0]+ROI_size)]
        # im = data_temp[int(centers_im[1]-ROI_size):int(centers_im[1]+ROI_size),int(centers_im[0]-ROI_size):int(centers_im[0]+ROI_size)]
        maxindex[i,0], maxindex[i,1] = np.unravel_index(mask.argmax(), mask.shape)
        # plt.figure('display mask')
        # plt.imshow(mask)
        # plt.show()
        #
        # plt.figure('display im')
        # plt.imshow(data)
        # plt.show()

        # data_temp[int(maxindex[i,0]-ROI_size):int(maxindex[i,0]+ROI_size),int(maxindex[i,1]-ROI_size):int(maxindex[i,1]+ROI_size)] = 0

    centers = sorted(maxindex, key=operator.itemgetter(0))

    ## line of 1 point
    c_0 = np.array(centers) # sorts along y
    # print('c0' , c_0)
    centers_final =c_0

    # print(centers_final)

    return centers_final

# def find_centers_vert_bis(data,centers_im):
#     '''
#     centers: ordered vertically first top -> bottom
#     note: code only works for a 5x5 grid, since it works around the tilt in the grid in a
#     brute force way
#
#     '''
#     # points =9 # grid spatial frp 17/03
#     # points =6 # grid spatial frp 03/03
#     # points =3
#     # points=16 # 3D dataset
#     ROI_size =20
#     maxindex = np.zeros((points,2))
#     data_temp=data
#     #finding centers
#     for i in range(points):
#         print(centers_im)
#         #initial center spot, incase of 0th order nearby
#         mask = np.zeros(data_temp.shape,np.uint16)
#         mask[int(centers_im[1]-ROI_size):int(centers_im[1]+ROI_size),int(centers_im[0]-ROI_size):int(centers_im[0]+ROI_size)] = data_temp[int(centers_im[1]-ROI_size):int(centers_im[1]+ROI_size),int(centers_im[0]-ROI_size):int(centers_im[0]+ROI_size)]
#         # im = data_temp[int(centers_im[1]-ROI_size):int(centers_im[1]+ROI_size),int(centers_im[0]-ROI_size):int(centers_im[0]+ROI_size)]
#         maxindex[i,0], maxindex[i,1] = np.unravel_index(mask.argmax(), mask.shape)
#         # plt.figure('display mask')
#         # plt.imshow(mask)
#         # plt.show()
#         # centers.append(maxindex)
#
#         # plt.figure('display im')
#         # plt.imshow(data)
#         # plt.show()
#
#         # data_temp[int(maxindex[i,0]-ROI_size):int(maxindex[i,0]+ROI_size),int(maxindex[i,1]-ROI_size):int(maxindex[i,1]+ROI_size)] = 0
#
#     centers = sorted(maxindex, key=operator.itemgetter(0))
#     ## line of 1 point
#     # c_0 = np.array(centers) # sorts along y
#     # # print('c0' , c_0)
#     # centers_final =c_0
#     # print(centers_final)
#     c_0 = np.array(sorted(centers[0:2],key=operator.itemgetter(1))) # sorts along y
#     print('c0' , c_0)
#     centers_final =c_0
#     return centers_final

def find_centers_cgh_order(data,centers_im):
    '''
    Find centers of ROIs based on the order of the ROIs in the cgh

    '''
    # points=16 # 3D dataset
    # ROI_size =20 #fov100 grid
    points=3 # 3D dataset
    ROI_size =20 #fov100 grid
    # ROI_size=20
    maxindex = np.zeros((points,2), dtype=np.float32)
    data_temp=data
    #finding centers
    for i in range(points):
        # print(centers_im[:,i])s
        #blank image
        print(i)
        mask = np.zeros(data_temp.shape,np.uint16)
        #fill blank image with one ROI
        mask[int(centers_im[1,i]-ROI_size):int(centers_im[1,i]+ROI_size),int(centers_im[0,i]-ROI_size):int(centers_im[0,i]+ROI_size)] = data_temp[int(centers_im[1,i]-ROI_size):int(centers_im[1,i]+ROI_size),int(centers_im[0,i]-ROI_size):int(centers_im[0,i]+ROI_size)]
        # find coords max
        maxindex[i,0], maxindex[i,1] = np.unravel_index(mask.argmax(), mask.shape)
        plt.figure('display mask')
        plt.imshow(mask)
        plt.show()


    return maxindex
def fcheck_order(data,centers_im):
    '''
    Find centers of ROIs based on the order of the ROIs in the cgh

    '''
    points=16 # 3D dataset
    ROI_size =40
    maxindex = np.zeros((points,2))
    data_temp=data
    # finding centers
    for i in range(points):
        data_temp[int(centers_im[0,i]-ROI_size):int(centers_im[0,i]+ROI_size),int(centers_im[1,i]-ROI_size):int(centers_im[1,i]+ROI_size)]=0
        plt.figure('display mask')
        plt.imshow(data_temp)
        plt.show()
    return
def find_displacement(folder):
    data_original = np.array(Image.open(general_folder+ folder + '\\total_original_syst.tif'))
    data_corrected = np.array(Image.open(general_folder+ folder + '\\total_corrected_syst.tif'))

    pixelsize = .195 #pixel size in
    centers_original = find_centers(data_original)
    centers_corrected = find_centers(data_corrected)

    displacement = (centers_corrected - centers_original)*pixelsize
    return centers_original,centers_corrected,displacement

def ROI_selector_saver(general_folder):
    '''

    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : the following directories are also needed within the save folder
    semi_path_save. Within that directories N folders starting from 0000,0001,...
    are created. Each of these N folders belongs to N ROI's. Within each folder the an image
    of the corrected and the corresponding original ROI are saved. The N folders are
    in order left->right, top->bottom.
    -------
    None.

    '''
    # semi_path_save = 'data_analysis\\gauss_fit\\'
    # semi_path_save = 'W:\\staff-groups\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\30_03\\fep_repetition\\data_analysis\\gauss_fit_baseline\\'
    # semi_path_save ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\02_03\\fep_fov100\\data_analysis\\gauss_fit_new\\'
    semi_path_save = general_folder+'gauss_fit\\'
    # os.mkdir(general_folder+semi_path_save)
    # data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))
    # data_corrected_fcent = np.array(Image.open(general_folder + 'after_correction.tif'))
    #
    data_original = np.array(Image.open(general_folder+ 'before_correction.tif'))
    data_corrected = np.array(Image.open(general_folder + 'after_correction.tif'))

    data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))
    data_corrected_fcent = np.array(Image.open(general_folder + 'after_correction.tif'))
    #
    # data_original = np.array(Image.open(general_folder+ 'total_original_syst.tif'))
    # data_corrected = np.array(Image.open(general_folder + 'total_corrected_syst.tif'))

    pixelsize = .195 #pixel size in

    centers_corrected = find_centers_vert(data_corrected_fcent)
    # centers_original= find_centers_vert(data_original_fcent)
    centers_original=centers_corrected
    # centers_original = centers_corrected
    print('centers_original', centers_original )
    print('centers_corr', centers_corrected )
    ROI_size =20
    for roi in range(0,len(centers_original)):
        if roi < 10:
            roi_folder = '000' + str(roi)
        else:
            roi_folder = '00' + str(roi)
        im_original = Image.fromarray(data_original[int(centers_original[roi,0]-ROI_size):int(centers_original[roi,0]+ROI_size),int(centers_original[roi,1]-ROI_size):int(centers_original[roi,1]+ROI_size)])
        im_corrected = Image.fromarray(data_corrected[int(centers_corrected[roi,0]-ROI_size):int(centers_corrected[roi,0]+ROI_size),int(centers_corrected[roi,1]-ROI_size):int(centers_corrected[roi,1]+ROI_size)])

        if os.path.exists(semi_path_save + roi_folder):
            shutil.rmtree(semi_path_save + roi_folder)
            os.mkdir(semi_path_save + roi_folder,0o755)
            # os.mkdir(general_folder+ semi_path_save + roi_folder)
        else:
            os.mkdir(semi_path_save + roi_folder,0o755)
            # os.mkdir(general_folder+ semi_path_save + roi_folder)


        im_original.save(semi_path_save + roi_folder + '\\original.tif')
        im_corrected.save(semi_path_save + roi_folder + '\\corrected.tif')
    np.save(general_folder+'centers_raw',centers_original )
    np.save(general_folder+'centers_corr', centers_corrected )
    return

def ROI_selector_saver_difficult_detection(general_folder):
    '''

    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : the following directories are also needed within the save folder
    semi_path_save. Within that directories N folders starting from 0000,0001,...
    are created. Each of these N folders belongs to N ROI's. Within each folder the an image
    of the corrected and the corresponding original ROI are saved. The N folders are
    in order left->right, top->bottom.
    -------
    None.

    '''
    # semi_path_save = 'data_analysis\\gauss_fit\\'
    # semi_path_save = 'W:\\staff-groups\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\30_03\\fep_repetition\\data_analysis\\gauss_fit_baseline\\'
    semi_path_save ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\02_03\\fep_fov100\\'

    # os.mkdir(general_folder+semi_path_save)
    # data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))
    # data_corrected_fcent = np.array(Image.open(general_folder + 'after_correction.tif'))
    #
    data_original = np.array(Image.open(general_folder+ 'before_correction.tif'))
    data_corrected = np.array(Image.open(general_folder + 'after_correction.tif'))

    data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))
    data_corrected_fcent = np.array(Image.open(general_folder + 'after_correction.tif'))
    #
    # data_original = np.array(Image.open(general_folder+ 'total_original_syst.tif'))
    # data_corrected = np.array(Image.open(general_folder + 'total_corrected_syst.tif'))

    pixelsize = .195 #pixel size in
    coords_im=np.load(general_folder+'coords_20210302-135328\\coords_cameraholoName.npy')
    print(coords_im, 'shape',coords_im.shape)
    centers_im=np.zeros((coords_im.shape[0],2), dtype=np.float)
    centers_original=np.zeros((coords_im.shape[0],2), dtype=np.float)
    centers_corrected=np.zeros((coords_im.shape[0],2), dtype=np.float)
    # centers_im= coords_im[0,:2]
    for i in range (coords_im.shape[0]):
        centers_im[i,:]=coords_im[i,:2]
        print(centers_im, 'shape',centers_im.shape)
        centers_original[i,:] = find_centers_vert_bis(data_original, centers_im[i])
        centers_corrected[i,:] = find_centers_vert_bis(data_corrected, centers_im[i])

    # ROI_size =20
    for roi in range(0,len(centers_original)):
        if roi < 10:
            roi_folder = '000' + str(roi)
        else:
            roi_folder = '00' + str(roi)
        im_original = Image.fromarray(data_original[int(centers_original[roi,0]-ROI_size):int(centers_original[roi,0]+ROI_size),int(centers_original[roi,1]-ROI_size):int(centers_original[roi,1]+ROI_size)])
        im_corrected = Image.fromarray(data_corrected[int(centers_corrected[roi,0]-ROI_size):int(centers_corrected[roi,0]+ROI_size),int(centers_corrected[roi,1]-ROI_size):int(centers_corrected[roi,1]+ROI_size)])

        if os.path.exists(semi_path_save + roi_folder):
            shutil.rmtree(semi_path_save + roi_folder)
            os.mkdir(semi_path_save + roi_folder,0o755)

        else:
            os.mkdir(semi_path_save + roi_folder,0o755)



        im_original.save(semi_path_save + roi_folder + '\\original.tif')
        im_corrected.save(semi_path_save + roi_folder + '\\corrected.tif')
    return

def ROI_selector_saver_order_cgh(general_folder):
    '''
    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : the following directories are also needed within the save folder
    semi_path_save. Within that directories N folders starting from 0000,0001,...
    are created. Each of these N folders belongs to N ROI's. Within each folder the an image
    of the corrected and the corresponding original ROI are saved. The N folders are
    in order of the order in the cgh
    -------
    None.
    '''
    # semi_path_save ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\02_03\\fep_fov100\\data_analysis\\gauss_fit_cgh\\'
    semi_path_save =general_folder+'data_analysis\\gauss_fit_cgh\\'

    data_original = np.array(Image.open(general_folder+ 'before_correction.tif'))
    data_corrected = np.array(Image.open(general_folder + 'after_correction.tif'))

    data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))
    data_corrected_fcent = np.array(Image.open(general_folder + 'after_correction.tif'))

    pixelsize = .195 #pixel size in
    coords_im=np.load(general_folder+'\\coords_cameraholoName.npy')
    coords_im=np.transpose(coords_im)
    print('coords_im', coords_im, coords_im.shape)
    print(coords_im, 'shape',coords_im.shape)
    # centers_im=np.zeros((coords_im.shape[0],2), dtype=np.float)
    centers_original=np.zeros((coords_im.shape[0],2), dtype=np.float)
    centers_corrected=np.zeros((coords_im.shape[0],2), dtype=np.float)
    centers_original = find_centers_cgh_order(data_original_fcent, coords_im[:2,:])
    centers_corrected= find_centers_cgh_order(data_corrected_fcent, coords_im[:2,:])
    ROI_size =20
    for roi in range(0,len(centers_original)):
        if roi < 10:
            roi_folder = '000' + str(roi)
        else:
            roi_folder = '00' + str(roi)
        im_original = Image.fromarray(data_original[int(centers_original[roi,0]-ROI_size):int(centers_original[roi,0]+ROI_size),int(centers_original[roi,1]-ROI_size):int(centers_original[roi,1]+ROI_size)])
        im_corrected = Image.fromarray(data_corrected[int(centers_corrected[roi,0]-ROI_size):int(centers_corrected[roi,0]+ROI_size),int(centers_corrected[roi,1]-ROI_size):int(centers_corrected[roi,1]+ROI_size)])
        if os.path.exists(semi_path_save + roi_folder):
            shutil.rmtree(semi_path_save + roi_folder)
            os.mkdir(semi_path_save + roi_folder,0o755)
        else:
            os.mkdir(semi_path_save + roi_folder,0o755)

        im_original.save(semi_path_save + roi_folder + '\\original.tif')
        im_corrected.save(semi_path_save + roi_folder + '\\corrected.tif')
    np.save(general_folder+'centers_raw_cgh_bis',centers_original )
    np.save(general_folder+'centers_corr_cgh_bis', centers_corrected )
    return
def check_order(general_folder):
    '''
    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : the following directories are also needed within the save folder
    semi_path_save. Within that directories N folders starting from 0000,0001,...
    are created. Each of these N folders belongs to N ROI's. Within each folder the an image
    of the corrected and the corresponding original ROI are saved. The N folders are
    in order of the order in the cgh
    -------
    None.
    '''
    # semi_path_save ='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\test_gui_AO\\02_03\\fep_fov100\\data_analysis\\gauss_fit_cgh\\'


    data_original = np.array(Image.open(general_folder+ 'before_correction.tif'))
    data_corrected = np.array(Image.open(general_folder + 'after_correction.tif'))

    data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))
    data_corrected_fcent = np.array(Image.open(general_folder + 'after_correction.tif'))

    pixelsize = .195 #pixel size in
    # coords_im=np.load(general_folder+'coords_20210302-135328\\coords_cameraholoName.npy')
    coords_im=np.load(general_folder+'centers_corr.npy')
    # coords_im_bis=np.copy(coords_im)
    coords_im_bis=np.transpose(coords_im)
    print('coords_im', coords_im, coords_im.shape)
    print('coords_im_bis', coords_im_bis, coords_im_bis.shape)
    print(coords_im, 'shape',coords_im.shape)
    fcheck_order(data_original_fcent, coords_im_bis[:2,:])
    fcheck_order(data_corrected_fcent, coords_im_bis[:2,:])


    return coords_im_bis, coords_im


def ROI_selector_saver_from_stack(general_folder):
# def ROI_selector_saver_from_stack(general_folder,semi_path):
    '''

    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : the following directories are also needed within the save folder
    semi_path_save. Within that directories N folders starting from 0000,0001,...
    are created. Each of these N folders belongs to N ROI's. Within each folder the an image
    of the corrected and the corresponding original ROI are saved. The N folders are
    in order left->right, top->bottom.
    -------
    None.

    '''

    print('ciao')
    # semi_path_save = 'data_analysis\\gauss_fit\\'
    # folder_stack_original= semi_path+date+'\\'+'stack\\'+'stack_20201222-153205_aberr_0.5um\\'
    # folder_stack_corrected= semi_path+date+'\\'+'stack\\'+'stack_20201222-152845_aniso_0.5um\\'

    semi_path_save = 'data_analysis\\gauss_fit\\'
    folder_stack_original= general_folder+'\\'+'stack_20210223-184146_raw\\'
    folder_stack_corrected= general_folder+'\\'+'stack_20210223-180256_fish_corr_fep_corr\\'

    # open single image to get centers
    data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))
    data_corrected_fcent = np.array(Image.open(general_folder + 'after_correction.tif'))
     # open single images of z stack in a 3D array
    im_original=fnmatch.filter(os.listdir(folder_stack_original), '*.tif')
    im_corrected=fnmatch.filter(os.listdir(folder_stack_corrected), '*.tif')
    data_original=np.zeros((2048,2048,len(im_original)),dtype=np.uint16)
    data_corrected=np.zeros((2048,2048,len(im_original)), dtype=np.uint16)
    for j in range (len(im_original)):

        data_original[:,:,j]=np.array(Image.open(folder_stack_original+ im_original[j]))
        data_corrected[:,:,j]=np.array(Image.open(folder_stack_corrected+ im_corrected[j]))


    # data_original = np.array(Image.open(general_folder+ 'total_original_syst.tif'))
    # data_corrected = np.array(Image.open(general_folder + 'total_corrected_syst.tif'))

    pixelsize = .195 #pixel size in
    centers_original= find_centers_vert(data_original_fcent)
    centers_corrected = find_centers_vert(data_corrected_fcent)



    ROI_size = 20 # 55
    for j in range (len(im_original)):
        print( 'z', j)


        data_original_temp=data_original[:,:,j]
        data_corrected_temp=data_corrected[:,:,j]

        for roi in range(0,len(centers_original)):
            # if roi < 10:
            #     roi_folder = '000' + str(roi)
            # else:
            #     roi_folder = '00' + str(roi)
            roi_folder=str(roi).zfill(4)
            if j == 0 :
                os.mkdir(folder_stack_original+ semi_path_save + roi_folder,0o755)
                os.mkdir(folder_stack_corrected+ semi_path_save + roi_folder,0o755)

            print('roi', roi, 'z', j)
            im_original = Image.fromarray(data_original_temp[int(centers_original[roi,0]-ROI_size):int(centers_original[roi,0]+ROI_size),int(centers_original[roi,1]-ROI_size):int(centers_original[roi,1]+ROI_size)])
            im_corrected = Image.fromarray(data_corrected_temp[int(centers_corrected[roi,0]-ROI_size):int(centers_corrected[roi,0]+ROI_size),int(centers_corrected[roi,1]-ROI_size):int(centers_corrected[roi,1]+ROI_size)])



            im_original.save(folder_stack_original + semi_path_save + roi_folder +  '\\'+str(j).zfill(6)+'_original.tif')
            im_corrected.save(folder_stack_corrected+ semi_path_save + roi_folder + '\\'+str(j).zfill(6)+'_corrected.tif')
    return


def prova(general_folder):

    pixelsize = .195 #pixel size in
    z_min=44.925
    dz=0.508
    ROI_size = 20 # 55


    semi_path_save = 'data_analysis\\gauss_fit\\'
    folder_stack_original= general_folder+'timelapse_20210317_170443_raw\\'
    folder_stack_corrected= general_folder+'timelapse_20210317_165518_corr\\'
    im_original=fnmatch.filter(os.listdir(folder_stack_original), '*.tiff')
    im_corrected=fnmatch.filter(os.listdir(folder_stack_corrected), '*.tiff')

    data_original=np.zeros((2048,2048),dtype=np.uint16)
    data_corrected=np.zeros((2048,2048), dtype=np.uint16)
    data_original[:,:]=np.array(Image.open(folder_stack_original+ im_original[28]))
    data_corrected[:,:]=np.array(Image.open(folder_stack_corrected+ im_corrected[28]))

    coords_im=np.load(general_folder+'coords_20210317-151117\\coords_cameraholoName.npy')
    z_index=np.zeros((coords_im.shape[0]),dtype=int )

    for i in range(coords_im.shape[0]):
        print(i, 'index')

        z_index[i]=int(round((coords_im[i,2]+z_min)/dz))
        print(z_index[i])

    centers_im= coords_im[0,:2]
    print(centers_im)

    centers_original = find_centers_vert_bis(data_original, centers_im)
    centers_corrected = find_centers_vert_bis(data_corrected, centers_im)

    print(centers_original)

    print(centers_corrected)

    return

def ROI_selector_saver_from3Dstack_to_single_tif(general_folder,z_min,dz_volume,coords_folder, filen_raw, filen_corr):
# def ROI_selector_saver_from_stack(general_folder,semi_path):
    '''

    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : a folder named as the z plane with corrected and original single images
    -------
    None.

    '''

    print('ciao')
    # semi_path_save = 'data_analysis\\gauss_fit\\'
    # folder_stack_original= semi_path+date+'\\'+'stack\\'+'stack_20201222-153205_aberr_0.5um\\'
    # folder_stack_corrected= semi_path+date+'\\'+'stack\\'+'stack_20201222-152845_aniso_0.5um\\'

    semi_path_save = 'data_analysis\\z_stacks_single\\'
    folder_stack_original= general_folder+filen_raw
    folder_stack_corrected= general_folder+filen_corr

     # open single images of z stack in a 3D array
    list_im_original=fnmatch.filter(os.listdir(folder_stack_original), '*.tiff')
    list_im_corrected=fnmatch.filter(os.listdir(folder_stack_corrected), '*.tiff')
    data_original=np.zeros((2048,2048,len(list_im_original)),dtype=np.uint16)
    data_corrected=np.zeros((2048,2048,len(list_im_original)), dtype=np.uint16)
    print('ciao')
    # useful parameters
    pixelsize = .195 #pixel size in
    ROI_size = 20 # 55
    ROI_int = 5
    # impor coorsds
    coords_im=np.load(general_folder+coords_folder+'\\coords_cameraholoName.npy')
    z_index=np.zeros((coords_im.shape[0]),dtype=int )
# z_index_exp_raw=np.zeros((coords_im.shape[0]),dtype=int )
# z_index_exp_corr=np.zeros((coords_im.shape[0]),dtype=int )
#find z focus
    for i in range(coords_im.shape[0]):
        print(i, 'index')

        z_index[i]=int(round((coords_im[i,2]+z_min)/dz_volume))
        # print(z_index)

    # save images in 3D array
    for j in range (len(list_im_original)):
        # print('j', j, 'im_name', im_original[j])

        data_original[:,:,j]=np.array(Image.open(folder_stack_original+ list_im_original[j]))
        data_corrected[:,:,j]=np.array(Image.open(folder_stack_corrected+ list_im_corrected[j]))


    np.save(general_folder+semi_path_save+'data_original.npy', data_original)
    np.save(general_folder+semi_path_save+'data_corrected.npy', data_corrected)
    # data_original=np.load(general_folder+semi_path_save+'data_original.npy')
    # data_corrected=np.load(general_folder+semi_path_save+'data_corrected.npy')

  # find centers for each plane
    for i,z in enumerate(z_index):
        print( 'z',z, 'i', i)


        data_original_temp=data_original[:,:,z]
        data_corrected_temp=data_corrected[:,:,z]



        centers_im= coords_im[i,:2]
        print(centers_im)

        centers_original = find_centers_vert_bis(data_original_temp, centers_im)
        centers_corrected = find_centers_vert_bis(data_corrected_temp, centers_im)

        z_folder=str(round(z_index[i],2)).zfill(4)+'\\'
        # os.mkdir(general_folder  + semi_path_save + z_folder,0o755)
        os.mkdir(general_folder+semi_path_save+z_folder,0o755)
        os.mkdir(general_folder+semi_path_save+z_folder+'original\\',0o755)
        os.mkdir(general_folder+semi_path_save+z_folder+'corrected\\',0o755)
        #loop over images to crop them


        int_corr=np.zeros((len(list_im_original),len(list_im_original)),dtype=float)
        int_raw=np.zeros((len(list_im_original),len(list_im_original)),dtype=float)
        for j in range (len(list_im_original)):

            print( 'slice z_stack', j)

            #select one slice
            data_original_bis=data_original[:,:,j]
            data_corrected_bis=data_corrected[:,:,j]
            # #save single images
            #================================================================================
            im_original = Image.fromarray(data_original_bis[int(centers_original[0,0]-ROI_size):int(centers_original[0,0]+ROI_size),int(centers_original[0,1]-ROI_size):int(centers_original[0,1]+ROI_size)])
            im_corrected = Image.fromarray(data_corrected_bis[int(centers_corrected[0,0]-ROI_size):int(centers_corrected[0,0]+ROI_size),int(centers_corrected[0,1]-ROI_size):int(centers_corrected[0,1]+ROI_size)])

            im_original.save(general_folder + semi_path_save + z_folder + '\\original\\'+str(j).zfill(6)+'_original.tif')
            im_corrected.save(general_folder+ semi_path_save +z_folder + '\\corrected\\'+str(j).zfill(6)+'_corrected.tif')

    return
#==============================================================================
def ROI_selector_saver_from_stack_3Dcgh(general_folder,z_min,dz_volume,coords_folder, filen_raw, filen_corr):
# def ROI_selector_saver_from_stack(general_folder,semi_path):
    '''

    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : a folder named as the z plane with corrected and original z stacks
    -------
    None.

    '''

    print('ciao')
    # semi_path_save = 'data_analysis\\gauss_fit\\'
    # folder_stack_original= semi_path+date+'\\'+'stack\\'+'stack_20201222-153205_aberr_0.5um\\'
    # folder_stack_corrected= semi_path+date+'\\'+'stack\\'+'stack_20201222-152845_aniso_0.5um\\'

    semi_path_save = 'data_analysis\\z_stacks\\'
    # folder_stack_original= general_folder+'timelapse_20210330_181930_raw_big\\'
    folder_stack_original= general_folder+filen_raw
    folder_stack_corrected= general_folder+filen_corr

    # folder_stack_original= general_folder+'timelapse_20210330_182514_corr_big\\'

     # open single images of z stack in a 3D array
    list_im_original=fnmatch.filter(os.listdir(folder_stack_original), '*.tiff')
    list_im_corrected=fnmatch.filter(os.listdir(folder_stack_corrected), '*.tiff')
    data_original=np.zeros((2048,2048,len(list_im_original)),dtype=np.uint16)
    data_corrected=np.zeros((2048,2048,len(list_im_original)), dtype=np.uint16)
    print('ciao')
    # useful parameters
    pixelsize = .195 #pixel size in

    ROI_size = 20 # 55
    ROI_int = 5
    # impor coorsds
    coords_im=np.load(general_folder+coords_folder+'\\coords_cameraholoName.npy')
    z_index_th=np.zeros((coords_im.shape[0]),dtype=int )
    z_index_exp_raw=np.zeros((coords_im.shape[0]),dtype=int )
    z_index_exp_corr=np.zeros((coords_im.shape[0]),dtype=int )
    #find z focus
    for i in range(coords_im.shape[0]):
        print(i, 'index')

        z_index_th[i]=int(round((coords_im[i,2]+z_min)/dz_volume))
        print(z_index_th[i])

    # save images in 3D array
    # for j in range (len(list_im_original)):
    #     # print('j', j, 'im_name', im_original[j])
    #
    #     data_original[:,:,j]=np.array(Image.open(folder_stack_original+ list_im_original[j]))
    #     data_corrected[:,:,j]=np.array(Image.open(folder_stack_corrected+ list_im_corrected[j]))
    #
    # np.save(general_folder+semi_path_save+'data_original.npy', data_original)
    # np.save(general_folder+semi_path_save+'data_corrected.npy', data_corrected)
    # load files
    data_original=np.load(general_folder+semi_path_save+'data_original.npy')
    data_corrected=np.load(general_folder+semi_path_save+'data_corrected.npy')

    int_corr=np.zeros((len(list_im_original),len(list_im_original)),dtype=float)
    int_raw=np.zeros((len(list_im_original),len(list_im_original)),dtype=float)
  # find centers for each plane
    for i,z in enumerate(z_index_th):
        print( 'z',z, 'i', i)


        data_original_temp=data_original[:,:,z]
        data_corrected_temp=data_corrected[:,:,z]



        centers_im= coords_im[i,:2]
        print(centers_im)

        centers_original = find_centers_vert_bis(data_original_temp, centers_im)
        centers_corrected = find_centers_vert_bis(data_corrected_temp, centers_im)

        z_folder=str(round(z_index_th[i],2)).zfill(4)
        os.mkdir(general_folder  + semi_path_save + z_folder,0o755)

        #loop over images to crop them
        stack_corr_list=[]
        stack_raw_list=[]
        stack_corr=np.zeros((len(list_im_original),2048,2048), dtype=np.uint16)
        stack_raw=np.zeros((len(list_im_original),2048,2048), dtype=np.uint16)



        for j in range (len(list_im_original)):

            # print( 'slice z_stack', j)

            #select one slice
            data_original_bis=data_original[:,:,j]
            data_corrected_bis=data_corrected[:,:,j]
            #----------------------save tiff file concatenated
            im_original = Image.fromarray(data_original_bis[int(centers_original[0,0]-ROI_size):int(centers_original[0,0]+ROI_size),int(centers_original[0,1]-ROI_size):int(centers_original[0,1]+ROI_size)])
            im_corrected = Image.fromarray(data_corrected_bis[int(centers_corrected[0,0]-ROI_size):int(centers_corrected[0,0]+ROI_size),int(centers_corrected[0,1]-ROI_size):int(centers_corrected[0,1]+ROI_size)])

            stack_corr_list.append(im_corrected)
            stack_raw_list.append(im_original)

            #measure intensity to check best focus
            int_raw[i,j]=np.sum(data_original_bis[int(centers_original[0,0]-ROI_int):int(centers_original[0,0]+ROI_int),int(centers_original[0,1]-ROI_int):int(centers_original[0,1]+ROI_int)])
            int_corr[i,j]=np.sum(data_corrected_bis[int(centers_original[0,0]-ROI_int):int(centers_original[0,0]+ROI_int),int(centers_original[0,1]-ROI_int):int(centers_original[0,1]+ROI_int)])
            if (j==len(list_im_original)-1):
                stack_corr_list[0].save(general_folder + semi_path_save + z_folder+'\\'+'stack_corr_z'+z_folder+'.tif',compression="none", save_all=True,append_images=stack_corr_list[1:])
                stack_raw_list[0].save(general_folder + semi_path_save + z_folder+'\\'+'stack_raw_z'+z_folder+'.tif',compression="none", save_all=True,append_images=stack_raw_list[1:])

    for i in range(coords_im.shape[0]):
        temp_int_raw=int_raw[i,:]
        temp_int_corr=int_corr[i,:]
        z_index_exp_raw[i]=np.argmax(temp_int_raw)
        z_index_exp_corr[i]=np.argmax(temp_int_corr)
        print(z_index_exp_raw, z_index_exp_corr)
    np.save(general_folder+semi_path_save+'focus_original.npy',z_index_exp_raw)
    np.save(general_folder+semi_path_save+'focus_cottected.npy',z_index_exp_corr)

    return z_index_exp_raw, z_index_exp_corr
#==============================================================================
def truncate_zstack(general_folder, z_min, dz_volume,coords_folder):
# def ROI_selector_saver_from_stack(general_folder,semi_path):
    '''

    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : a tiff file truncated
    -------
    None.

    '''

# useful parameters
    pixelsize = .195 #pixel size in

    ROI_size = 20 # 55
    ROI_int = 5
    slices_number=41
    semi_path_save = 'data_analysis\\z_stacks_truncated\\'
    folder_stacks= general_folder+'data_analysis\\z_stacks_single\\'
    list_folders=os.listdir(folder_stacks)

    # impor coorsds
    coords_im=np.load(general_folder+coords_folder+'\\coords_cameraholoName.npy')
    z_index_th=np.zeros((coords_im.shape[0]),dtype=int )
    z_index_exp_raw=np.zeros((coords_im.shape[0]),dtype=int )
    z_index_exp_corr=np.zeros((coords_im.shape[0]),dtype=int )
    #find z focus
    for i in range(coords_im.shape[0]):
        print(i, 'index')

        z_index_th[i]=int(round((coords_im[i,2]+z_min)/dz_volume))
        print(z_index_th[i])


    #loop over planes subfolders
    for i in range(len(list_folders)):



         # open single images of z stack in a 3D array
        list_im_original=fnmatch.filter(os.listdir(folder_stacks+list_folders[i]+'\\original\\'), '*.tif')
        list_im_corrected=fnmatch.filter(os.listdir(folder_stacks+list_folders[i]+'\\corrected\\'), '*.tif')
        data_original=np.zeros((2048,2048,slices_number),dtype=np.uint16)
        data_corrected=np.zeros((2048,2048,slices_number), dtype=np.uint16)
        # data_original=np.zeros((2048,2048,len(list_im_original)),dtype=np.uint16)
        # data_corrected=np.zeros((2048,2048,len(list_im_original)), dtype=np.uint16)

        # save images in 3D array
        index_start=z_index_th[i]-19
        index_stop=z_index_th[i]+19
        print('Z index', z_index_th[i], index_start, index_stop)
        stack_corr_list=[]
        stack_raw_list=[]
        z_folder=str(round(z_index_th[i],2)).zfill(4)
        os.mkdir(general_folder  + semi_path_save + z_folder,0o755)
        for j in range (index_start,index_stop+1):
            print('j', j, 'im_name', list_im_original[j])

            im_original=Image.open(folder_stacks+list_folders[i]+'\\original\\'+ list_im_original[j])
            im_corrected=Image.open(folder_stacks+list_folders[i]+'\\corrected\\'+ list_im_corrected[j])

            stack_corr_list.append(im_corrected)
            stack_raw_list.append(im_original)

            if (j==index_stop):
                print('loop')
                stack_corr_list[0].save(general_folder + semi_path_save + z_folder+'\\'+'z_corrected.tif',compression="none", save_all=True,append_images=stack_corr_list[1:])
                stack_raw_list[0].save(general_folder + semi_path_save + z_folder+'\\'+'z_original.tif',compression="none", save_all=True,append_images=stack_raw_list[1:])
        # for j in range (len(list_im_original)):
        #     # print('j', j, 'im_name', im_original[j])
        #
        #     data_original[:,:,j]=np.array(Image.open(folder_stack_original+ list_im_original[j]))
        #     data_corrected[:,:,j]=np.array(Image.open(folder_stack_corrected+ list_im_corrected[j]))


    return


#==============================================================================
def ROI_selector_saver_from_stack_to_focalplane(general_folder, z_min, dz_xy,coords_folder, filen_raw, filen_corr):
# def ROI_selector_saver_from_stack(general_folder,semi_path):
    '''

    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : a folder named as the z focal plane with inside corrected and raw image
    -------
    None.

    '''

    print('ciao')
    # semi_path_save = 'data_analysis\\gauss_fit\\'
    # folder_stack_original= semi_path+date+'\\'+'stack\\'+'stack_20201222-153205_aberr_0.5um\\'
    # folder_stack_corrected= semi_path+date+'\\'+'stack\\'+'stack_20201222-152845_aniso_0.5um\\'
    pixelsize = .195 #pixel size in


    ROI_size = 20 # 55
    semi_path_save = 'data_analysis\\gauss_fit\\'
    # folder_stack_original= general_folder+'\\'+'stack_20210223-184146_raw\\'
    # folder_stack_corrected= general_folder+'\\'+'stack_20210223-180256_fish_corr_fep_corr\\'
    folder_stack_original= general_folder+filen_raw
    folder_stack_corrected= general_folder+filen_corr
    # open single image to get centers
    # data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))
    # data_corrected_fcent = np.array(Image.open(general_folder + 'after_correction.tif'))

     # open single images of z stack in a 3D array
    im_original=fnmatch.filter(os.listdir(folder_stack_original), '*.tif')
    im_corrected=fnmatch.filter(os.listdir(folder_stack_corrected), '*.tiff')
    data_original=np.zeros((2048,2048,len(im_original)),dtype=np.uint16)
    data_corrected=np.zeros((2048,2048,len(im_original)), dtype=np.uint16)
    print('ciao')

    coords_im=np.load(general_folder+coords_folder+'\\coords_cameraholoName.npy')
    z_index=np.zeros((coords_im.shape[0]),dtype=int )
    print('n_coords', coords_im.shape)
    for i in range(coords_im.shape[0]):
        print(i, 'index')

        z_index[i]=int(round((coords_im[i,2]+z_min)/dz_xy))
        print(z_index[i])

    # save images in 3D array
    for j in range (len(im_original)):
        # print('j', j, 'im_name', im_original[j])

        data_original[:,:,j]=np.array(Image.open(folder_stack_original+ im_original[j]))
        data_corrected[:,:,j]=np.array(Image.open(folder_stack_corrected+ im_corrected[j]))

    #loop over the z planes
    for i,z in enumerate(z_index):
        print( 'z',z, 'i', i)


        data_original_temp=data_original[:,:,z]
        data_corrected_temp=data_corrected[:,:,z]



        centers_im= coords_im[i,:2]
        print(centers_im)

        centers_original = (data_original_temp, centers_im)
        centers_corrected = (data_corrected_temp, centers_im)
        # print(centers_original.shape, len(centers_original),centers_original)


        z_folder=str(z).zfill(4)

        os.mkdir(general_folder  + semi_path_save + z_folder,0o755)

        # im_original = Image.fromarray(data_original_temp[int(centers[1]-ROI_size):int(centers[1]+ROI_size),int(centers[0]-ROI_size):int(centers[0]+ROI_size)])
        # im_corrected = Image.fromarray(data_corrected_temp[int(centers[1]-ROI_size):int(centers[1]+ROI_size),int(centers[0]-ROI_size):int(centers[0]+ROI_size)])

        im_original = Image.fromarray(data_original_temp[int(centers_original[0,0]-ROI_size):int(centers_original[0,0]+ROI_size),int(centers_original[0,1]-ROI_size):int(centers_original[0,1]+ROI_size)])
        im_corrected = Image.fromarray(data_corrected_temp[int(centers_corrected[0,0]-ROI_size):int(centers_corrected[0,0]+ROI_size),int(centers_corrected[0,1]-ROI_size):int(centers_corrected[0,1]+ROI_size)])


        #
        # im_original.save(general_folder + semi_path_save + z_folder +  '\\'+str(j).zfill(6)+'_original.tif')
        # im_corrected.save(general_folder+ semi_path_save + z_folder + '\\'+str(j).zfill(6)+'_corrected.tif')

        im_original.save(general_folder + semi_path_save + z_folder +  '\\'+'original.tif')
        im_corrected.save(general_folder+ semi_path_save + z_folder + '\\'+'corrected.tif')
    return

def ROI_selector_saver_from_stack_to_best_focalplane(general_folder,coords_folder, filen_raw, filen_corr):
# def ROI_selector_saver_from_stack(general_folder,semi_path):
    '''

    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : a folder named as the z focal plane with inside corrected and raw image
    -------
    None.

    '''

    print('ciao')
    # semi_path_save = 'data_analysis\\gauss_fit\\'
    # folder_stack_original= semi_path+date+'\\'+'stack\\'+'stack_20201222-153205_aberr_0.5um\\'
    # folder_stack_corrected= semi_path+date+'\\'+'stack\\'+'stack_20201222-152845_aniso_0.5um\\'
    pixelsize = .195 #pixel size in


    ROI_size = 20 # 55
    semi_path_save = 'data_analysis\\best_gauss_fit_tris\\'
    # folder_stack_original= general_folder+'\\'+'stack_20210223-184146_raw\\'
    # folder_stack_corrected= general_folder+'\\'+'stack_20210223-180256_fish_corr_fep_corr\\'
    folder_stack_original= general_folder+filen_raw
    folder_stack_corrected= general_folder+filen_corr
    # open single image to get centers
    # data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))
    # data_corrected_fcent = np.array(Image.open(general_folder + 'after_correction.tif'))

     # open single images of z stack in a 3D array
    im_original=fnmatch.filter(os.listdir(folder_stack_original), '*.tif')
    im_corrected=fnmatch.filter(os.listdir(folder_stack_corrected), '*.tiff')
    data_original=np.zeros((2048,2048,len(im_original)),dtype=np.uint16)
    data_corrected=np.zeros((2048,2048,len(im_original)), dtype=np.uint16)
    print('ciao')

    coords_im=np.load(general_folder+coords_folder+'\\coords_cameraholoName.npy')
    # print('coords_im', coords_im, coords_im.shape)
    # coords_im=np.transpose(coords_im)
    centers_original=np.zeros((coords_im.shape[0],2), dtype=np.float)
    centers_corrected=np.zeros((coords_im.shape[0],2), dtype=np.float)


    # z_index=[9,14,19,27,35,44,49,52,64,66]
    # z_index=[15,16,24,32,37,43,48,54,62,63]
    z_index=[[11,12,13,14,15],[15,16,17,18],[23,24,25,26],[31,32,33,34],[36,37,38],[42,43,44],[47,48,49],[52,53,54,55],[61,62,63,64],[62,63,64,65,66,67]]
    centers_original_list=[[],[],[],[],[],[],[],[],[],[]]
    centers_corrected_list=[[],[],[],[],[],[],[],[],[],[]]


    # save images in 3D array
    for j in range (len(im_original)):
        # print('j', j, 'im_name', im_original[j])

        data_original[:,:,j]=np.array(Image.open(folder_stack_original+ im_original[j]))
        data_corrected[:,:,j]=np.array(Image.open(folder_stack_corrected+ im_corrected[j]))

    #loop over the z planes

    for i,z in enumerate(z_index):
        print( 'z',z, 'i', i)
        roi_folder='roi_'+str(i)+'\\'
        # os.mkdir(general_folder +semi_path_save+roi_folder)
        for k_index, k in enumerate (z):
            data_original_temp=data_original[:,:,z[k_index]]
            data_corrected_temp=data_corrected[:,:,z[k_index]]



            centers_im=coords_im[i,:2]
            # print(centers_im)
            centers_original=find_centers_vert_bis(data_original_temp, centers_im)
            centers_corrected=find_centers_vert_bis(data_corrected_temp, centers_im)
            centers_original_list[i].append(centers_original)
            centers_corrected_list[i].append(centers_corrected )
            # print(centers_original.shape, len(centers_original),centers_original)


            # z_folder=str(z[k_index]).zfill(4)
            #
            # os.mkdir(general_folder+semi_path_save+roi_folder+z_folder,0o755)
            #
            #
            #
            # im_original = Image.fromarray(data_original_temp[int(centers_original[0,0]-ROI_size):int(centers_original[0,0]+ROI_size),int(centers_original[0,1]-ROI_size):int(centers_original[0,1]+ROI_size)])
            # im_corrected = Image.fromarray(data_corrected_temp[int(centers_corrected[0,0]-ROI_size):int(centers_corrected[0,0]+ROI_size),int(centers_corrected[0,1]-ROI_size):int(centers_corrected[0,1]+ROI_size)])
            #
            #
            #
            # im_original.save(general_folder + semi_path_save+roi_folder+z_folder +'\\'+'original.tif')
            # im_corrected.save(general_folder+ semi_path_save+roi_folder+z_folder +'\\'+'corrected.tif')
    print('original', centers_original_list, 'corr:', centers_corrected_list)
    centers_original=np.asarray(centers_original_list,dtype=object)
    centers_corrected=np.asarray(centers_corrected_list,dtype=object)
    np.save(general_folder+'centers_raw_cgh_tris',centers_original )
    np.save(general_folder+'centers_corr_cgh_tris', centers_corrected )
    return
def roi_Iweig(folder,file):
    data_centers = np.array(Image.open(general_folder+file))
    data = np.array(Image.open(general_folder+file))
    centers = find_centers_vert(data_centers)
    roi_intensity = np.zeros(len(centers))
    for row in range(0,len(centers)):
        roi_intensity[row] = data[(int(centers[row,0]),int(centers[row,1]))]

    roi_weights = roi_intensity / data.max()
    roi_weights_inverse = 1/roi_weights
    roi_weights_inverse = roi_weights_inverse/roi_weights_inverse.max()
    return centers,roi_weights_inverse


def sim_metric(general_folder):
    '''

    Parameters
    ----------
    folder : location where the corrected and original image are saved

    Returns : the following directories are also needed within the save folder
    semi_path_save. Within that directories N folders starting from 0000,0001,...
    are created. Each of these N folders belongs to N ROI's. Within each folder the an image
    of the corrected and the corresponding original ROI are saved. The N folders are
    in order left->right, top->bottom.
    -------
    None.

    '''
    import time

    data_original = np.array(Image.open(general_folder+ 'before_correction.tif'))

    data_original_fcent = np.array(Image.open(general_folder+ 'before_correction.tif'))


    pixelsize = .195 #pixel size in

    # centers_corrected = find_centers_vert(data_corrected_fcent)
    centers_original= find_centers_vert(data_original_fcent)
    # centers_original=centers_corrected
    # centers_original = centers_corrected
    centers_corrected=centers_original
    data_corrected=data_original
    print('centers_original', centers_original )
    # print('centers_corr', centers_corrected )
    ROI_size =20
    rad_ROI=5
    ROI=np.zeros((4), dtype = int)
    t0=time.clock()
    for roi in range(0,len(centers_original)):

        im_corrected = Image.fromarray(data_corrected[int(centers_corrected[roi,0]-ROI_size):int(centers_corrected[roi,0]+ROI_size),int(centers_corrected[roi,1]-ROI_size):int(centers_corrected[roi,1]+ROI_size)])
        ROI[:]=([int(centers_corrected[roi,0]-rad_ROI),int(centers_corrected[roi,0]+rad_ROI),int(centers_corrected[roi,1]-rad_ROI),int(centers_corrected[roi,1]+rad_ROI)])


        ROI_im=data_corrected[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        # ROI_im.save(save_path+'ROI_interm'+str(n).zfill(3)+'_'+str(i)+'_'+str(j).zfill(5)+'.tif')

        intensity = np.sum(data_corrected[ROI[0]:ROI[1], ROI[2]:ROI[3]])


        #calculation of diameter for diameter metric
                            #apply threshold
        im_inter = np.max(data_corrected[:,:])
        #location of maximum
        c_y,c_x = np.where(data_corrected==im_inter)
        threshold = np.exp(-1)*im_inter

        binary_image = np.ones(np.shape(data_corrected))
        binary_image[data_corrected[:,:]<threshold] = 0

        #actually inverse of diameter to create a maximum for smallest spot size
        #it takes the diameter as the major axis, so here the maximum of either the binary image summed
        #in y direction around maximum or in x direction around maximum
        diameter = 1/(np.max((np.sum(binary_image[c_y,:]),np.sum(binary_image[:,c_x]))))
        tempo=time.clock()-t0

    return tempo

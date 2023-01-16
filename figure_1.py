#aniso_best_weights# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from scipy.special import factorial as fact
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from SLMcontroller import player,hologram
from importlib import reload

plt.style.use('seaborn-white')
#plt.xkcd()
plt.rcParams['image.cmap'] = 'gray'
font = {'family' : 'calibri',
        'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#-----------------------------Functions--------------------------------

#-------------------Aberratios-----------------------------------------

#--------------------Helmoltz modes
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

def HelmIndex(n,m):
    if n >= m:
        ind = n**2 + m
    else:
        ind = (m+1)**2 - 1 - n
    return ind

#-------------------------- Generation bases
def nollIndex(n,m):
    return np.sum(np.arange(n+1))+np.arange(-n,n+1,2).tolist().index(m)


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


#-------------------------------------------------------------------------------
#---------------------------------CGH-------------------------------------------
def cgh_calculation(coords,aberrations, intensities, save_path, holo_name):


    SLM=player.fromFile("Meadowlark_test_@sample.slm")
    holo=hologram(coords,intensities,aberrations,SLM,0,0)

        #holo=hologram(coords,None,SLM,0,0)
    holo.compute("CSWGS",100,0.05)
        #holo.compute("RS")
    holo.wait()

    np.save(save_path+holo_name,holo.phase)
    np.save(save_path+'coords_'+holo_name,coords)
    return holo.phase

#---------------------------------Simulation focal plane-----------------------
def simulate_focal_plane(intensity,phase,z_m,focal,gamma):

    xc,yc=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))
    xc=xc*res*pixSize/2.0
    yc=yc*res*pixSize/2.0
    k = np.pi*2/lam
    propagator =  k*z_m/(2*focal**2)*(xc**2+yc**2)
    phase = (phase)
    pupil_field=intensity*np.exp(1j*phase)*np.exp(-1j*propagator)
    focal_plane_field=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupil_field)))

    return (np.abs(focal_plane_field)**2)**(gamma)


def simulate_focal_plane_bis(intensity,phase,z_m,focal,gamma):

    xc,yc=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))
    xc=xc*res*pixSize/2.0
    yc=yc*res*pixSize/2.0
    k = np.pi*2/lam
    propagator =  k*z_m/(2*focal**2)*(xc**2+yc**2)
    tot_phase = (phase+propagator)
    pupil_field=intensity*np.exp(-1j*tot_phase)

    focal_plane_field=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupil_field)))
    return (np.abs(focal_plane_field)**2)**(gamma)


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------



#------------------------------CGH calculation ---------------------------------------
#---------------Paths
if __name__ == '__main__':
    semi_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\DATA\\Laura\\adaptive_optics\\simulation_ao_corrections\\04_12_20\\'
    save_path='M:\\tnw\\ist\\do\\projects\\Zebrafish\\Laura\\manuscripts\\spatially precise manipulations\\py_figures\\figure_2\\'
    # holo_name='holotest'
    # save_path='C:\\Users\\31681\\Desktop\\MEP_Carroll_Lab_\\data\\data_analysis-master\\'
    # holo_name='holo_z7_intermediatefov'
    c=np.load(semi_path+'coords_holo_z7_intermediatefov.npy')
    # holo_name='holo_z7_noab_intermediatefov'
    holo_name_iso='holo_z7_iso_intermediatefov'
    # holo_name_aniso='holo_z7_intermediatefov'
    #------------system parameters in meters
    res = int(1152)
    pixSize = 1.8399999999999997e-05
    focal = 0.0092
    lam = 8.000000000000001e-07
    gamma = 1/2
    #----------------------------Define coords CGH
    n_points = 16
    n_coeff=21
    coords=np.zeros((n_points,3))
    # xc,yc=np.meshgrid(np.linspace(-60,90,4),np.linspace(-75,75,4)) # big fov
    # xc,yc=np.meshgrid(np.linspace(0,20,4),np.linspace(0,20,4)) # small fov
    xc,yc=np.meshgrid(np.linspace(0,40,4),np.linspace(0,40,4)) # intermediate fov
    z=7
    aberrations=[]
    one_aberration=np.zeros(n_coeff)
    intensities = np.ones(n_points)

    for i in range(coords.shape[0]):
        if i < 30:
            one_aberration[i+3] = 0
        aberrations.append(one_aberration)
        one_aberration=np.zeros(n_coeff)


    coords[:,0]=xc.flatten()-10
    coords[:,1]=yc.flatten()-10
    coords[:,2]=z

    # Choose here between calculation/ loading CGH phase
    # calculation on the fly
    # holophase=cgh_calculation(coords,aberrations,intensities,save_path,holo_name)

    # import pre-alculated CGH
    holophase = np.load(semi_path+holo_name_iso+'.npy')
    #---------------------------Simulation focal plane + phase on SLM----------------
    phase = holophase[:,384:1536]*(2*np.pi)/(2**16) # consider just pupil

    xc,yc=np.meshgrid(np.linspace(-1.0,1.0,res),np.linspace(-1.0,1.0,res))

    r_unit = np.sqrt(xc**2+yc**2)
    out_pupil = r_unit > 1
    xc=xc*res*pixSize/2.0
    yc=yc*res*pixSize/2.0

    r=np.sqrt(xc**2+yc**2)

    phase = phase.astype('float64')
    intensity = np.ones(np.shape(phase))
    intensity[out_pupil] = 0
#Simualtion focal pount
    z_m = z*10**-6
    focus = simulate_focal_plane(intensity,phase,z_m,focal,gamma)
    focus = (focus-np.amin(focus))/(np.amax(focus)-np.amin(focus))

##---------Plot focal point with iso aberrations, figure 1 , panel b and c depending on the file loaded
#To get aniso aberrated foci change CGH loaded above to holo_name_aniso
    fig1 = plt.figure('focus field', figsize=(30, 30) )
    # focus_plot=focus
    # focus_plot[focus_plot==np.amin(focus)]=0
    im_focus_field=plt.imshow(focus[520:688,514:692],vmin=0, vmax=1,cmap='jet')
    cbar1= plt.colorbar(im_focus_field,ticks=[0, 1],orientation="horizontal", fraction=0.048, pad=0.01)
    cbar1.set_ticklabels([r'$0$',r' $1$ '])
    plt.axis('off')
    plt.show()
    fig1.savefig(save_path+'focus_field_iso.png',dpi=300, bbox_inches='tight')

##------------------Plot SLM total phase
    #check_values
    # print('max phase', np.amax(phase), 'min phase', np.amin(phase))
    # phase_plot = 2*np.pi*(phase-np.min(phase))/(np.max(phase)-np.min(phase))-np.pi
    # print('max phase_plot', np.amax(phase_plot), 'min phase_plot', np.amin(phase_plot))
    # phase_plot[out_pupil] = np.nan

    # fig2 = plt.figure('SLM phase', figsize=(45, 30))
    # im_phase_plot=plt.imshow(phase_plot, cmap='Greys' )
    #
    # cbar2  = plt.colorbar(im_phase_plot, ticks=[-np.pi, np.pi],orientation="horizontal", fraction=0.048, pad=0.01)
    # cbar2.set_ticklabels([r'$- \pi$', r'$\pi$'])
    #
    # plt.axis('off')
    # plt.show()
    # fig2.savefig(save_path+'pos_phase.png',dpi=300, bbox_inches='tight' )
    #-----------------------Simualtion z stack--------------------------
    # z_min=4
    # z_max=10
    # dz=5
    # n_planes=dz
    # z_m_stack = 10**-6*np.linspace(z_min,z_max,dz)
    # z_stack = np.zeros((res,res,n_planes))
    #
    # for i in range(n_planes):
    #     z_m = z_m_stack[i]
    #     field = simulate_focal_plane(intensity,phase,z_m,focal,gamma)
    #     field = field
    #     z_stack[:,:,i] = field
    #
    # z_stack = z_stack/np.amax(z_stack)
    # #plot z stack
    # a = np.unravel_index(z_stack.argmax(), z_stack.shape)
    # z_field = np.transpose(z_stack[a[0],500:650,:])
    #
    # fig3 = plt.figure('z_stack')
    # plt.imshow(z_field,aspect=2,cmap='magma')
    # plt.show()
##---------------------------Simulation positional phase maps---------------
    f_x = coords[:,0]*(10**(-6))
    f_y = coords[:,1]*(10**(-6))
    f_z = coords[:,2]*(10**(-6))

    # initialize total phase for all the 16 points
    total_phase = np.zeros((res,res,n_points))

    fig4, axs4 = plt.subplots(4,4,figsize=(30, 30))
    fig4.subplots_adjust(hspace = .1, wspace=.002)

    axs4 = axs4.ravel()

    i=0
    for j in range(n_points):
        sim_pos = 2.0*np.pi/(lam*(focal))*(f_x[i]*xc+f_y[i]*yc)+(np.pi*f_z[i])/(lam*(focal)**2)*(xc**2+yc**2)
        pos_phase = np.angle(np.exp(1j*sim_pos))+np.pi
        pos_phase[out_pupil] = 0
        total_phase[:,:,j] += pos_phase

        #phase wrapping between -pi and pi

        pos_phase_plot = 2*np.pi*(pos_phase-np.min(pos_phase))/(np.max(pos_phase)-np.min(pos_phase))-np.pi
        pos_phase_plot[out_pupil] = np.nan

        im = axs4[i].imshow(pos_phase_plot,cmap='Greys',vmin=-np.pi,vmax=np.pi)
        axs4[i].axis('off')
        i += 1

    fig4.subplots_adjust(right=0.5)
    cbar =plt.colorbar(im,ticks=[-np.pi, np.pi],ax=axs4,orientation='horizontal',fraction=0.06, pad=0.01)
    cbar.set_ticklabels([r'$- \pi$',r'$ \pi $'])
    plt.show()
    # fig4.savefig(save_path+'pos_phase_Greys.png',dpi=300, bbox_inches='tight' )

    #----------------------- Simulation Aberrations Phase-----------------

    base = "Zernike"
    order = 5

    single_ab=4

    #generate aberrations dataset
    zern_dataset = generateAberrationDataset(res,base,order)
    # ---  SIMULATION create aberrations manually
    #coefficients
    aberrations = []
    one_aberration=np.zeros(n_coeff)
    for i in range(n_points):
        if i < 30:
            one_aberration[i+3] = 1
        aberrations.append(one_aberration)
        one_aberration=np.zeros(n_coeff)
    sim_best_weights = np.stack(aberrations,axis=0)



    # ---ISOPLANATIC case resizes a single phase maps to apply to all points in grid
    if np.shape(sim_best_weights)[0] == 13:
        print('ISO')
        best_weights = np.ones((16,np.shape(sim_best_weights)[0]))
        best_weights[:] = sim_best_weights
        sim_best_weights = best_weights


    fig5, axs5 = plt.subplots(4,4,figsize=(30, 30))
    fig5.subplots_adjust(hspace =.002, wspace=.002)
    axs5 = axs5.ravel()
    i=0
    for j in range (sim_best_weights.shape[0]):
        coeffs = sim_best_weights[j,:]
        #input in nm
        pointabberation = getAberrationPhase(zern_dataset,lam*10**9,coeffs)*(lam*10**9/1000.0)/2**16*(2*np.pi)
        pointabberation[out_pupil]=0
        total_phase[:,:,j] += pointabberation

        pointabberation_plot = 2*np.pi*(pointabberation-np.min(pointabberation))/(np.max(pointabberation)-np.min(pointabberation))-np.pi

        pointabberation_plot[out_pupil] = np.nan

        im =axs5[i].imshow(pointabberation_plot,cmap = 'bwr',extent=[-1,1,-1,1])
        axs5[i].axis('off')
        i += 1


##--------------Plot aniso phase, figure 1 panel c
    fig5.subplots_adjust(right=0.5)
    cbar = fig5.colorbar(im,ticks=[-np.pi, np.pi], ax=axs5,orientation='horizontal',fraction=0.06, pad=0.01)
    cbar.set_ticklabels([r'$- \pi$',r'$ \pi $'])
    plt.show()
    # fig5.savefig(save_path+'aniso_phase_bwr.png',dpi=300, bbox_inches='tight' )

#     #check values
#     print('max pointabberation', np.amax(pointabberation), 'pointabberation', np.amin(pointabberation))
#     print('max pointabberation_plot', np.amax(pointabberation_plot), 'pointabberation_plot', np.amin(pointabberation_plot))
#
##-------------------Single phase map-------------------------------------------
#
    coeffs = sim_best_weights[single_ab,:]
    zern = ((getAberrationPhase(zern_dataset,lam*10**9,coeffs)*2*np.pi/(2**16)))
    zern_zero = zern == 0
    #phase wrapping between ) and pi
    zern = 2*np.pi*(zern-np.min(zern))/(np.max(zern)-np.min(zern))-np.pi
    zern[out_pupil] = np.nan
##-----------------Plot single phase map, figure 1 panel b
    fig6 = plt.figure('single phase map',figsize=(30, 30))
    imsingle = plt.imshow(zern,cmap = 'bwr',extent=[-1,1,-1,1])
    # imsingle = plt.imshow(zern,cmap = 'Greys')
    plt.axis('off')
    cbar_single = plt.colorbar(imsingle, ticks=[-np.pi, np.pi],orientation='horizontal',fraction=0.047, pad=0.01)
    cbar_single.set_ticklabels([r'$- \pi$', r'$\pi$'])
    plt.show()
    # fig6.savefig(save_path+'iso_phase_bwr_bis.png',dpi=300, bbox_inches='tight' )
    # #------------------- simulate position + aberration phase
    # fig5, axs = plt.subplots(4,4,figsize=(45, 30), num='total phase map')
    # fig5.subplots_adjust(hspace = .5, wspace=.002)
    # axs = axs.ravel()
    #
    # i=0
    # for j in range (sim_best_weights.shape[0]):
    #     total_phase_map = total_phase[:,:,j]
    #     im = axs[i].imshow(total_phase_map,cmap = 'jet',extent=[-1,1,-1,1])
    #     axs[i].axis('off')
    #     i += 1
    #
    # cbar = fig5.colorbar(im, cax=cbar_ax,ticks=[-np.pi, np.pi])
    # cbar.set_ticklabels([r'$- \pi$',r'$ \pi $'])
    # # cbar.ax.tick_params(labelsize=20)
    #
    # ints_total = np.zeros((res,res,n_points))
    # for iii in range(n_points):
    #     ints_total[:,:,iii] = intensity
    #
    # # Plot phase obtained adding positional and aberration phase maps
    # fig7 = plt.figure('SLM_phase total')
    # SLM_phase_total = np.sum(ints_total*np.exp(1j*(total_phase+np.pi)),axis=2)
    # plt.imshow(np.angle(SLM_phase_total),cmap='jet')
    # plt.axis('off')
    # plt.colorbar()
    #
    # # Plot z stack
    # z_m = z*10**-6
    # focus_t = simulate_focal_plane(intensity,np.angle(SLM_phase_total),z_m,focal,gamma)
    # focus_t = focus_t/np.amax(focus_t)
    #
    #
    # fig6 = plt.figure('focus field total')
    # plt.imshow(focus_t,cmap='jet')
    # plt.colorbar()
    # plt.axis('off')
    # plt.show()

    # im_t = Image.fromarray(focus_t)
    # im_t.save('focus_t.tif')


    # #-----------------------generate HNM directly

    # fig, axs = plt.subplots(4,4,figsize=(15, 10))
    # fig.subplots_adjust(hspace = .5, wspace=.001)
    # i = 0

    # axs = axs.ravel()
    # for n in range(0,4):
    #     for m in range(0,4):
    #         hnm=h_mode(n,m,rho,phi)
    #         hnm_mode = np.angle(hnm)
    #         hnm_mode[rho_limit] = np.nan
    #         im = axs[i].imshow(hnm_mode,cmap = 'inferno',extent=[-1,1,-1,1])
    #         axs[i].set_title('n:'+str(n)+' m:'+str(m))
    #         i += 1

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.71])
    # fig.colorbar(im, cax=cbar_ax)

    # #-----------------------generate HNM via SLM.controller

    # res = 200
    # base = "Helmholtz"
    # order = 4
    # coeffs = np.zeros(16)
    # lam = 60000000
    # hlm_dataset = generateAberrationDataset(res,base,order)


    # fig, axs = plt.subplots(4,4,figsize=(15, 10))
    # fig.subplots_adjust(hspace = .5, wspace=.001)

    # axs = axs.ravel()

    # for i in range(0,len(coeffs)):
    #     coeffs[i] = 1
    #     hlm = getAberrationPhase(hlm_dataset,lam,coeffs)
    #     hlm_zero = hlm == 0
    #     hlm[hlm_zero] = np.nan

    #     im = axs[i].imshow(hlm,cmap = 'inferno',extent=[-1,1,-1,1])
    #     axs[i].set_title('helmholtz j='+str(i),pad = 10)

    #     coeffs = np.zeros(16)

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.71])
    # fig.colorbar(im, cax=cbar_ax)


    # #-----------------------generate Zernike via SLM.controller
    # fig, axs = plt.subplots(4,5,figsize=(15, 10))
    # fig.subplots_adjust(hspace = .7, wspace=.002)

    # axs = axs.ravel()

    # for i in range(1,len(coeffs)):
    #     coeffs[i] = 1
    #     zern = getAberrationPhase(zern_dataset,lam,coeffs)
    #     zern_zero = zern == 0
    #     zern = 2*np.pi*(zern-np.min(zern))/(np.max(zern)-np.min(zern))-np.pi
    #     zern[zern_zero] = np.nan

    #     im = axs[i-1].imshow(zern,cmap = 'jet',extent=[-1,1,-1,1])
    #     axs[i-1].set_title('Zernike j='+str(i),pad = 10)

    #     coeffs = np.zeros(21)

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.71])
    # fig.colorbar(im, cax=cbar_ax)

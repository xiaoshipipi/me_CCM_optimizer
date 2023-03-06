# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:50:23 2023

@author: Mi Xiaoshi
"""

#%% 二维查找表
from scipy.interpolate import interp2d,interpn,LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(0)
u,v=np.meshgrid(np.linspace(-0.5,0.5,11),np.linspace(-0.5,0.5,11))
z=(np.random.rand(11,11)-0.5)*0.04
f=LinearNDInterpolator((u.flatten(),v.flatten()),z.flatten())
plt.figure()
uu=np.linspace(-0.5,0.5,6001)
vv=np.linspace(-0.5,0.5,4001)
u_new,v_new=np.meshgrid(uu,vv)
start_time=time.time()
z_new=f(u_new,v_new)
print('插值耗时:{}s'.format(time.time()-start_time))
plt.plot(u_new+z_new,v_new,'r.')
plt.plot(u+z,v,'k.')

#%%
#%% x-rite 色彩标准

lab_ideal=np.array( # X-Rite官网提供的LAB色彩真值
    [[37.986,13.555,14.059],
      [65.711,18.13,17.81],
      [49.927,-4.88,-21.925],
      [43.139,-13.095,21.905],
      [55.112,8.844,-25.399],
      [70.719,-33.397,-0.199],
      [62.661,36.067,57.096],
      [40.02,10.41,-45.964],
      [51.124,48.239,16.248],
      [30.325,22.976,-21.587],
      [72.532,-23.709,57.255],
      [71.941,19.363,67.857],
      [28.778,14.179,-50.297],
      [55.261,-38.342,31.37],
      [42.101,53.378,28.19],
      [81.733,4.039,79.819],
      [51.935,49.986,-14.574],
      [51.038,-28.631,-28.638],
      [96.539,-0.425,1.186],
      [81.257,-0.638,-0.335],
      [66.766,-0.734,-0.504],
      [50.867,-0.153,-0.27],
      [35.656,-0.421,-1.231],
      [20.461,-0.079,-0.973]],dtype='float32')

rgb_ideal=lab2rgb(lab_ideal)
M_rgb2yuv=np.array([[0.299,0.587,0.114],
                    [-0.169,-0.331,0.499],
                    [0.499,-0.418,-0.081]])
M_yuv2rgb=np.array([[9.99999554e-01, -4.46062343e-04,1.40465882],
                     [9.99655449e-01, -3.44551299e-01,-7.15683665e-01],
                     [1.00177531e+00,1.77530689,9.94081794e-04]])

yuv_ideal=(M_rgb2yuv@gamma_reverse(rgb_ideal).T).T
#%%
def fig_yuv(N=501):
    u,v=np.meshgrid(np.linspace(-0.5,0.5,N),np.linspace(0.5,-0.5,N))
    Y=np.ones(u.shape)*0.5
    img_background=M_yuv2rgb @ np.vstack((Y.flatten(),u.flatten(),v.flatten()))
    img_background=img_background.T.reshape((501,501,3))
    fig=plt.figure(tight_layout=True)
    h_ax=plt.axes(xlim=[-0.5,0.5],ylim=[-0.5,0.5],xticks=np.arange(-0.5,0.5,0.1),yticks=np.arange(-0.5,0.5,0.1))
    plt.grid()
    plt.imshow(isp.gamma(img_background),extent=(-0.5,0.5,-0.5,0.5))
    
    return fig
#%% YUV色谱图
import isp_function as isp
fig_yuv()
for idx,yuv in enumerate(yuv_ideal):
    plt.plot(yuv[1],yuv[2],'ks')
    plt.text(yuv[1]+0.01,yuv[2]+0.01,'{}'.format(idx+1))
#%%
rgb_mean=np.array([
       [0.05604649, 0.03763649, 0.03084474],
       [0.23760724, 0.16160099, 0.13566643],
       [0.1005693 , 0.12852928, 0.17241962],
       [0.0717919 , 0.08149751, 0.054678  ],
       [0.13842899, 0.13839421, 0.1935168 ],
       [0.13314316, 0.21936157, 0.20805185],
       [0.21592635, 0.09212222, 0.04138732],
       [0.05916373, 0.08043833, 0.16416494],
       [0.26455184, 0.09372092, 0.08674786],
       [0.06830533, 0.04834968, 0.07834507],
       [0.22371823, 0.25566936, 0.1190641 ],
       [0.28020669, 0.16114731, 0.06404834],
       [0.02021428, 0.03257127, 0.08527481],
       [0.07551719, 0.13500342, 0.07635126],
       [0.19093955, 0.05140478, 0.03901855],
       [0.44907791, 0.32956755, 0.12193429],
       [0.22170259, 0.09314077, 0.14076718],
       [0.0540875 , 0.1253151 , 0.17184836],
       [0.32161676, 0.31624249, 0.31230679],
       [0.26459358, 0.26259166, 0.26249043],
       [0.18669077, 0.18759722, 0.18778614],
       [0.09848111, 0.09848111, 0.09848111],
       [0.04052817, 0.04142432, 0.04216452],
       [0.01271575, 0.0130849 , 0.01328809]])    

ccm=np.array([[ 1.8693276 , -0.53901856, -0.33030905],
              [-0.20339718,  1.50823653, -0.30483935],
              [-0.00366836, -0.60574533,  1.60941369]])
awb_para=[rgb_mean[21,1]/rgb_mean[21,0],1,rgb_mean[21,1]/rgb_mean[21,2]]
rgb_mean=isp.awb(rgb_mean,awb_para)
ae_comp=1.86942235
yuv_mean=(M_rgb2yuv@(isp.ccm(rgb_mean*ae_comp,ccm).T)).T
for idx,yuv in enumerate(yuv_mean):
    plt.plot(yuv[1],yuv[2],'ko')
    plt.text(yuv[1]+0.01,yuv[2]+0.01,'{}'.format(idx+1))


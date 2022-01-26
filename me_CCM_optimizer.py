# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 09:32:32 2021

@author: Mi Xiaoshi
"""

#%%
import rawpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
#%% 用户设置参数，可编辑
file_path=r'.\IMG_1548.DNG'

black_level=None #黑电平值，需根据设备不同更改，iPhone12的值为528
white_level=None #白电平值，需根据设备不同更改，iPhone12的值为4095

#%% 定义内部函数
M_xyz2rgb=np.array([[3.24096994,-1.53738318,-0.49861076],
                    [-0.96924364,1.8759675,0.04155506],
                    [0.05563008,-0.20397695,1.05697151]])
M_rgb2xyz=np.array([[0.4123908 , 0.35758434, 0.18048079],
                    [0.21263901, 0.71516868, 0.07219231],
                    [0.01933082, 0.11919478, 0.95053216]])

def gamma(x,colorspace='sRGB'): #Gamma变换
    y=np. zeros (x. shape)
    y[x>1]=1
    if colorspace in ( 'sRGB', 'srgh'):
        y[(x>=0)&(x<=0.0031308)]=(323/25*x[ (x>=0)&(x<=0.0031308)])
        y[(x<=1)&(x>0.0031308)]=(1.055*abs(x[ (x<=1)&(x>0.0031308)])**(1/2.4)-0.055)
    elif colorspace in ('TP', 'my'):  
        y[ (x>=0)&(x<=1)]=(1.42*(1-(0.42/(x[(x>=0)&(x<=1)]+0.42))))
    elif (type(colorspace)==float)|(type(colorspace)==int):
        beta=colorspace
        y[ (x>=0)&(x<=1)]=((1+beta)*(1-(beta/(x[(x>=0)&(x<=1)]+beta))))
    return y

def gamma_reverse(x,colorspace= 'sRGB'): #逆Gamma变换
    y=np.zeros(x.shape)
    y[x>1]=1
    if colorspace in ('sRGB', 'srgb'):
        y[(x>=0)&(x<=0.04045)]=x[(x>=0)&(x<=0.04045)]/12.92
        y[(x>0.04045)&(x<=1)]=((x[(x>0.04045)&(x<=1)]+0.055)/1.055)**2.4
    elif colorspace in ('TP','my'):
        y[(x>=0)&(x<=1)]=0.42/(1-(x[(x>=0)&(x<=1)]/1.42))-0.42         
    return y

def im2vector(img): #将图片转换为向量形式
    size=img.shape
    rgb=np.reshape(img,(size[0]*size[1],3))
    func_reverse=lambda rgb : np.reshape(rgb,(size[0],size[1],size[2]))
    return rgb, func_reverse    

def awb(img, awb_para):  
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img
        func_reverse=lambda x : x    
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb,func_reverse)=im2vector(img)   
    rgb[:,0]=rgb[:,0]*awb_para[0]    
    rgb[:,1]=rgb[:,1]*awb_para[1]    
    rgb[:,2]=rgb[:,2]*awb_para[2]    
    img=func_reverse(rgb)    
    return img

def ccm(img, ccm):
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img
        func_reverse=lambda x : x    
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb,func_reverse)=im2vector(img)    
    rgb=rgb.transpose()
    rgb=ccm@rgb
    rgb=rgb.transpose()    
    img_out=func_reverse(rgb)    
    return img_out

def rgb2lab(img,whitepoint='D65'): #rgb转lab
    if (img.ndim==3):
        if (img.shape[2]==3):
            (rgb,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            rgb=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (rgb,Func_reverse)=im2vector(img)
    rgb=rgb.transpose()
    rgb=gamma_reverse(rgb,colorspace='sRGB')
    xyz=M_rgb2xyz@rgb
    xyz=xyz.transpose()
    f=lambda t : (t>((6/29)**3))*(t**(1/3))+\
        (t<=(6/29)**3)*(29*29/6/6/3*t+4/29)
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    L=116*f(xyz[:,1]/Yn)-16
    a=500*(f(xyz[:,0]/Xn)-f(xyz[:,1]/Yn))
    b=200*(f(xyz[:,1]/Yn)-f(xyz[:,2]/Zn))
    Lab=np.vstack((L,a,b)).transpose()
    img_out=func_reverse(Lab)
    return img_out

def lab2rgb(img,whitepoint='D65'): #lab转rgb
    if (img.ndim==3):
        if (img.shape[2]==3):
            (lab,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            lab=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (lab,Func_reverse)=im2vector(img)
    lab=lab.transpose()
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    f_reverse=lambda t : (t>(6/29))*(t**3)+\
        (t<=(6/29))*(3*((6/29)**2)*(t-4/29))
    xyz=np.vstack((Xn*f_reverse((lab[0,:]+16)/116+lab[1,:]/500),
                   Yn*f_reverse((lab[0,:]+16)/116),
                   Zn*f_reverse((lab[0,:]+16)/116-lab[2,:]/200) ))
    rgb=M_xyz2rgb@xyz
    rgb=rgb.transpose()
    rgb=gamma(rgb,colorspace='sRGB')
    rgb_out=func_reverse(rgb)
    return rgb_out

def impoly(img,poly_position=None): #四边形框选图像ROI
    "(rgb_mean,rgb_std,poly_position)=impoly(img)\n(rgb_mean,rgb_std,poly_position)=impoly(img,poly_position)"
    import matplotlib.pyplot as plt
    if poly_position is None:
        fig=plt.figure(figsize=[12.,7.5],tight_layout=True)
        plt.imshow(img)
        fig.show()
        # fig.canvas.set_window_title('waiting. ..')
        fig.canvas.manager.set_window_title('waiting. ..')
        pos=plt.ginput(n=4)
        # plt.close(fig)
    else:
        pos=poly_position
    (n,m)=np.meshgrid(np.arange(0.5,6.5)/6,np.arange(0.5,4.5)/4)
    n=n.flatten()
    m=m.flatten()
    x_center=(1-m)*((1-n)*pos[0][0]+n*pos[1][0])+m*(n*pos[2][0]+(1-n)*pos[3][0])
    y_center=(1-m)*((1-n)*pos[0][1]+n*pos[1][1])+m*(n*pos[2][1]+(1-n)*pos[3][1])
    r_sample=np.floor(min([abs(pos[1][0]-pos[0][0])/6,
                           abs(pos[2][0]-pos[3][0])/6,
                           abs(pos[1][1]-pos[2][1])/4,
                           abs(pos[0][1]-pos[3][1])/4]))*0.2
    if poly_position is None:
        plt.plot(pos[0][0],pos[0][1],'r+')
        plt.plot(pos[1][0],pos[1][1],'r+')
        plt.plot(pos[2][0],pos[2][1],'r+')
        plt.plot(pos[3][0],pos[3][1],'r+')
        # plt.plot(x_center,y_center,'yo')
        plt.plot(x_center-r_sample,y_center-r_sample,'y+')
        plt.plot(x_center+r_sample,y_center-r_sample,'y+')
        plt.plot(x_center-r_sample,y_center+r_sample,'y+')
        plt.plot(x_center+r_sample,y_center+r_sample,'y+')
        fig.show()
        poly_position=pos
    else:
        pass
    rgb_mean=np.zeros((24,3))   
    rgb_std=np.zeros((24,3))   
    for block_idx in range(24):
        block=img[np.int(y_center[block_idx]-r_sample):np.int(y_center[block_idx]+r_sample),
                  np.int(x_center[block_idx]-r_sample):np.int(x_center[block_idx]+r_sample),:]
        rgb_vector,_=im2vector(block)
        rgb_mean[block_idx,:]=rgb_vector.mean(axis=0)
        rgb_std[block_idx,:]=rgb_vector.std(axis=0)
    return (rgb_mean,rgb_std,poly_position)

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

#%% 读取Raw图，预处理，转浮点，OB

raw=rawpy.imread(file_path)
img=raw.raw_image
img_r=img[0::2,0::2].astype('float32')
img_gr=img[0::2,1::2].astype('float32')
img_gb=img[1::2,0::2].astype('float32')
img_b=img[1::2,1::2].astype('float32')

img_g=(img_gr+img_gb)/2


#OB
if black_level is None: #读取Raw图MetaData中黑电平值
    black_level=raw.black_level_per_channel
if white_level is None: #读取Raw图MetaData中白电平值
    white_level=raw.white_level

img_0=(np.dstack((img_r,img_g,img_b)))/(white_level) #img_0:未OB的图像

img_r=(img_r-black_level[0])/(white_level-black_level[0])
img_gr=(img_gr-black_level[1])/(white_level-black_level[1])
img_gb=(img_gb-black_level[2])/(white_level-black_level[2])
img_b=(img_b-black_level[3])/(white_level-black_level[3])

img_g=(img_gr+img_gb)/2

img=np.dstack((img_r,img_g,img_b))

img_1=img #img_0:OB后的图像

#%% 框选图片ROI
(rgb_mean,rgb_std,poly_position)=impoly(img)
# print(rgb_mean)
#%% AE补偿和AWB自动白平衡

func_ae=lambda ae_comp : np.prod(gamma(ae_comp*rgb_mean[20:24,1]))/np.prod(rgb_ideal[20:24,1])-1
ae_res=optimize.root_scalar(func_ae, bracket=[0, 100], method='brentq')
ae_comp=ae_res.root
print('AE补偿:',ae_comp)
img=ae_comp*img
rgb_mean=ae_comp*rgb_mean

awb_para=[rgb_mean[22,1]/rgb_mean[22,0],1,rgb_mean[22,1]/rgb_mean[22,2]]
print('AWB Gain = :',awb_para)
img=awb(img,awb_para)
rgb_mean=awb(rgb_mean,awb_para)

img_2=img
# (rgb_mean,rgb_std,poly_position)=impoly(img,poly_position)

#%%
x2ccm=lambda x : np.array([[1-x[0]-x[1],x[0],x[1]],
                            [x[2],1-x[2]-x[3],x[3]],
                            [x[4],x[5],1-x[4]-x[5]]])

f_lab=lambda x : rgb2lab(gamma(ccm(rgb_mean,x2ccm(x)),colorspace='sRGB'))
f_error=lambda x : f_lab(x)-lab_ideal
f_DeltaE=lambda x : np.sqrt((f_error(x)**2).sum(axis=1,keepdims=True)).mean()
x0=np.array([0,0,0,0,0,0])
print('初始值:',f_DeltaE(x0))
func=lambda x : print('x = ',f_DeltaE(x))
result=optimize.minimize(f_DeltaE,x0,method='Powell',callback=func)
print(result)

img_opti=gamma(ccm(img,x2ccm(result.x)))
img_4=img_opti
img_40=gamma(ccm(img,x2ccm(x0)))
# plt.figure()
# plt.imshow(img_opti)


plt.subplots(2,3,squeeze=False,tight_layout=True)
plt.subplot(2,3,1,xticks=[],yticks=[])
plt.imshow(img_0)
plt.title('RAW')
plt.subplot(2,3,2,xticks=[],yticks=[])
plt.imshow(img_1)
plt.title('BLC')
plt.subplot(2,3,3,xticks=[],yticks=[])
plt.imshow(img_2)
plt.title('AWB')
plt.subplot(2,3,4,xticks=[],yticks=[])
plt.imshow(img_40)
plt.title('Gamma')
plt.subplot(2,3,5,xticks=[],yticks=[])
plt.imshow(img_4)
plt.title('CCM-Gamma')
plt.subplot(2,3,6,xticks=[],yticks=[])





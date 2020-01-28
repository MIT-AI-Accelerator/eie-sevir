"""
Sample code for accessing and visualizing cases in SEVIR.

===============================================================================
   (c) Copyright, 2019 Massachusetts Institute of Technology.
===============================================================================

"""

import sys
import os
import  argparse
import datetime
import numpy as np

import matplotlib as mpl 
if os.path.isdir("/home/gridsan"):
    mpl.use('Agg')

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")

#import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import pandas as pd
import h5py

TYPES    = ['vis','ir069','ir107','vil','lght']

# Nomial Frame time offsets in minutes (used for non-raster types)
FRAME_TIMES = np.arange(-122.5,127.5,5)

def parse_args():
    """
    Parse arguments for this function
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_location","-o",help="location to store output",default='sample_images')
    parser.add_argument("--catalog","-c",help='SEVIR catalog',default='CATALOG.csv')
    parser.add_argument("--data_path","-d",help='Path to SEVIR data',default='data/')
    parser.add_argument("--id","-i",help='SEVIR id (if not provided, a random case is selected',default=None)
    parser.add_argument("--resolution","-r",help='res for statelines',default='c')

    args=parser.parse_args()
    return args

def main():  
    """
    Function to create sample images of a single event in SEVIR

    USAGE:
        
        python view_sample.py    # only works in SEVIR directory

        or, 

        python view_sample.py    --output_location  OUTLOC  --catalog  CATALOG.csv  --data_path  SEVIR_DATA_PATH  

        where,
              OUTLOC  is where your sample images will be saved (a new dir is made to hold them)
              SEVIR_DATA_PATH  is path to the SEVIR data files  ()

    """
    args = parse_args()
    catalog = pd.read_csv(args.catalog,low_memory=False)
    cat_groups = catalog.groupby('id')
    output_location = args.output_location
    
    if args.id:
        sevir_id = args.id
        cat_groups.get_group(sevir_id) # make sure it's valid
    else:
        # pick random group with full sensor coverage 
        allsensors = (cat_groups.size()==5)
        ids = np.array(list(cat_groups.groups.keys()))
        sevir_id = np.random.choice( ids[allsensors], 1 )[0]
        print('Using SEVIR ID',sevir_id)
    output_location=f'{output_location}/{sevir_id}'
    try:
        os.mkdir(output_location)
    except:
        pass

    #print('Loading data')
    data = get_data(sevir_id, cat_groups, args.data_path)

    #print('Making plots')
    make_images2(data, output_location, sevir_id, res=args.resolution)
    update_progress(1.0)

def get_data( sevir_id, grouped_catalog, path ):
    """ 
    returns dict { img_type : {"meta" : META, "data": DATA} }
    """
    cases = grouped_catalog.get_group(sevir_id)
    data = {}
    for typ in TYPES:
        data[typ]={}
        if typ in cases.img_type.values:
            meta = cases[cases.img_type==typ].squeeze()
            data[typ]['meta']=meta
            file_name=f'{path}/{meta.file_name}'
            with h5py.File(file_name,'r') as hf:
                if typ=='lght':
                    data[typ]['data'] = hf[meta.id][:]
                else:
                    data[typ]['data'] = hf[meta.img_type][meta.file_index]     
    return data


def make_images2(data, out_location, id,  res='c'):
    
    # initialize maps 
    #fig = plt.figure(set_frameon=False)
    fig,ax = plt.subplots(1,len(TYPES),figsize=(20,4))
    
    #fig.set_visible(False)
    maps = []
    ims  = []
    for i,typ in enumerate(TYPES):
        if typ not in data:
            maps.append(None)
            ims.append(None)
            continue
        # Scale data go get units correct
        if typ=='vis':
            data[typ]['data']= data[typ]['data']/10000
        elif 'ir' in typ:
            data[typ]['data']=data[typ]['data']/100

        if typ != 'lght':
            n_img = data[typ]['data'].shape[2]
            maps.append( make_basemap(data[typ]['meta'],ax=ax[i],res=res) )
            c = 'r' if typ=='vis' else 'k'
            cmap,norm,vmin,vmax=get_cmap(typ)
            maps[-1].drawstates(color=c)
            maps[-1].drawcoastlines(color=c)
            ims.append(  maps[-1].imshow(data[typ]['data'][:,:,0],cmap=cmap,norm=norm,vmin=vmin,vmax=vmax) )
            time = get_time(data[typ]['meta'],0)
            ax[i].set_xlabel(time)
            ax[i].set_title(get_title(typ))
        else:
            # create first lght frame
            lght_time0 = datetime.datetime.strptime( data['lght']['meta'].time_utc, '%Y-%m-%d %H:%M:%S')
            lght_times = np.array([lght_time0 + datetime.timedelta(seconds=int(s)) for s in data['lght']['data'][:,0]])
            maps.append( make_basemap(data['lght']['meta'],ax=ax[i],res=res) )
            maps[-1].drawstates(color=c)
            maps[-1].drawcoastlines(color=c)
            t0 = lght_time0+datetime.timedelta(minutes=FRAME_TIMES[0])
            t1 = lght_time0+datetime.timedelta(minutes=FRAME_TIMES[1])
            make_lght_frames(data,maps[-1],t0,t1,lght_times=lght_times)
            ax[-1].set_xlabel(time)
            ax[-1].set_title('GOES-16 GLM Lightning Flashes')
        update_progress(np.round( 100 * 1/len(FRAME_TIMES) ) / 100)


    fig.savefig(f'{out_location}/{id}_000.png')
    
    n_img = 49 # Think of way to get this from data
    
    for t in range(1,n_img):
        for i,typ in enumerate(TYPES):
            if typ not in data:
                continue
            if typ != 'lght':
                ims[i].set_array(data[typ]['data'][:,:,t])
                time = get_time(data[typ]['meta'],t)
                ax[i].set_xlabel(time)
            elif typ=='lght':
                #time = get_time(data[typ]['meta'],t)
                time = lght_time0 + datetime.timedelta(minutes=(FRAME_TIMES[t]+2.5))
                t0 = lght_time0+datetime.timedelta(minutes=FRAME_TIMES[t])
                t1 = lght_time0+datetime.timedelta(minutes=FRAME_TIMES[t+1])
                make_lght_frames(data,maps[i],t0,t1,lght_times=lght_times)
                ax[-1].set_xlabel(time)
        fig.savefig(f'{out_location}/{id}_%.3d.png' % t)
        update_progress(np.round( 100 * t/len(FRAME_TIMES) ) / 100)


def make_lght_frames(data,bmap,t0,t1,lght_times=None):
    
    meta = data['lght']['meta']
    if lght_times is None:
        time0 = datetime.datetime.strptime( meta.time_utc, '%Y-%m-%d %H:%M:%S')
        lght_times = np.array([time0 + datetime.timedelta(seconds=m) for m in data[:,0]])
    mask = np.logical_and(lght_times>=t0, lght_times<t1)
    lats,lons = data['lght']['data'][mask,1], data['lght']['data'][mask,2]
    x,y=bmap(lons,lats)
    try:
        bmap.ax.lines[-1].remove()
    except:
        pass
    bmap.plot(x,y,'rx')

    return


def get_title(typ):
    if typ=='vil':
        return "Vertically Integrated Liquid"
    elif typ=='vis':
        return "GOES-16 C02 VIS "
    elif typ=='ir069':
        return 'GOES-16 C09 IR Water Vapor'
    elif typ=='ir107':
        return 'GOES-16 C13 IR Brightness Temp'
    return ''


def get_time(meta,t):
    time=datetime.datetime.strptime(meta.time_utc,'%Y-%m-%d %H:%M:%S')
    d=[datetime.timedelta(minutes=int(n)) for n in meta.minute_offsets.split(':')]
    return (time+d[t]).isoformat()




#groups.get_group('S832112')
#data = get_data('S832112', sevir_catalog.groupby('id') )



def plot_case( h5file, meta, idx ):
    arr = h5file.root.OUT[idx,:,:,:]
    s = meta.iloc[idx]
    n_cols = 6
    n_rows = 24//n_cols
    fig,axs=plt.subplots(n_rows,n_cols,figsize=(15,6))
    cmap,norm = vil_cmap()

    for i in range(24):
        m = Basemap(llcrnrlat=s.llcrnrlat, llcrnrlon=s.llcrnrlon,
                    urcrnrlat=s.urcrnrlat,urcrnrlon=s.urcrnrlon,
                    width=s.width_m, height=s.height_m,
                    lat_0=38, lon_0=-98,
                    projection='laea', 
                    resolution='c',
                    ax=axs[i//n_cols,i % n_cols])
        m.drawstates()
        m.imshow(arr[:,:,i],cmap=cmap,norm=norm)
        axs[i//n_cols,i % n_cols].set_xlabel(s.times[i][1:-1])
    try:
        fig.suptitle('Event ID:  %d  ' % s.EVENT_ID, fontsize=16)        
    except Exception:
        pass
    plt.show()



def make_basemap(s,ax=None,res='c'):
    if ax:
        return Basemap(llcrnrlat=s.llcrnrlat, llcrnrlon=s.llcrnrlon,
                    urcrnrlat=s.urcrnrlat,urcrnrlon=s.urcrnrlon,
                    width=s.width_m, height=s.height_m,
                    lat_0=38, lon_0=-98,
                    projection='laea', 
                    resolution=res,
                    ax=ax)
    else:
        return Basemap(llcrnrlat=s.llcrnrlat, llcrnrlon=s.llcrnrlon,
                    urcrnrlat=s.urcrnrlat,urcrnrlon=s.urcrnrlon,
                    width=s.width_m, height=s.height_m,
                    lat_0=38, lon_0=-98,
                    projection='laea', 
                    resolution=res)


def make_images(data_file, meta_file, idx, type, out_location,res='c'):
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    with h5py.File(data_file,'r') as hf:
        name= get_name(type)
        arr = hf[name][idx] # L x W x T
    meta = pd.read_csv(meta_file)
    m = make_basemap(meta.iloc[idx],ax=ax,res=res)
    c = 'r' if type=='VIS' else 'k'
    m.drawstates(color=c)
    cmap,norm=get_cmap(type)

    im=m.imshow(arr[:,:,0],cmap=cmap,norm=norm)
    fig.savefig(f'{out_location}/{type}_000.png')
    for i in range(1,arr.shape[2]):
        im.set_array(arr[:,:,i])
        fig.savefig(f'{out_location}/{type}_%.3d.png' % i)





def get_name(type):
    return type




def get_cmap(type):
    if type.lower()=='vis':
        cmap,norm = vis_cmap()
        vmin,vmax=None,None
    elif type.lower()=='vil':
        cmap,norm=vil_cmap()
        vmin,vmax=None,None
    elif type.lower()=='ir069':
        cmap,norm=c09_cmap()
        vmin,vmax=-80,-10
    else:
        cmap,norm='jet',None
        vmin,vmax=-70,20
#    elif type=='IR107':
#        cmap,norm=ir_cmap()

    return cmap,norm,vmin,vmax



def vil_cmap():
    cols=[   [0,0,0],
             [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
             [0.1568627450980392,  0.7450980392156863,  0.1568627450980392],
             [0.09803921568627451, 0.5882352941176471,  0.09803921568627451],
             [0.0392156862745098,  0.4117647058823529,  0.0392156862745098],
             [0.0392156862745098,  0.29411764705882354, 0.0392156862745098],
             [0.9607843137254902,  0.9607843137254902,  0.0],
             [0.9294117647058824,  0.6745098039215687,  0.0],
             [0.9411764705882353,  0.43137254901960786, 0.0],
             [0.6274509803921569,  0.0, 0.0],
             [0.9058823529411765,  0.0, 1.0]]
    lev = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm
    

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
## Taken from https://stackoverflow.com/questions/3160699/python-progress-bar
def update_progress(progress):
    barLength = 50 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), int(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()    
    

def vis_cmap():
    cols=[[0,0,0],
             [0.0392156862745098, 0.0392156862745098, 0.0392156862745098],
             [0.0784313725490196, 0.0784313725490196, 0.0784313725490196],
             [0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
             [0.1568627450980392, 0.1568627450980392, 0.1568627450980392],
             [0.19607843137254902, 0.19607843137254902, 0.19607843137254902],
             [0.23529411764705882, 0.23529411764705882, 0.23529411764705882],
             [0.27450980392156865, 0.27450980392156865, 0.27450980392156865],
             [0.3137254901960784, 0.3137254901960784, 0.3137254901960784],
             [0.35294117647058826, 0.35294117647058826, 0.35294117647058826],
             [0.39215686274509803, 0.39215686274509803, 0.39215686274509803],
             [0.43137254901960786, 0.43137254901960786, 0.43137254901960786],
             [0.47058823529411764, 0.47058823529411764, 0.47058823529411764],
             [0.5098039215686274, 0.5098039215686274, 0.5098039215686274],
             [0.5490196078431373, 0.5490196078431373, 0.5490196078431373],
             [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
             [0.6274509803921569, 0.6274509803921569, 0.6274509803921569],
             [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
             [0.7058823529411765, 0.7058823529411765, 0.7058823529411765],
             [0.7450980392156863, 0.7450980392156863, 0.7450980392156863],
             [0.7843137254901961, 0.7843137254901961, 0.7843137254901961],
             [0.8235294117647058, 0.8235294117647058, 0.8235294117647058],
             [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
             [0.9019607843137255, 0.9019607843137255, 0.9019607843137255],
             [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
             [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
             [0.9803921568627451, 0.9803921568627451, 0.9803921568627451]]
    lev=[0.  , 0.02, 0.04, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.2 , 0.24,
       0.28, 0.32, 0.36, 0.4 , 0.44, 0.48, 0.52, 0.56, 0.6 , 0.64, 0.68,
       0.72, 0.76, 0.8 , 0.9 , 1.  ]
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm


def ir_cmap():
    cols=[[0,0,0],[1.0, 1.0, 1.0],
     [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
     [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
     [0.9019607843137255, 0.9019607843137255, 0.9019607843137255],
     [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
     [0.8235294117647058, 0.8235294117647058, 0.8235294117647058],
     [0.7843137254901961, 0.7843137254901961, 0.7843137254901961],
     [0.7450980392156863, 0.7450980392156863, 0.7450980392156863],
     [0.7058823529411765, 0.7058823529411765, 0.7058823529411765],
     [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
     [0.6274509803921569, 0.6274509803921569, 0.6274509803921569],
     [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
     [0.5490196078431373, 0.5490196078431373, 0.5490196078431373],
     [0.5098039215686274, 0.5098039215686274, 0.5098039215686274],
     [0.47058823529411764, 0.47058823529411764, 0.47058823529411764],
     [0.43137254901960786, 0.43137254901960786, 0.43137254901960786],
     [0.39215686274509803, 0.39215686274509803, 0.39215686274509803],
     [0.35294117647058826, 0.35294117647058826, 0.35294117647058826],
     [0.3137254901960784, 0.3137254901960784, 0.3137254901960784],
     [0.27450980392156865, 0.27450980392156865, 0.27450980392156865],
     [0.23529411764705882, 0.23529411764705882, 0.23529411764705882],
     [0.19607843137254902, 0.19607843137254902, 0.19607843137254902],
     [0.1568627450980392, 0.1568627450980392, 0.1568627450980392],
     [0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
     [0.0784313725490196, 0.0784313725490196, 0.0784313725490196],
     [0.0392156862745098, 0.0392156862745098, 0.0392156862745098],
     [0.0, 0.803921568627451, 0.803921568627451]]
    lev=[-110. , -105.2,  -95.2,  -85.2,  -75.2,  -65.2,  -55.2,  -45.2,
        -35.2,  -28.2,  -23.2,  -18.2,  -13.2,   -8.2,   -3.2,    1.8,
          6.8,   11.8,   16.8,   21.8,   26.8,   31.8,   36.8,   41.8,
         46.8,   51.8,   90. ,  100. ]
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm         


def c09_cmap():
    cols=[
    [1.000000, 0.000000, 0.000000],
    [1.000000, 0.031373, 0.000000],
    [1.000000, 0.062745, 0.000000],
    [1.000000, 0.094118, 0.000000],
    [1.000000, 0.125490, 0.000000],
    [1.000000, 0.156863, 0.000000],
    [1.000000, 0.188235, 0.000000],
    [1.000000, 0.219608, 0.000000],
    [1.000000, 0.250980, 0.000000],
    [1.000000, 0.282353, 0.000000],
    [1.000000, 0.313725, 0.000000],
    [1.000000, 0.349020, 0.003922],
    [1.000000, 0.380392, 0.003922],
    [1.000000, 0.411765, 0.003922],
    [1.000000, 0.443137, 0.003922],
    [1.000000, 0.474510, 0.003922],
    [1.000000, 0.505882, 0.003922],
    [1.000000, 0.537255, 0.003922],
    [1.000000, 0.568627, 0.003922],
    [1.000000, 0.600000, 0.003922],
    [1.000000, 0.631373, 0.003922],
    [1.000000, 0.666667, 0.007843],
    [1.000000, 0.698039, 0.007843],
    [1.000000, 0.729412, 0.007843],
    [1.000000, 0.760784, 0.007843],
    [1.000000, 0.792157, 0.007843],
    [1.000000, 0.823529, 0.007843],
    [1.000000, 0.854902, 0.007843],
    [1.000000, 0.886275, 0.007843],
    [1.000000, 0.917647, 0.007843],
    [1.000000, 0.949020, 0.007843],
    [1.000000, 0.984314, 0.011765],
    [0.968627, 0.952941, 0.031373],
    [0.937255, 0.921569, 0.050980],
    [0.901961, 0.886275, 0.074510],
    [0.870588, 0.854902, 0.094118],
    [0.835294, 0.823529, 0.117647],
    [0.803922, 0.788235, 0.137255],
    [0.772549, 0.756863, 0.160784],
    [0.737255, 0.725490, 0.180392],
    [0.705882, 0.690196, 0.200000],
    [0.670588, 0.658824, 0.223529],
    [0.639216, 0.623529, 0.243137],
    [0.607843, 0.592157, 0.266667],
    [0.572549, 0.560784, 0.286275],
    [0.541176, 0.525490, 0.309804],
    [0.509804, 0.494118, 0.329412],
    [0.474510, 0.462745, 0.349020],
    [0.752941, 0.749020, 0.909804],
    [0.800000, 0.800000, 0.929412],
    [0.850980, 0.847059, 0.945098],
    [0.898039, 0.898039, 0.964706],
    [0.949020, 0.949020, 0.980392],
    [1.000000, 1.000000, 1.000000],
    [0.964706, 0.980392, 0.964706],
    [0.929412, 0.960784, 0.929412],
    [0.890196, 0.937255, 0.890196],
    [0.854902, 0.917647, 0.854902],
    [0.815686, 0.894118, 0.815686],
    [0.780392, 0.874510, 0.780392],
    [0.745098, 0.850980, 0.745098],
    [0.705882, 0.831373, 0.705882],
    [0.670588, 0.807843, 0.670588],
    [0.631373, 0.788235, 0.631373],
    [0.596078, 0.764706, 0.596078],
    [0.560784, 0.745098, 0.560784],
    [0.521569, 0.721569, 0.521569],
    [0.486275, 0.701961, 0.486275],
    [0.447059, 0.678431, 0.447059],
    [0.411765, 0.658824, 0.411765],
    [0.376471, 0.635294, 0.376471],
    [0.337255, 0.615686, 0.337255],
    [0.301961, 0.592157, 0.301961],
    [0.262745, 0.572549, 0.262745],
    [0.227451, 0.549020, 0.227451],
    [0.192157, 0.529412, 0.192157],
    [0.152941, 0.505882, 0.152941],
    [0.117647, 0.486275, 0.117647],
    [0.078431, 0.462745, 0.078431],
    [0.043137, 0.443137, 0.043137],
    [0.003922, 0.419608, 0.003922],
    [0.003922, 0.431373, 0.027451],
    [0.003922, 0.447059, 0.054902],
    [0.003922, 0.462745, 0.082353],
    [0.003922, 0.478431, 0.109804],
    [0.003922, 0.494118, 0.137255],
    [0.003922, 0.509804, 0.164706],
    [0.003922, 0.525490, 0.192157],
    [0.003922, 0.541176, 0.215686],
    [0.003922, 0.556863, 0.243137],
    [0.007843, 0.568627, 0.270588],
    [0.007843, 0.584314, 0.298039],
    [0.007843, 0.600000, 0.325490],
    [0.007843, 0.615686, 0.352941],
    [0.007843, 0.631373, 0.380392],
    [0.007843, 0.647059, 0.403922],
    [0.007843, 0.662745, 0.431373],
    [0.007843, 0.678431, 0.458824],
    [0.007843, 0.694118, 0.486275],
    [0.011765, 0.705882, 0.513725],
    [0.011765, 0.721569, 0.541176],
    [0.011765, 0.737255, 0.568627],
    [0.011765, 0.752941, 0.596078],
    [0.011765, 0.768627, 0.619608],
    [0.011765, 0.784314, 0.647059],
    [0.011765, 0.800000, 0.674510],
    [0.011765, 0.815686, 0.701961],
    [0.011765, 0.831373, 0.729412],
    [0.015686, 0.843137, 0.756863],
    [0.015686, 0.858824, 0.784314],
    [0.015686, 0.874510, 0.807843],
    [0.015686, 0.890196, 0.835294],
    [0.015686, 0.905882, 0.862745],
    [0.015686, 0.921569, 0.890196],
    [0.015686, 0.937255, 0.917647],
    [0.015686, 0.952941, 0.945098],
    [0.015686, 0.968627, 0.972549],
    [1.000000, 1.000000, 1.000000]]
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    return ListedColormap(cols),None



if __name__=='__main__':
    main()





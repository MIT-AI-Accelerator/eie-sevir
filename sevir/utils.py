"""
Input generator for sevir
"""

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import h5py

os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import GeneratorEnqueuer

DEFAULT_CATALOG = '/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/CATALOG.csv'
DEFAULT_DATA_HOME = '/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/data/'

class SEVIRSequence(Sequence):
    """
    Sequence class for generating batches from SEVIR
    
    Parameters
    ----------
    catalog  str or pd.DataFrame
        name of SEVIR catalog file to be read in, or an already read in and processed catalog
    x_img_types  list 
        List of image types to be used as model inputs
    y_img_types  list or None
       List of image types to be used as model targets (if None, __getitem__ returns only x_img_types )
       
    
    """
    def __init__(self,
                 x_img_types=['vil'],
                 y_img_types=None, 
                 catalog=DEFAULT_CATALOG,
                 batch_size = 3,
                 datetime_filter=None,
                 start_date=None,
                 end_date=None,
                 unwrap_time=False,
                 sevir_data_home=DEFAULT_DATA_HOME,
                 shuffle=False,
                 shuffle_seed=1
                 ):
        self._samples = None
        self._hdf_files = {}
        self.x_img_types = x_img_types
        self.y_img_types = y_img_types
        if isinstance(catalog,(str,)):
            self.catalog=pd.read_csv(catalog,parse_dates=['time_utc'],low_memory=False)
        else:
            self.catalog=catalog
        self.batch_size=batch_size

        self.datetime_filter=datetime_filter
        self.start_date=start_date
        self.end_date=end_date
        self.unwrap_time = unwrap_time
        self.sevir_data_home=sevir_data_home
        self.shuffle=shuffle
        self.shuffle_seed=shuffle_seed
        
        if self.start_date:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date ]
        if self.end_date:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        self._compute_samples()
        self._open_files()

    def __del__(self):
        for f,hf in self._hdf_files.items():
            hf.close()

    def __len__(self):
        """
        How many batches are present
        """
        #return int(np.ceil(len(self.x) / float(self.batch_size)))
        if self._samples is not None:
            return int(np.ceil(self._samples.shape[0] / float(self.batch_size)))
        else:
            return 0
        

    def __getitem__(self, idx):
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)    
        """
        batch = self._samples.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = {}
        for index, row in batch.iterrows():
            data = self._read_data(row,data)
        if self.y_img_types is None:
            return [data[t] for t in self.x_img_types]
        else:
            return [data[t] for t in self.x_img_types],[data[t] for t in self.y_img_types]
    
    
    def _read_data(self,row,data):
        """
        row is a series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index
        """
        imgtyps = np.unique([x.split('_')[0] for x in list(row.keys())])
        for t in imgtyps:
            fname = row[f'{t}_filename']
            idx   = row[f'{t}_index']
            if self.unwrap_time:
                tidx = row[f'{t}_time_index']
                data_i = self._hdf_files[fname][t][idx:idx+1,:,:,tidx]
                data[t] = np.concatenate( (data[t],data_i),axis=0 ) if (t in data) else data_i # TODO: faster to preallocate
            else:
                data_i = self._hdf_files[fname][t][idx:idx+1,:,:,:]
                data[t] = np.concatenate( (data[t],data_i),axis=0 ) if (t in data) else data_i
        return data


    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets
           self._samples  

        """
        # locate all events containing colocated x_img_types and y_img_types
        imgt = self.x_img_types
        if self.y_img_types:
            imgt=list( set(imgt + self.y_img_types) ) # remove duplicates
        imgts = set(imgt)            
        filtcat = self.catalog[ np.logical_or.reduce([self.catalog.img_type==i for i in imgt]) ]
        # remove rows missing one or more requested img_types
        filtcat = filtcat.groupby('id').filter(lambda x: imgts.issubset(set(x['img_type'])))
        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        filtcat = filtcat.groupby('id').filter(lambda x: x.shape[0]==len(imgt))
        self._samples = filtcat.groupby('id').apply( lambda df: self._df_to_series(df,imgt) )
        if self.shuffle:
            self._samples=self._samples.sample(frac=1,random_state=self.shuffle_seed)
        

    def _df_to_series(self,df,imgt):
        N_FRAMES=49  # TODO:  don't hardcode this
        d = {}
        df = df.set_index('img_type')
        for i in imgt:
            s = df.loc[i]
            if not self.unwrap_time:
                d.update( {f'{i}_filename':[s.file_name], 
                           f'{i}_index':[s.file_index]} )
            else:
                d.update( {f'{i}_filename':[s.file_name]*N_FRAMES, 
                           f'{i}_index':[s.file_index]*N_FRAMES,
                           f'{i}_time_index':range(N_FRAMES)} )      
        return pd.DataFrame(d)

    def _open_files(self):
        """
        Opens HDF files
        """
        imgt = self.x_img_types
        if self.y_img_types:
            imgt=list( set(imgt + self.y_img_types) ) # remove duplicates
        hdf_filenames = []
        for t in imgt:
            hdf_filenames += list(np.unique( self._samples[f'{t}_filename'].values ))
        self._hdf_files = {}
        for f in hdf_filenames:
            print('Opening HDF5 file for reading',f)
            self._hdf_files[f] = h5py.File(self.sevir_data_home+'/'+f,'r')

    def on_epoch_end(self):
        if self.shuffle:
            df.sample(frac=1,random_state=self.shuffle)


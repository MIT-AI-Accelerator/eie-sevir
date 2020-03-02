import io
import os
import pickle
import h5py
import argparse
from datetime import datetime
from functools import partial
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sevir.utils import SEVIRSequence


from sevir.display import get_cmap
 
# Set up callbacks
datetag=datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = f'{os.environ["HOME"]}/logs/sevir_vae/' + datetag
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
file_writer_tstimg = tf.summary.create_file_writer(logdir+'/imgs')

netsave_dir = f'{os.environ["HOME"]}/trained_networks/sevir_vae/'+datetag
Path(netsave_dir).mkdir(parents=True,exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Settings for VAE training')
    parser.add_argument('--version','-v', type=str, help='VAE version',default='1')

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    x_trn,x_tst = load_datasets()  
    
    if args.version=='1':
        from sevir.models.vaes.classes import VAE_v1 as VAE
    elif args.version=='2':
        from sevir.models.vaes.classes import VAE_v2 as VAE
    vae = VAE()

    # Callbacks
    def predict(x):
        yhat=vae.predict(x)
        return SEVIRSequence.unnormalize(yhat,(1/255,0))
    testimg_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=partial(plot_test_images,x_test=x_tst,predict=predict) )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(netsave_dir+'/weights.{epoch:02d}-{val_loss:.2f}.tf', 
                              monitor='val_loss', verbose=0, 
                              save_best_only=True, save_weights_only=False, 
                              mode='auto', period=1)
    
    vae.fit(x_trn[0],x_trn[0],
            epochs=200,
            batch_size=64,
            max_queue_size=10,
            validation_data=(x_tst[0],x_tst[0]),
            workers=5,
            verbose=1,
            use_multiprocessing=True,
            callbacks=[tensorboard_callback,
                      testimg_cb,
                      checkpoint_cb])


def load_generators():
    # date ranges for train/test
    train_dates = (datetime(2018,1,1),datetime(2019,6,1))
    test_dates  = (datetime(2019,6,1),datetime(2020,1,1))

    # Generate single images of weather radar echos
    trn_gen_file='/tmp/trn_gen.pkl'
    if not os.path.exists(trn_gen_file):
        data_gen_trn = SEVIRSequence(x_img_types=['vil'],
                                     batch_size=256,
                                     n_batch_per_epoch=200,
                                     unwrap_time=True, # don't generate sequences
                                     shuffle=True,
                                     start_date=train_dates[0],
                                     end_date=train_dates[1],
                                     normalize_x=[(1/255,0)])
        #data_gen_trn.save(trn_gen_file)
    else:
        data_gen_trn = SEVIRSequence.load(trn_gen_file)

    tst_gen_file='/tmp/tst_gen.pkl'
    if not os.path.exists(tst_gen_file):
        data_gen_tst = SEVIRSequence(x_img_types=['vil'], 
                                     batch_size=256,
                                     n_batch_per_epoch=50,
                                     unwrap_time=True, # don't generate sequences
                                     shuffle=False,
                                     start_date=test_dates[0],
                                     end_date=test_dates[1],
                                     normalize_x=[(1/255,0)])
        #data_gen_tst.save(tst_gen_file)
    else:
        data_gen_tst = SEVIRSequence.load(tst_gen_file)

    return data_gen_trn,data_gen_tst


def load_datasets(n_batches_train=200,n_batches_test=50):
    
    trn_data_name = f'{os.environ["HOME"]}/data'
    if not os.path.exists(trn_data_name):
        os.mkdir(trn_data_name)
    trn_data_name+='/sevir_vae_train.h5'
    if not os.path.exists(trn_data_name):
        # make it
        data_gen_trn,data_gen_tst = load_generators()
        print('Loading training data')
        x_train = data_gen_trn.load_batches(n_batches=n_batches_train,progress_bar=True)
        print('Loading test data')
        x_test = data_gen_tst.load_batches(n_batches=n_batches_test,progress_bar=True)
        with h5py.File(trn_data_name,'w') as hf:
            hf.create_dataset("TRAIN",data=x_train)
            hf.create_dataset("TEST",data=x_test)
    else:
        with h5py.File(trn_data_name,'r') as hf:
            x_train = hf['TRAIN'][:]
            x_test = hf['TEST'][:]
    return x_train,x_test


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image



def plot_test_images(epoch,logs,x_test,predict):
    cmap,norm,vmin,vmax = get_cmap('vil')
    fig,axs = plt.subplots(3,6,figsize=(15,10))
    np.random.seed(seed=2)
    idx = np.random.choice( x_test[0].shape[0],9)
    ii=0
    yhat=predict(x_test[0][idx])
    for i in range(3):
        for j in range(0,6,2):
            xx=SEVIRSequence.unnormalize(x_test[0][idx[ii],:,:,0],(1/255,0))
            axs[i][j].imshow(xx,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
            axs[i][j].set_xticks([], []), axs[i][j].set_yticks([], [])
            axs[i][j].set_xlabel('Original Image')
            axs[i][j+1].imshow(yhat[ii,:,:,0],cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
            axs[i][j+1].set_xticks([], []), axs[i][j+1].set_yticks([], [])
            axs[i][j+1].set_xlabel('Decoded Image')
            ii+=1
    tst_images = plot_to_image(fig)
    # Log the confusion matrix as an image summary.
    with file_writer_tstimg.as_default():
        tf.summary.image("VAE Test Images", tst_images, step=epoch)








if __name__=='__main__':
    main()

"""
Variational autoencoder models for SEVIR
"""

import sys
import numpy as np
from datetime import datetime

import tensorflow as tf



class VAE_v1(object):
    """
    A U-net based convolutional variational autoencoder model for SEVIR data
    """
    def __init__(self, 
                 latent_dim=256,
                 in_shape=(384,384,1),
                 decoder_std=1,
                 optimizer='adam' ):
        self.latent_dim=latent_dim
        self.in_shape=in_shape
        self.decoder_std=decoder_std
                                               # Comments after layers  
                                               # are only for default case
        self.input = tf.keras.Input(shape=in_shape) # 384 @1km  
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.vae = tf.keras.Model(inputs=self.input, 
                                  outputs=self.decoder, 
                                  name='vae')
        self.loss = build_loss(self)

        self.vae.compile(optimizer=optimizer,
                         loss=self.loss,
                         experimental_run_tf_function=False) # I don't know why I need this, 
                                                # I found it on some random forum bc this wouldn't
                                                # compile without it.
    

    def build_encoder(self):
        enc = tf.keras.layers.Conv2D( filters=32, kernel_size=7, strides=(2, 2), 
                                          activation='relu',padding='SAME',name='econv1')(self.input) # 192 @2km
        enc = tf.keras.layers.Conv2D( filters=64, kernel_size=5, strides=(2, 2), 
                                           activation='relu',padding='SAME',name='econv2')(enc) # 96 @4km
        enc = tf.keras.layers.Conv2D( filters=128, kernel_size=5, strides=(2, 2), 
                                          activation='relu',padding='SAME',name='econv3')(enc)  # 48 @8km
        enc = tf.keras.layers.Conv2D( filters=128, kernel_size=3, strides=(2, 2), 
                                          activation='relu',padding='SAME',name='econv4')(enc)  # 24 @16km
        enc = tf.keras.layers.Conv2D( filters=256, kernel_size=3, strides=(2, 2), 
                                          activation='relu',padding='SAME',name='econv5')(enc)  # 12 @32km

        # Encoded mean -- 6 x 6 x latentdim  @ 64km
        self.encoder_mu  = tf.keras.layers.Conv2D( filters=self.latent_dim, kernel_size=3, strides=(2, 2), 
                                              padding='SAME',name='mu')(encoder)
        # Encoded log(var) -- 6 x 6 x latentdim
        self.encoder_log_var = tf.keras.layers.Conv2D( filters=self.latent_dim, kernel_size=3, strides=(2, 2), 
                                              activation='relu',padding='SAME',name='var')(encoder)

        self.z = self.sample()
        return tf.keras.Model(inputs=self.input,
                              outputs=[self.encoder_mu,
                                       self.encoder_log_var,
                                       self.z],name='encoder')

    def sample(self, mean, log_var):
        """
        Reparameterization trick
        """
        batch = tf.shape(self.encoder_mu)[0]
        dim1  = tf.shape(self.encoder_mu)[1]
        eps=tf.random.normal(shape=[batch,dim1,dim1,latent_dim], mean=0.0, stddev=1.0,name='rns')
        return self.encoder_mu  + tf.exp(0.5*self.encoder_log_var) * eps

    def build_decoder(self):
        dec = tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=3,
                        strides=2,padding='SAME',activation='relu',name='dconv5')(self.z)      # 12 @16km
        dec = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=3,
                                strides=2,padding='SAME',activation='relu',name='dconv4')(dec) # 24 @16km
        dec = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=3,
                                strides=2,padding='SAME',activation='relu',name='dconv3')(dec) # 48 @8km
        dec = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=5,
                                strides=2,padding='SAME',activation='relu',name='dconv2')(dec) # 96 @4km
        dec = tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=5,
                                strides=2,padding='SAME',activation='relu',name='dconv1')(dec) # 192 @2km
        dec = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=7,
                                strides=2,padding='SAME',name='decoder')(dec)     # 384 @1km
        return dec

    def build_loss(self):
        def vae_loss(img,decoded_img):
            if int(tf.__version__[0])<2: # TF 1
                reconstruction_loss = -tf.reduce_sum( tf.contrib.distributions.Normal(
                                        decoded_img,self.decoder_std ).log_prob(img), axis=[1,2,3],name='reconloss' )
            else:  # TF 2
                import tensorflow_probability as tfp
                reconstruction_loss = -tf.reduce_sum( tfp.distributions.Normal(
                              decoded_img, self.decoder_std).log_prob(img), axis=[1,2,3],name='reconloss' )
            
            kl_loss = -0.5 * tf.reduce_sum( (1 + self.encoder_log_var 
                                                - tf.exp(self.encoder_log_var) 
                                                - self.encoder_mu ** 2 ), axis=[1,2,3],name='klloss' )
            return tf.reduce_mean(reconstruction_loss + kl_loss,axis=0)
        return vae_loss














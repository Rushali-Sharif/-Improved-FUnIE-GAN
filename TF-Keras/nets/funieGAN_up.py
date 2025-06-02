from _future_ import print_function, division
import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
from keras.layers import Input, Dropout, Concatenate
from keras.layers import Add, BatchNormalization, Activation, UpSampling2D, Conv2D, MaxPooling2D, LeakyReLU
from keras.applications import VGG19
import numpy as np
from scipy.ndimage import gaussian_filter
from keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import vgg19
from keras.applications.convnext import ConvNeXtBase
from tensorflow.keras.layers import Concatenate
from keras.layers import Dense, Concatenate
from tensorflow.keras.layers import LSTM, Reshape



def VGG19_Content(dataset='imagenet'):
    input_layer = Input(shape=(128, 128, 3))
    conv = ConvNeXtBase(include_top=False, weights=dataset, input_tensor=input_layer)

    content_layers_stage_0 = ['convnext_base_stage_0_block_2_pointwise_conv_2', 'convnext_base_stage_0_block_2_depthwise_conv', 'convnext_base_stage_0_block_2_identity']
    content_layers_stage_1 = ['convnext_base_stage_1_block_2_pointwise_conv_2', 'convnext_base_stage_1_block_2_depthwise_conv', 'convnext_base_stage_1_block_2_identity']
    content_layers_stage_2 = ['convnext_base_stage_2_block_2_pointwise_conv_2', 'convnext_base_stage_2_block_2_depthwise_conv', 'convnext_base_stage_2_block_2_identity']
    content_layers_stage_3 = ['convnext_base_stage_3_block_2_pointwise_conv_2', 'convnext_base_stage_3_block_2_depthwise_conv', 'convnext_base_stage_3_block_2_identity']

    content_outputs_stage_0 = [conv.get_layer(name).output for name in content_layers_stage_0]
    content_outputs_stage_1 = [conv.get_layer(name).output for name in content_layers_stage_1]
    content_outputs_stage_2 = [conv.get_layer(name).output for name in content_layers_stage_2]
    content_outputs_stage_3 = [conv.get_layer(name).output for name in content_layers_stage_3]

    flattened_content_stage_0 = [Flatten()(output) for output in content_outputs_stage_0]
    flattened_content_stage_1 = [Flatten()(output) for output in content_outputs_stage_1]
    flattened_content_stage_2 = [Flatten()(output) for output in content_outputs_stage_2]
    flattened_content_stage_3 = [Flatten()(output) for output in content_outputs_stage_3]

    merged_content = Concatenate()([*flattened_content_stage_0, *flattened_content_stage_1, *flattened_content_stage_2, *flattened_content_stage_3])

    content_model = Model(input_layer, merged_content)
    return content_model

def blind_denoising_submodule():
    model = tf.keras.models.Sequential()

    # Convolutional layers
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', input_shape=(128, 128, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', dilation_rate=2))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', dilation_rate=4))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))

    # Residual block
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same'))
    model.add(tf.keras.layers.Add())  # Skip connection
    model.add(tf.keras.layers.Activation('relu'))

    # Residual network with feature attention
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))

    # Assume feature attention mechanism here
    # You may need to implement the attention mechanism separately

    # Output layer
    model.add(tf.keras.layers.Conv2D(3, (3, 3), strides=1, padding='same'))
    model.add(tf.keras.layers.Activation('linear'))  # Linear activation for reconstruction

    return model

def color_correction_module(input_layer):
    # Add your color correction layers here
    x = Dense(256, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    corrected_output = Dense(3, activation='linear')(x)  # Assuming RGB images

    return corrected_output 

def deblurring_submodule(input_shape):
    input_layer = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(64, (5, 5), strides=1, padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(128, (5, 5), strides=1, padding='same', activation='relu')(conv1)
    conv3 = Conv2D(256, (5, 5), strides=1, padding='same', activation='relu')(conv2)

    # Reshape the output before LSTM
    reshaped_output = Reshape((-1, 256))(conv3)

    # LSTM layer
    lstm = LSTM(256, return_sequences=True)(reshaped_output)
    
    # Reshape the output before upsampling
    reshaped_lstm_output = Reshape((65536 // (8 * 😎, 8, 8, 256))(lstm)

    # Decoder
    upsample1 = UpSampling2D(size=(2, 2))(reshaped_lstm_output)
    conv4 = Conv2D(128, (5, 5), strides=1, padding='same', activation='relu')(upsample1)
    
    upsample2 = UpSampling2D(size=(2, 2))(conv4)
    conv5 = Conv2D(64, (5, 5), strides=1, padding='same', activation='relu')(upsample2)
    
    upsample3 = UpSampling2D(size=(2, 2))(conv5)
    deblurred_output = Conv2D(3, (5, 5), strides=1, padding='same', activation='linear')(upsample3)
    
    return Model(inputs=input_layer, outputs=deblurred_output)

class FUNIE_GAN_UP():
    def __init__(self, imrow=256, imcol=256, imchan=3):
        self.img_rows, self.img_cols, self.channels = imrow, imcol, imchan
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        optimizer = LegacyAdam(learning_rate=0.0003, beta_1=0.5)
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        noisy_A = Input(shape=self.img_shape)
        noisy_B = Input(shape=self.img_shape)
        self.vgg_content = VGG19_Content()
        self.disc_patch = (16, 16, 1)
        self.n_residual_blocks = 5
        self.gf, self.df = 32, 32

        # Deblurring submodule for both A and B
        deblur_A = deblurring_submodule(self.img_shape)(noisy_A)
        deblur_B = deblurring_submodule(self.img_shape)(noisy_B)

        self.d_A = self.FUNIE_UP_discriminator()
        self.d_B = self.FUNIE_UP_discriminator()
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        self.g_AB = self.FUNIE_UP_generator()
        self.g_BA = self.FUNIE_UP_generator()

        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        self.d_A.trainable = False
        self.d_B.trainable = False

        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Blind denoising sub-module
        denoised_A = blind_denoising_submodule()(noisy_A)
        denoised_B = blind_denoising_submodule()(noisy_B)

        # Combined model
        self.combined = Model(inputs=[img_A, img_B, noisy_A, noisy_B],
                              outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id, denoised_A, denoised_B, deblur_A, deblur_B])
        self.combined.compile(
            loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae', 'mse', 'mse', 'mse', 'mse'],
            loss_weights=[1, 1, 10, 10, 1, 1, 1, 1, 1, 1],
            optimizer=optimizer
        )

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def perceptual_distance(self, y_true, y_pred):
        """
           Calculating perceptual distance
           Thanks to github.com/wandb/superres
        """
        y_true = (y_true+1.0)*127.5 # [-1,1] -> [0, 255]
        y_pred = (y_pred+1.0)*127.5 # [-1,1] -> [0, 255]
        rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
        r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
        g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
        b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
        return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

    def getSSIM(self, X, Y):
        """
           Computes the mean structural similarity between two images.
        """
        assert (X.shape == Y.shape), "Image-patches provided have different dimensions"
        nch = 1 if X.ndim == 2 else X.shape[-1]
        mssim = []
        for ch in range(nch):
            Xc, Yc = X[..., ch].astype(np.float64), Y[..., ch].astype(np.float64)
            mssim.append(self.compute_ssim(Xc, Yc))
        return np.mean(mssim)

    def compute_ssim(self, X, Y):
        """
           Compute the structural similarity per single channel (given two images)
        """
        # variables are initialized as suggested in the paper
        K1 = 0.01
        K2 = 0.03
        sigma = 1.5
        win_size = 5   

        # means
        ux = gaussian_filter(X, sigma)
        uy = gaussian_filter(Y, sigma)

        # variances and covariances
        uxx = gaussian_filter(X * X, sigma)
        uyy = gaussian_filter(Y * Y, sigma)
        uxy = gaussian_filter(X * Y, sigma)

        # normalize by unbiased estimate of std dev 
        N = win_size ** X.ndim
        unbiased_norm = N / (N - 1)  # eq. 4 of the paper
        vx  = (uxx - ux * ux) * unbiased_norm
        vy  = (uyy - uy * uy) * unbiased_norm
        vxy = (uxy - ux * uy) * unbiased_norm

        R = 255
        C1 = (K1 * R) ** 2
        C2 = (K2 * R) ** 2
        # compute SSIM (eq. 13 of the paper)
        sim = (2 * ux * uy + C1) * (2 * vxy + C2)
        D = (ux * 2 + uy * 2 + C1) * (vx + vy + C2)
        SSIM = sim / D 
        mssim = SSIM.mean()

        return mssim
    


    def ssim_loss(self, y_true, y_pred):
        ssim_loss = 1.0 - self.getSSIM(y_true.numpy(), y_pred.numpy())
        return ssim_loss


    def total_gen_loss(self, org_content, gen_content):
        # custom perceptual loss function
        vgg_org_content = self.vgg_content(org_content)
        vgg_gen_content = self.vgg_content(gen_content)
        content_loss = K.mean(K.square(vgg_org_content - vgg_gen_content), axis=-1)
        print("Content Loss :",content_loss)
        mae_gen_loss = K.mean(K.abs(org_content-gen_content))
        print("MAE Loss :", mae_gen_loss)
        # Mean Square Error (MSE) loss
        mse_loss = K.mean(K.square(org_content - gen_content))
        perceptual_loss = self.perceptual_distance(org_content, gen_content)
        print("Perceptual Loss :",perceptual_loss)
        # Structural Similarity Index (SSIM) loss
        ssim_loss = self.ssim_loss(org_content, gen_content)
        #gen_total_err = 0.7*mae_gen_loss+0.3*content_loss # v1
        # updated loss function in v2
        gen_total_err = 0.7*mae_gen_loss+0.2*content_loss+0.1*perceptual_loss+0.5 * mse_loss+ 0.3 * ssim_loss
        return gen_total_err


    def FUNIE_UP_generator(self):
        """
           Inspired by the U-Net Generator with skip connections
           This is a much simpler architecture with fewer parameters
        """
        def conv2d(layer_input, filters, f_size=3, bn=True):
            ## for downsampling
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn: d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            ## for upsampling
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate: u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u
        color_correction_input = Input(shape=(256,))  # Adjust the input shape based on your model
        color_correction_output = color_correction_module(color_correction_input)

        # Connect color correction module to the generator
        input_layer = Input(shape=self.img_shape)
        d0, d5 = input_layer, None

        # Connect color correction module to the generator
        #generator_output = self.generator_output  # Replace with your actual generator output
        #generator_output_with_correction = Concatenate()([generator_output, color_correction_output])

        #return Model(inputs=[input_layer, color_correction_input], outputs=generator_output_with_correction)



        def _residual_block(ip):
            init = ip
            x = Conv2D(64, (3, 3), activation='linear', padding='same')(ip)
            x = BatchNormalization(momentum=0.8)(x, training=False)
            x = Activation('relu')(x)
            x = Conv2D(64, (3, 3), activation='linear', padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x, training=False)
            m = Add()([x, init])
            return m

        print("Printing Generator model architecture")
        ## input
        d0 = Input(shape=self.img_shape); print(d0)
        ## downsample
        d1 = conv2d(d0, self.gf*1, f_size=5, bn=False) ;print(d1)
        d2 = conv2d(d1, self.gf*4, f_size=4, bn=True)  ;print(d2)
        d3 = conv2d(d2, self.gf*8, f_size=4, bn=True)  ;print(d3)
        d4 = conv2d(d3, self.gf*8, f_size=3, bn=True)  ;print(d4)
        d5 = conv2d(d4, self.gf*8, f_size=3, bn=True)  ;print(d5); print();

        # three additional conv layers
        x0 = Conv2D(64, (1, 1), activation='relu', padding='same')(d5)
        x1 = Conv2D(64, (3, 3), activation='leaky_relu', padding='same', strides=1)(x0)
        x2 = Conv2D(64, (5, 5), activation='swish', padding='same', strides=1)(x1)
        x3 = Conv2D(64, (7, 7), activation='elu', padding='same', strides=1)(x2)
        x4 = Conv2D(64, (9, 9), activation='selu', padding='same', strides=1)(x3)
        # additional res layers
        x = _residual_block(x4)
        for i in range(self.n_residual_blocks - 1):
            x = _residual_block(x)
        # skip connect and up-scale
        x = Add()([x, x0])

        ## now upsample
        u1 = deconv2d(x, d4, self.gf*8) ;print(u1)
        u2 = deconv2d(u1, d3, self.gf*8) ;print(u2)
        u3 = deconv2d(u2, d2, self.gf*4) ;print(u3)
        u4 = deconv2d(u3, d1, self.gf*1) ;print(u4)
        u5 = UpSampling2D(size=2)(u4)    ;print(u5)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)
        print(output_img); print();

        # Output layer
        generator_output = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)
        print(generator_output); print()

        # Connect color correction module to the generator output
        generator_output_with_correction = Concatenate()([generator_output, color_correction_output])

        return Model(inputs=[d0, color_correction_input], outputs=generator_output_with_correction)



    def FUNIE_UP_discriminator(self):
        """
           Inspired by the pix2pix discriminator
        """
        def d_layer(layer_input, filters, strides_=2, f_size=3, bn=True):
            ## Discriminator layers
            d = Conv2D(filters, kernel_size=f_size, strides=strides_, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn: d = BatchNormalization(momentum=0.8)(d)
            return d

        print("Printing Discriminator model architecture")
        ## input
        img = Input(shape=self.img_shape)
        ## Discriminator layers
        d1 = d_layer(img, self.df, bn=False) ; print(d1)
        d2 = d_layer(d1, self.df*2) ; print(d2)
        d3 = d_layer(d2, self.df*4) ; print(d3)
        d4 = d_layer(d3, self.df*8) ; print(d4)
        d5 = d_layer(d4, self.df*8, strides_=1) ; print(d5)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)
        print(validity); print()

        return Model(img, validity)
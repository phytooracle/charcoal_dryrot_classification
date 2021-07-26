from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Add
from tensorflow.keras.layers import SpatialDropout3D, UpSampling3D, Dropout, RepeatVector, Average
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from Utils import weighted_dice_coef
import tensorflow as tf
import tensorflow.keras.layers as layers

class ResNet:

    def __init__(self,dic):

        model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_shape=dic['input_shape'], pooling=None, classes=2)

        opt = eval(dic['optimizer'])

        loss = dic['loss']
        
        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['accuracy'])

        self.model = model

class MobileNetV1:

    def __init__(self,dic):

        model = tf.keras.applications.mobilenet.MobileNet(include_top=True, weights=None, input_shape=dic['input_shape'], pooling=None, classes=2)

        opt = eval(dic['optimizer'])

        loss = dic['loss']
        
        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['accuracy'])

        self.model = model

class MobileNetV2:

    def __init__(self,dic):

        model = tf.keras.applications.mobilenet.MobileNetV2(include_top=True, weights=None, input_shape=dic['input_shape'], pooling=None, classes=2)

        opt = eval(dic['optimizer'])

        loss = dic['loss']
        
        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['accuracy'])

        self.model = model

class UNET:

    def __init__(self, dic):

        input_layer = Input(dic['input_shape'])
        
        x_init = input_layer
        
        x_conv1_b1 = Conv2D(dic['start_filter'], dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_init)
        x_conv2_b1 = Conv2D(dic['start_filter'], dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_conv1_b1)
        x_max_b1 = MaxPool2D([2, 2],padding='same')(x_conv2_b1)
        x_bn_b1 = BatchNormalization()(x_max_b1)
        x_do_b1 = Dropout(dic['dr_rate'])(x_bn_b1)

        x_conv1_b2 = Conv2D(dic['start_filter']*2, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_do_b1)
        x_conv2_b2 = Conv2D(dic['start_filter']*2, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_conv1_b2)
        x_max_b2 = MaxPool2D([2, 2],padding='same')(x_conv2_b2)
        x_bn_b2 = BatchNormalization()(x_max_b2)
        x_do_b2 = Dropout(dic['dr_rate'])(x_bn_b2)

        x_conv1_b3 = Conv2D(dic['start_filter']*4, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_do_b2)
        x_conv2_b3 = Conv2D(dic['start_filter']*4, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_conv1_b3)
        x_max_b3 = MaxPool2D([2, 2],padding='same')(x_conv2_b3)
        x_bn_b3 = BatchNormalization()(x_max_b3)
        x_do_b3 = Dropout(dic['dr_rate'])(x_bn_b3)

        x_conv1_b4 = Conv2D(dic['start_filter']*8, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_do_b3)
        x_conv2_b4 = Conv2D(dic['start_filter']*8, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_conv1_b4)
        x_max_b4 = MaxPool2D([2, 2],padding='same')(x_conv2_b4)
        x_bn_b4 = BatchNormalization()(x_max_b4)
        x_do_b4 = Dropout(dic['dr_rate'])(x_bn_b4)

        # ------- Head Normal Output (normal decoder)

        x_conv1_b5 = Conv2D(dic['start_filter']*16, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_do_b4)
        x_conv2_b5 = Conv2D(dic['start_filter']*16, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_conv1_b5)
        x_deconv_b5 = Conv2DTranspose(dic['start_filter']*8, dic['deconv_kernel_size'] ,(2,2),padding='same', activation=dic['activation'])(x_conv2_b5)
        x_bn_b5 = BatchNormalization()(x_deconv_b5)
        x_do_b5 = Dropout(dic['dr_rate'])(x_bn_b5)

        x_conv1_b6 = Conv2D(dic['start_filter']*8, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(layers.concatenate([x_conv2_b4,x_do_b5]))
        x_conv2_b6 = Conv2D(dic['start_filter']*8, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_conv1_b6)
        x_deconv_b6 = Conv2DTranspose(dic['start_filter']*4, dic['deconv_kernel_size'] ,(2,2),padding='same', activation=dic['activation'])(x_conv2_b6)
        x_bn_b6 = BatchNormalization()(x_deconv_b6)
        x_do_b6 = Dropout(dic['dr_rate'])(x_bn_b6)

        x_conv1_b7 = Conv2D(dic['start_filter']*4, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(layers.concatenate([x_conv2_b3,x_do_b6]))
        x_conv2_b7 = Conv2D(dic['start_filter']*4, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_conv1_b7)
        x_deconv_b7 = Conv2DTranspose(dic['start_filter']*2, dic['deconv_kernel_size'] ,(2,2),padding='same', activation=dic['activation'])(x_conv2_b7)
        x_bn_b7 = BatchNormalization()(x_deconv_b7)
        x_do_b7 = Dropout(dic['dr_rate'])(x_bn_b7)

        x_conv1_b8 = Conv2D(dic['start_filter']*2, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(layers.concatenate([x_conv2_b2,x_do_b7]))
        x_conv2_b8 = Conv2D(dic['start_filter']*2, dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_conv1_b8)
        x_deconv_b8 = Conv2DTranspose(dic['start_filter'], dic['deconv_kernel_size'] ,(2,2),padding='same', activation=dic['activation'])(x_conv2_b8)
        x_bn_b8 = BatchNormalization()(x_deconv_b8)
        x_do_b8 = Dropout(dic['dr_rate'])(x_bn_b8)

        x_conv1_b9 = Conv2D(dic['start_filter'], dic['conv_kernel_size'],padding='same', activation=dic['activation'])(layers.concatenate([x_conv2_b1,x_do_b8]))
        x_conv2_b9 = Conv2D(dic['start_filter'], dic['conv_kernel_size'],padding='same', activation=dic['activation'])(x_conv1_b9)
        x_bn_b9 = BatchNormalization()(x_conv2_b9)
        x_do_b9 = Dropout(dic['dr_rate'])(x_bn_b9)

        normal_output = Conv2D(1, [1, 1], activation='sigmoid')(x_do_b9)

        # ----------

        model = Model(inputs=[input_layer], outputs=[normal_output])
        
        opt = eval(dic['optimizer'])

        if dic['loss'] == "weighted_dice_coef":
            loss = weighted_dice_coef
        else:
            loss = dic['loss']
        
        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['accuracy'])
        
        self.model = model

# test_dic = {
#     'input_shape':[256,256,3],
#     'start_filter':64,
#     'conv_kernel_size':3,
#     'activation':'relu',
#     'dr_rate':0.2,
#     'deconv_kernel_size':3,
#     'optimizer':'SGD',
#     'loss':'binary_crossentropy',
#     'lr':0.001
# }

# unet = UNET(test_dic)
# unet.model.summary()

# resnet = ResNet(test_dic)
# resnet.model.summary()

# resnet = MobileNetV1(test_dic)
# resnet.model.summary()
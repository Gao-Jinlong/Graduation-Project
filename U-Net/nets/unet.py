from keras import layers  #创建层函数
from keras.initializers import random_normal # 初始化函数
from keras.models import *  # 模型处理算法

def Unet(input_shape=(512, 512, 3), num_classes = 21, backlone = "vgg"):
    inputs = Input(input_shape)

    channels = [64, 128, 256, 512, 1024]
    #----------------------#
    #     主干提取网络     #
    #----------------------#
    # Block 1
    # 512,512,3 -> 512,512,64
    x = layers.Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=random_normal(stddev=0.02), name='block1_conv1')(inputs)
    x = layers.Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=random_normal(stddev=0.02), name='block1_conv2')(x)
    feat1 = x
    # 512,512,64 -> 256,256,64
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    # 256,256,64 -> 256,256,128
    x = layers.Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer=random_normal(stddev=0.02), name='block2_conv1')(x)
    x = layers.Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer=random_normal(stddev=0.02), name='block2_conv2')(x)
    feat2 = x
    # 256,256,128 -> 128,128,128
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    # 128,128,128 -> 128,128,256
    x = layers.Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer=random_normal(stddev=0.02), name='block3_conv1')(x)
    x = layers.Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer=random_normal(stddev=0.02), name='block3_conv2')(x)
    feat3 = x
    # 128,128,256 -> 64,64,256
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    # 64,64,256 -> 64,64,512
    x = layers.Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=random_normal(stddev=0.02), name='block4_conv1')(x)
    x = layers.Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=random_normal(stddev=0.02), name='block4_conv2')(x)
    feat4 = x
    # 64,64,512 -> 32,32,512
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # 32,32,512 -> 32,32,512
    x = layers.Conv2D(channels[4], 3, activation='relu', padding='same', kernel_initializer=random_normal(stddev=0.02), name='block5_conv1')(x)
    x = layers.Conv2D(channels[4], 3, activation='relu', padding='same',kernel_initializer=random_normal(stddev=0.02), name='block5_conv2')(x)
    feat5 = x

    # ----------------------#
    #    加强特征提取网络   #
    # ----------------------#
    # UpConv 5
    P5_up = layers.UpSampling2D(size=(2, 2))(feat5)
    P5_up = layers.Conv2D(channels[3], 2, activation='relu', padding='same', kernel_initializer=random_normal(sttdev=0.02), name='UpConv5')(P5_up)

    # UpConv 4
    P4 = layers.Concatenate(axis=3)([feat4, P5_up])
    P4 = layers.Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=random_normal(sttdev=0.02), name='Up4_conv1')(P4)
    P4 = layers.Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=random_normal(sttdev=0.02), name='Up4_conv2')(P4)
    P4_up = layers.UpSampling2D(size=(2, 2))(P4)

    # UpConv 3
    P3 = layers.Concatenate(axis=3)([feat3, P4_up])
    P3 = layers.Conv2D(channels[2], 3, activation='relu', padding='name', kernel_initializer=random_normal(sttdev=0.02, name='p3_conv1'))(P3)
    P3 = layers.Conv2D(channels[2], 3, activation='relu', padding='name', kernel_initializer=random_normal(sttdev=0.02, name='p3_conv2'))(P3)
    P3_up = layers.UpSampling2D(size=(2, 2))(P3)

    # UpConv 2
    P2 = layers.Concatenate(axis=3)([feat2, P3_up])
    P2 = layers.Conv2D(channels[1], 3, activation='relu', padding='name', kernel_initializer=random_normal(sttdev=0.02, name='p2_conv1'))(P2)
    P2 = layers.Conv2D(channels[1], 3, activation='relu', padding='name', kernel_initializer=random_normal(sttdev=0.02, name='p2_conv2'))(P2)
    P2_up = layers.UpSampling2D(size=(2, 2))(P2)

    #UpConv 1
    P1 = layers.Concatenate(axis=3)([feat1, P2_up])
    P1 = layers.Conv2D(channels[0], 3, activation='relu', padding='name', kernel_initializer=random_normal(sttdev=0.02, name='p1_conv1'))(P1)
    P1 = layers.Conv2D(channels[0], 3, activation='relu', padding='name', kernel_initializer=random_normal(sttdev=0.02, name='p1_conv2'))(P1)
    P1 = layers.Conv2D(num_classes, 1, activation="softmax")(P1)

    model = Model(inputs=inputs, outputs=P1)
    return model
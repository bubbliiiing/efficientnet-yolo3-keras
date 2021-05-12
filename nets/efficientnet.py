import math
from copy import deepcopy

import tensorflow as tf
from keras import backend, layers
from keras.initializers import random_normal

#-------------------------------------------------#
#   一共七个大结构块，每个大结构块都有特定的参数
#-------------------------------------------------#
DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},

    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},

    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},

    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},

    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},

    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},

    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

#-------------------------------------------------#
#   Kernel的初始化器
#-------------------------------------------------#
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

#-------------------------------------------------#
#   用于计算卷积层的padding的大小
#-------------------------------------------------#
def correct_pad(inputs, kernel_size):
    img_dim = 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

#-------------------------------------------------#
#   该函数的目的是保证filter的大小可以被8整除
#-------------------------------------------------#
def round_filters(filters, divisor, width_coefficient):
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

#-------------------------------------------------#
#   计算模块的重复次数
#-------------------------------------------------#
def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))

#-------------------------------------------------#
#   efficient_block
#-------------------------------------------------#
def block(inputs, activation_fn=tf.nn.swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):

    filters = filters_in * expand_ratio
    #-------------------------------------------------#
    #   利用Inverted residuals
    #   part1 利用1x1卷积进行通道数上升
    #-------------------------------------------------#
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=3, name=name + 'expand_bn')(x)
        x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    #------------------------------------------------------#
    #   如果步长为2x2的话，利用深度可分离卷积进行高宽压缩
    #   part2 利用3x3卷积对每一个channel进行卷积
    #------------------------------------------------------#
    if strides == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(x, kernel_size),
                                 name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
        
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=3, name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)

    #------------------------------------------------------#
    #   完成深度可分离卷积后
    #   对深度可分离卷积的结果施加注意力机制
    #------------------------------------------------------#
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        #------------------------------------------------------#
        #   通道先压缩后上升，最后利用sigmoid将值固定到0-1之间
        #------------------------------------------------------#
        se = layers.Conv2D(filters_se, 1,
                           padding='same',
                           activation=activation_fn,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_reduce')(se)
        se = layers.Conv2D(filters, 1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    #------------------------------------------------------#
    #   part3 利用1x1卷积进行通道下降
    #------------------------------------------------------#
    x = layers.Conv2D(filters_out, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=3, name=name + 'project_bn')(x)

    #------------------------------------------------------#
    #   part4 如果满足残差条件，那么就增加残差边
    #------------------------------------------------------#
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1),
                               name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')

    return x

def EfficientNet(width_coefficient,
                 depth_coefficient,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=tf.nn.swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 inputs=None,
                 **kwargs):
    img_input = inputs

    #-------------------------------------------------#
    #   创建stem部分
    #   416,416,3 -> 208,208,32
    #-------------------------------------------------#
    x = img_input
    x = layers.ZeroPadding2D(padding=correct_pad(x, 3),
                             name='stem_conv_pad')(x)
    x = layers.Conv2D(round_filters(32, depth_divisor, width_coefficient), 3,
                      strides=2,
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=3, name='stem_bn')(x)
    x = layers.Activation(activation_fn, name='stem_activation')(x)

    #-------------------------------------------------#
    #   进行一个深度的copy
    #-------------------------------------------------#
    blocks_args = deepcopy(blocks_args)
    
    #-------------------------------------------------#
    #   计算总的efficient_block的数量
    #-------------------------------------------------#
    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))

    feats = []
    filters_outs = []
    #------------------------------------------------------------------------------#
    #   对结构块参数进行循环、一共进行7个大的结构块。
    #   每个大结构块下会重复小的efficient_block、每个大结构块的shape变化为：
    #   208,208,32 -> 208,208,16 -> 104,104,24 -> 52,52,40 
    #   -> 26,26,80 -> 26,26,112 -> 13,13,192 -> 13,13,320
    #   输入为208,208,32，最终获得三个shape的有效特征层
    #   104,104,24、26,26,112、13,13,320
    #------------------------------------------------------------------------------#
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        args['filters_in'] = round_filters(args['filters_in'], depth_divisor, width_coefficient)
        args['filters_out'] = round_filters(args['filters_out'], depth_divisor, width_coefficient)

        for j in range(round_repeats(args.pop('repeats'), depth_coefficient)):
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation_fn, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1
        feats.append(x)
        if i == 2 or i == 4 or i == 6:
            filters_outs.append(args['filters_out'])
    return feats, filters_outs


def EfficientNetB0(inputs=None, **kwargs):
    return EfficientNet(1.0, 1.0, inputs=inputs, **kwargs)


def EfficientNetB1(inputs=None, **kwargs):
    return EfficientNet(1.0, 1.1, inputs=inputs, **kwargs)


def EfficientNetB2(inputs=None, **kwargs):
    return EfficientNet(1.1, 1.2, inputs=inputs, **kwargs)


def EfficientNetB3(inputs=None, **kwargs):
    return EfficientNet(1.2, 1.4, inputs=inputs, **kwargs)


def EfficientNetB4(inputs=None, **kwargs):
    return EfficientNet(1.4, 1.8, inputs=inputs, **kwargs)


def EfficientNetB5(inputs=None, **kwargs):
    return EfficientNet(1.6, 2.2, inputs=inputs, **kwargs)



def EfficientNetB6(inputs=None, **kwargs):
    return EfficientNet(1.8, 2.6, inputs=inputs, **kwargs)


def EfficientNetB7(inputs=None, **kwargs):
    return EfficientNet(2.0, 3.1, inputs=inputs, **kwargs)


if __name__ == '__main__':
    print(EfficientNetB0())

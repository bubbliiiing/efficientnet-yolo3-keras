from functools import wraps

from keras.initializers import random_normal
from keras.layers import (BatchNormalization, Concatenate, Conv2D, Input,
                          Lambda, LeakyReLU, UpSampling2D)
from keras.models import Model
from keras.regularizers import l2
from utils.utils import compose

from nets.efficientnet import (EfficientNetB0, EfficientNetB1, EfficientNetB2,
                               EfficientNetB3, EfficientNetB4, EfficientNetB5,
                               EfficientNetB6, EfficientNetB7)
from nets.yolo_training import yolo_loss

Efficient = [EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7]

#------------------------------------------------------#
#   单次卷积
#   DarknetConv2D
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'   
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)
    
#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def make_five_conv(x, num_filters, weight_decay=5e-4):
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=weight_decay)(x)
    return x

def make_yolo_head(x, num_filters, out_filters, weight_decay=5e-4):
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=weight_decay)(x)
    y = DarknetConv2D(out_filters, (1,1), weight_decay=weight_decay)(y)
    return y

#---------------------------------------------------#
#   FPN网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes, phi = 0, weight_decay=5e-4):
    inputs      = Input(input_shape)
    #---------------------------------------------------#   
    #   生成efficientnet的主干模型，以efficientnetB0为例
    #   获得三个有效特征层，他们的shape分别是：
    #   52, 52, 40
    #   26, 26, 112
    #   13, 13, 320
    #---------------------------------------------------#
    feats, filters_outs = Efficient[phi](inputs = inputs)
    feat1 = feats[2]
    feat2 = feats[4]
    feat3 = feats[6]

    #------------------------------------------------------------------------#
    #   以efficientnet网络的输出通道数，构建FPN
    #------------------------------------------------------------------------#

    x   = make_five_conv(feat3, int(filters_outs[2]), weight_decay)
    #---------------------------------------------------#
    #   第一个特征层
    #   out0 = (batch_size, 255, 13, 13)
    #---------------------------------------------------#
    P5  = make_yolo_head(x, int(filters_outs[2]), len(anchors_mask[0]) * (num_classes+5), weight_decay)

    x   = compose(DarknetConv2D_BN_Leaky(int(filters_outs[1]), (1,1), weight_decay=weight_decay), UpSampling2D(2))(x)

    x   = Concatenate()([x, feat2])
    x   = make_five_conv(x, int(filters_outs[1]), weight_decay)
    #---------------------------------------------------#
    #   第二个特征层
    #   out1 = (batch_size, 255, 26, 26)
    #---------------------------------------------------#
    P4  = make_yolo_head(x, int(filters_outs[1]), len(anchors_mask[1]) * (num_classes+5), weight_decay)

    x   = compose(DarknetConv2D_BN_Leaky(int(filters_outs[0]), (1,1), weight_decay=weight_decay), UpSampling2D(2))(x)

    x   = Concatenate()([x, feat1])
    x   = make_five_conv(x, int(filters_outs[0]), weight_decay)
    #---------------------------------------------------#
    #   第三个特征层
    #   out3 = (batch_size, 255, 52, 52)
    #---------------------------------------------------#
    P3  = make_yolo_head(x, int(filters_outs[0]), len(anchors_mask[2]) * (num_classes+5), weight_decay)
    return Model(inputs, [P5, P4, P3])

def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {
            'input_shape'       : input_shape, 
            'anchors'           : anchors, 
            'anchors_mask'      : anchors_mask, 
            'num_classes'       : num_classes, 
            'balance'           : [0.4, 1.0, 4],
            'box_ratio'         : 0.05,
            'obj_ratio'         : 5 * (input_shape[0] * input_shape[1]) / (416 ** 2), 
            'cls_ratio'         : 1 * (num_classes / 80)
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model

import os
import tensorflow as tf
import numpy as np
import inspect
from math import ceil

class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """


        self.conv1_1 = self.conv_layer(rgb, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")



    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            bn = tf.contrib.layers.batch_norm(bias)
            relu = tf.nn.relu(bn)
            return relu


    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")


def Saliency_Prediction(x, name):
    with tf.variable_scope(name):

        VGG = Vgg19('./vgg19.npy')
        VGG.build(x)

        f1_1 = VGG.conv1_1
        f1_2 = VGG.conv1_2
        f2_1 = VGG.conv2_1
        f2_2 = VGG.conv2_2
        f3_1 = VGG.conv3_1
        f3_2 = VGG.conv3_2
        f3_3 = VGG.conv3_3
        f3_4 = VGG.conv3_4
        f4_1 = VGG.conv4_1
        f4_2 = VGG.conv4_2
        f4_3 = VGG.conv4_3
        f4_4 = VGG.conv4_4
        f5_1 = VGG.conv5_1
        f5_2 = VGG.conv5_2
        f5_3 = VGG.conv5_3
        f5_4 = VGG.conv5_4

        f5_4_32 = cnn_layer_bn(f5_4, [1, 1, 512, 32], [32], 'f5_4_32')
        f5_3_32 = cnn_layer_bn(f5_3, [1, 1, 512, 32], [32], 'f5_3_32')
        f5_2_32 = cnn_layer_bn(f5_2, [1, 1, 512, 32], [32], 'f5_2_32')
        f5_1_32 = cnn_layer_bn(f5_1, [1, 1, 512, 32], [32], 'f5_1_32')
        f5 = cnn_layer_bn(tf.concat([f5_4_32, f5_3_32, f5_2_32, f5_1_32], axis=3), [1, 1, 128, 64], [64], 'f5')

        f4_4_32 = cnn_layer_bn(f4_4, [1, 1, 512, 32], [32], 'f4_4_32')
        f4_3_32 = cnn_layer_bn(f4_3, [1, 1, 512, 32], [32], 'f4_3_32')
        f4_2_32 = cnn_layer_bn(f4_2, [1, 1, 512, 32], [32], 'f4_2_32')
        f4_1_32 = cnn_layer_bn(f4_1, [1, 1, 512, 32], [32], 'f4_1_32')
        f4 = cnn_layer_bn(tf.concat([f4_4_32, f4_3_32, f4_2_32, f4_1_32], axis=3), [1, 1, 128, 64], [64], 'f4')

        f3_4_32 = cnn_layer_bn(f3_4, [1, 1, 256, 32], [32], 'f3_4_32')
        f3_3_32 = cnn_layer_bn(f3_3, [1, 1, 256, 32], [32], 'f3_3_32')
        f3_2_32 = cnn_layer_bn(f3_2, [1, 1, 256, 32], [32], 'f3_2_32')
        f3_1_32 = cnn_layer_bn(f3_1, [1, 1, 256, 32], [32], 'f3_1_32')
        f3 = cnn_layer_bn(tf.concat([f3_4_32, f3_3_32, f3_2_32, f3_1_32], axis=3), [1, 1, 128, 64], [64], 'f3')

        f2_2_32 = cnn_layer_bn(f2_2, [1, 1, 128, 32], [32], 'f2_2_32')
        f2_1_32 = cnn_layer_bn(f2_1, [1, 1, 128, 32], [32], 'f2_1_32')
        f2 = cnn_layer_bn(tf.concat([f2_2_32, f2_1_32], axis=3), [1, 1, 64, 64], [64], 'f2')

        f1_2_32 = cnn_layer_bn(f1_2, [1, 1, 64, 32], [32], 'f1_2_32')
        f1_1_32 = cnn_layer_bn(f1_1, [1, 1, 64, 32], [32], 'f1_1_32')
        f1 = cnn_layer_bn(tf.concat([f1_2_32, f1_1_32], axis=3), [1, 1, 64, 64], [64], 'f1')

        f5_p = cnn_layer(f5, [3, 3, 64, 1], [1], 'f5_p')
        f5_up = deconv_layer(f5_p, [2, 2, 1, 1], [14, 32, 32, 1], 2, 'f5_up')

        f4_p = cnn_layer(tf.concat([f5_up, f4], axis=3), [3, 3, 65, 1], [1], 'f4_p')
        f4_up = deconv_layer(f4_p, [2, 2, 1, 1], [14, 64, 64, 1], 2, 'f4_up')

        f3_p = cnn_layer(tf.concat([f4_up, f3], axis=3), [3, 3, 65, 1], [1], 'f3_p')
        f3_up = deconv_layer(f3_p, [2, 2, 1, 1], [14, 128, 128, 1], 2, 'f3_up')

        f2_p = cnn_layer(tf.concat([f3_up, f2], axis=3), [3, 3, 65, 1], [1], 'f2_p')
        f2_up = deconv_layer(f2_p, [2, 2, 1, 1], [14, 256, 256, 1], 2, 'f2_up')

        f1_p = cnn_layer(tf.concat([f2_up, f1], axis=3), [3, 3, 65, 1], [1], 'f1_p')

        return f5_p, f4_p, f3_p, f2_p, f1_p

def multi_view_attention(p1, p2, p3, p4, p5, depth_16, depth_32, depth_64, depth_128, depth_256, name):
    with tf.variable_scope(name):

        p1 = tf.expand_dims(p1, dim=4)
        depth_256 = tf.expand_dims(depth_256, dim=4)
        p1_depth = tf.concat([p1, depth_256], axis=4)
        p1c1 = cnn_layer_3D(p1_depth, [3, 3, 3, 2, 8], [8], 'p1c1')
        p1c2 = cnn_layer_3D(p1c1, [3, 3, 3, 8, 8], [8], 'p1c2')
        p1_r = tf.reshape(p1c2, [1, 256, 256, 112])
        ac1 = tf.nn.avg_pool(p1_r, [1, 256, 256, 1], [1, 256, 256, 1], padding='SAME')
        ac1 = cnn_layer(ac1, [1, 1, 112, 14], [14], 'ac1')
        ac1 = tf.nn.softmax(ac1, axis=3)
        ac1 = tf.expand_dims(ac1, axis=4)
        ac1 = tf.tile(ac1, [1, 256, 256, 1, 1])
        p1_ca = tf.multiply(p1, ac1)
        pre1 = tf.reduce_sum(p1_ca, axis=3)
        pre1 = tf.reshape(pre1, [1, 256, 256, 1])

        p2 = tf.expand_dims(p2, dim=4)
        depth_128 = tf.expand_dims(depth_128, dim=4)
        p2_depth = tf.concat([p2, depth_128], axis=4)
        p2c1 = cnn_layer_3D(p2_depth, [3, 3, 3, 2, 8], [8], 'p2c1')
        p2c2 = cnn_layer_3D(p2c1, [3, 3, 3, 8, 8], [8], 'p2c2')
        p2_r = tf.reshape(p2c2, [1, 128, 128, 112])
        ac2 = tf.nn.avg_pool(p2_r, [1, 128, 128, 1], [1, 128, 128, 1], padding='SAME')
        ac2 = cnn_layer(ac2, [1, 1, 112, 14], [14], 'ac2')
        ac2 = tf.nn.softmax(ac2, axis=3)
        ac2 = tf.expand_dims(ac2, axis=4)
        ac2 = tf.tile(ac2, [1, 128, 128, 1, 1])
        p2_ca = tf.multiply(p2, ac2)
        pre2 = tf.reduce_sum(p2_ca, axis=3)
        pre2 = tf.reshape(pre2, [1, 128, 128, 1])

        p3 = tf.expand_dims(p3, dim=4)
        depth_64 = tf.expand_dims(depth_64, dim=4)
        p3_depth = tf.concat([p3, depth_64], axis=4)
        p3c1 = cnn_layer_3D(p3_depth, [3, 3, 3, 2, 8], [8], 'p3c1')
        p3c2 = cnn_layer_3D(p3c1, [3, 3, 3, 8, 8], [8], 'p3c2')
        p3_r = tf.reshape(p3c2, [1, 64, 64, 112])
        ac3 = tf.nn.avg_pool(p3_r, [1, 64, 64, 1], [1, 64, 64, 1], padding='SAME')
        ac3 = cnn_layer(ac3, [1, 1, 112, 14], [14], 'ac3')
        ac3 = tf.nn.softmax(ac3, axis=3)
        ac3 = tf.expand_dims(ac3, axis=4)
        ac3 = tf.tile(ac3, [1, 64, 64, 1, 1])
        p3_ca = tf.multiply(p3, ac3)
        pre3 = tf.reduce_sum(p3_ca, axis=3)
        pre3 = tf.reshape(pre3, [1, 64, 64, 1])

        p4 = tf.expand_dims(p4, dim=4)
        depth_32 = tf.expand_dims(depth_32, dim=4)
        p4_depth = tf.concat([p4, depth_32], axis=4)
        p4c1 = cnn_layer_3D(p4_depth, [3, 3, 3, 2, 8], [8], 'p4c1')
        p4c2 = cnn_layer_3D(p4c1, [3, 3, 3, 8, 8], [8], 'p4c2')
        p4_r = tf.reshape(p4c2, [1, 32, 32, 112])
        ac4 = tf.nn.avg_pool(p4_r, [1, 32, 32, 1], [1, 32, 32, 1], padding='SAME')
        ac4 = cnn_layer(ac4, [1, 1, 112, 14], [14], 'ac4')
        ac4 = tf.nn.softmax(ac4, axis=3)
        ac4 = tf.expand_dims(ac4, axis=4)
        ac4 = tf.tile(ac4, [1, 32, 32, 1, 1])
        p4_ca = tf.multiply(p4, ac4)
        pre4 = tf.reduce_sum(p4_ca, axis=3)
        pre4 = tf.reshape(pre4, [1, 32, 32, 1])

        p5 = tf.expand_dims(p5, dim=4)
        depth_16 = tf.expand_dims(depth_16, dim=4)
        p5_depth = tf.concat([p5, depth_16], axis=4)
        p5c1 = cnn_layer_3D(p5_depth, [3, 3, 3, 2, 8], [8], 'p5c1')
        p5c2 = cnn_layer_3D(p5c1, [3, 3, 3, 8, 8], [8], 'p5c2')
        p5_r = tf.reshape(p5c2, [1, 16, 16, 112])
        ac5 = tf.nn.avg_pool(p5_r, [1, 16, 16, 1], [1, 16, 16, 1], padding='SAME')
        ac5 = cnn_layer(ac5, [1, 1, 112, 14], [14], 'ac5')
        ac5 = tf.nn.softmax(ac5, axis=3)
        ac5 = tf.expand_dims(ac5, axis=4)
        ac5 = tf.tile(ac5, [1, 16, 16, 1, 1])
        p5_ca = tf.multiply(p5, ac5)
        pre5 = tf.reduce_sum(p5_ca, axis=3)
        pre5 = tf.reshape(pre5, [1, 16, 16, 1])

        return pre5, pre4, pre3, pre2, pre1

def weight_variable(w_shape):
    return tf.get_variable('weights', w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


def bias_variable(b_shape, init_bias=0.0):
    return tf.get_variable('bias', b_shape, initializer=tf.constant_initializer(init_bias))


def cnn_layer_bn(input_tensor, w_shape, b_shape, layer_name, ds=1):
    with tf.variable_scope(layer_name):
        W = weight_variable(w_shape)
        h = tf.nn.conv2d(input_tensor, W, strides=[1, ds, ds, 1], padding='SAME')
        h = tf.contrib.layers.batch_norm(h + bias_variable(b_shape))
        h = tf.nn.relu(h)
        return h

def cnn_layer(input_tensor, w_shape, b_shape, layer_name, ds=1):
    with tf.variable_scope(layer_name):
        W = weight_variable(w_shape)
        h = tf.nn.conv2d(input_tensor, W, strides=[1, ds, ds, 1], padding='SAME')
        h = h + bias_variable(b_shape)
        return h


def cnn_layer_3D(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    with tf.variable_scope(layer_name):
        W = weight_variable(w_shape)
        pad_amt_0 = rate * (w_shape[0] - 1) // 2
        pad_amt_1 = rate * (w_shape[1] - 1) // 2
        pad_amt_2 = rate * (w_shape[2] - 1) // 2
        input_tensor = tf.pad(input_tensor,
                              [[0, 0], [pad_amt_0, pad_amt_0], [pad_amt_1, pad_amt_1], [pad_amt_2, pad_amt_2], [0, 0]],
                              mode='SYMMETRIC')
        h = tf.nn.convolution(input_tensor, W, strides=[ds, ds, ds], padding='VALID', dilation_rate=[rate, rate, rate],
                              name=layer_name + '_conv')
        h = tf.contrib.layers.instance_norm(h + bias_variable(b_shape))
        h = tf.nn.leaky_relu(h)
        return h


def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)


def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):

  # output_shape = [b, w, h, c]

  sess_temp = tf.global_variables_initializer()
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv

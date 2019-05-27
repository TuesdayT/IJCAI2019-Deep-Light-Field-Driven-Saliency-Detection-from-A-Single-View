import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

from Saliency_model import Saliency_Prediction
from Saliency_model import multi_view_attention

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}

# hyperparameters settings

lfsize = [256, 256, 7, 7]     # dimensions of Lytro light fields
lfsize_128 = [128, 128, 7, 7]
lfsize_64 = [64, 64, 7, 7]
lfsize_32 = [32, 32, 7, 7]
lfsize_16 = [16, 16, 7, 7]
batchsize = 1    # modify based on user's GPU memory
center_size = [256, 256]    # spatial dimensions of training light fields
disp_mult = 2.0    # max disparity between adjacent veiws
learning_rate_depth = 0.00001
learning_rate_salmap = 0.001
train_iters = 200000
is_training = 0

# functions for CNN layers

def weight_variable(w_shape):
    return tf.get_variable('weights', w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


def bias_variable(b_shape, init_bias=0.0):
    return tf.get_variable('bias', b_shape, initializer=tf.constant_initializer(init_bias))


def cnn_layer(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    with tf.variable_scope(layer_name):
        W = weight_variable(w_shape)
        pad_amt_0 = rate * (w_shape[0] - 1) // 2
        pad_amt_1 = rate * (w_shape[1] - 1) // 2
        input_tensor = tf.pad(input_tensor, [[0, 0], [pad_amt_0, pad_amt_0], [pad_amt_1, pad_amt_1], [0, 0]],
                              mode='SYMMETRIC')
        h = tf.nn.convolution(input_tensor, W, strides=[ds, ds], padding='VALID', dilation_rate=[rate, rate],
                              name=layer_name + '_conv')
        h = tf.contrib.layers.instance_norm(h + bias_variable(b_shape))
        h = tf.nn.leaky_relu(h)
        return h


def cnn_layer_plain(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    with tf.variable_scope(layer_name):
        W = weight_variable(w_shape)
        pad_amt_0 = rate * (w_shape[0] - 1) // 2
        pad_amt_1 = rate * (w_shape[1] - 1) // 2
        input_tensor = tf.pad(input_tensor, [[0, 0], [pad_amt_0, pad_amt_0], [pad_amt_1, pad_amt_1], [0, 0]],
                              mode='SYMMETRIC')
        h = tf.nn.convolution(input_tensor, W, strides=[ds, ds], padding='VALID', dilation_rate=[rate, rate],
                              name=layer_name + '_conv')
        h = h + bias_variable(b_shape)
        return h


# network to predict ray depths from input image

def depth_v_network(x, lfsize, disp_mult, name):
    with tf.variable_scope(name):
        b_sz = tf.shape(x)[0]
        y_sz = tf.shape(x)[1]
        x_sz = tf.shape(x)[2]
        v_sz = lfsize[2]

        cu1 = cnn_layer(x, [3, 3, 3, 16], [16], 'cu1')
        cu2 = cnn_layer(cu1, [3, 3, 16, 64], [64], 'cu2')
        cu3 = cnn_layer(cu2, [3, 3, 64, 128], [128], 'cu3')
        cu4 = cnn_layer(cu3, [3, 3, 128, 128], [128], 'cu4', rate=2)
        cu5 = cnn_layer(cu4, [3, 3, 128, 128], [128], 'cu5', rate=4)
        cu6 = cnn_layer(cu5, [3, 3, 128, 128], [128], 'cu6', rate=8)
        cu7 = cnn_layer(cu6, [3, 3, 128, 128], [128], 'cu7', rate=16)
        cu8 = cnn_layer(cu7, [3, 3, 128, 128], [128], 'cu8')
        cu9 = cnn_layer(cu8, [3, 3, 128, lfsize[2]], [lfsize[2]], 'cu9')
        cu10 = disp_mult * tf.tanh(cnn_layer_plain(cu9, [3, 3, lfsize[2], lfsize[2]], [lfsize[2]], 'cu10'))

        return tf.reshape(cu10, [b_sz, y_sz, x_sz, v_sz])

def depth_u_network(x, lfsize, disp_mult, name):
    with tf.variable_scope(name):
        b_sz = tf.shape(x)[0]
        y_sz = tf.shape(x)[1]
        x_sz = tf.shape(x)[2]
        u_sz = lfsize[3]

        cv1 = cnn_layer(x, [3, 3, 3, 16], [16], 'cv1')
        cv2 = cnn_layer(cv1, [3, 3, 16, 64], [64], 'cv2')
        cv3 = cnn_layer(cv2, [3, 3, 64, 128], [128], 'cv3')
        cv4 = cnn_layer(cv3, [3, 3, 128, 128], [128], 'cv4', rate=2)
        cv5 = cnn_layer(cv4, [3, 3, 128, 128], [128], 'cv5', rate=4)
        cv6 = cnn_layer(cv5, [3, 3, 128, 128], [128], 'cv6', rate=8)
        cv7 = cnn_layer(cv6, [3, 3, 128, 128], [128], 'cv7', rate=16)
        cv8 = cnn_layer(cv7, [3, 3, 128, 128], [128], 'cv8')
        cv9 = cnn_layer(cv8, [3, 3, 128, lfsize[3]], [lfsize[3]], 'cv9')
        cv10 = disp_mult * tf.tanh(cnn_layer_plain(cv9, [3, 3, lfsize[3], lfsize[3]], [lfsize[3]], 'cv10'))

        return tf.reshape(cv10, [b_sz, y_sz, x_sz, u_sz])


# render light field from input image and ray depths

def depth_v_rendering(central, ray_v_depths, lfsize):
    with tf.variable_scope('depth_v_rendering') as scope:
        b_sz = tf.shape(central)[0]
        y_sz = tf.shape(central)[1]
        x_sz = tf.shape(central)[2]
        v_sz = lfsize[3]

        central = tf.expand_dims(central, 3)   # b*y*x*1

        # create and reparameterize light field grid
        b_vals = tf.to_float(tf.range(b_sz))
        v_vals = tf.to_float(tf.range(v_sz)) - tf.to_float(v_sz - 1) / 2.0
        y_vals = tf.to_float(tf.range(y_sz))
        x_vals = tf.to_float(tf.range(x_sz))

        b, y, x, v = tf.meshgrid(b_vals, y_vals, x_vals, v_vals, indexing='ij')

        # warp coordinates by ray depths
        y_t = y + v * ray_v_depths
        x_t = x

        v_r = tf.zeros_like(b)

        # indices for linear interpolation
        b_1 = tf.to_int32(b)
        y_1 = tf.to_int32(tf.floor(y_t))
        y_2 = y_1 + 1
        x_1 = tf.to_int32(tf.floor(x_t))
        x_2 = x_1 + 1
        v_1 = tf.to_int32(v_r)

        y_1 = tf.clip_by_value(y_1, 0, y_sz - 1)
        y_2 = tf.clip_by_value(y_2, 0, y_sz - 1)
        x_1 = tf.clip_by_value(x_1, 0, x_sz - 1)
        x_2 = tf.clip_by_value(x_2, 0, x_sz - 1)

        # assemble interpolation indices
        interp_pts_1 = tf.stack([b_1, y_1, x_1, v_1], -1)
        interp_pts_2 = tf.stack([b_1, y_2, x_1, v_1], -1)
        interp_pts_3 = tf.stack([b_1, y_1, x_2, v_1], -1)
        interp_pts_4 = tf.stack([b_1, y_2, x_2, v_1], -1)

        # gather light fields to be interpolated
        lf_1 = tf.gather_nd(central, interp_pts_1)
        lf_2 = tf.gather_nd(central, interp_pts_2)
        lf_3 = tf.gather_nd(central, interp_pts_3)
        lf_4 = tf.gather_nd(central, interp_pts_4)

        # calculate interpolation weights
        y_1_f = tf.to_float(y_1)
        x_1_f = tf.to_float(x_1)
        d_y_1 = 1.0 - (y_t - y_1_f)
        d_y_2 = 1.0 - d_y_1
        d_x_1 = 1.0 - (x_t - x_1_f)
        d_x_2 = 1.0 - d_x_1

        w1 = d_y_1 * d_x_1
        w2 = d_y_2 * d_x_1
        w3 = d_y_1 * d_x_2
        w4 = d_y_2 * d_x_2

        lf = tf.add_n([w1 * lf_1, w2 * lf_2, w3 * lf_3, w4 * lf_4])

    return lf

def depth_u_rendering(central, ray_u_depths, lfsize):
    with tf.variable_scope('depth_u_rendering') as scope:
        b_sz = tf.shape(central)[0]
        y_sz = tf.shape(central)[1]
        x_sz = tf.shape(central)[2]
        u_sz = lfsize[2]

        central = tf.expand_dims(central, 3)   # b*y*x*1

        # create and reparameterize light field grid
        b_vals = tf.to_float(tf.range(b_sz))
        u_vals = -1 * (tf.to_float(tf.range(u_sz)) - tf.to_float(u_sz - 1) / 2.0)
        y_vals = tf.to_float(tf.range(y_sz))
        x_vals = tf.to_float(tf.range(x_sz))

        b, y, x, u = tf.meshgrid(b_vals, y_vals, x_vals, u_vals, indexing='ij')

        # warp coordinates by ray depths
        y_t = y
        x_t = x + u * ray_u_depths

        # v_r = tf.zeros_like(b)
        u_r = tf.zeros_like(b)

        # indices for linear interpolation
        b_1 = tf.to_int32(b)
        y_1 = tf.to_int32(tf.floor(y_t))
        y_2 = y_1 + 1
        x_1 = tf.to_int32(tf.floor(x_t))
        x_2 = x_1 + 1
        # v_1 = tf.to_int32(v_r)
        u_1 = tf.to_int32(u_r)

        y_1 = tf.clip_by_value(y_1, 0, y_sz - 1)
        y_2 = tf.clip_by_value(y_2, 0, y_sz - 1)
        x_1 = tf.clip_by_value(x_1, 0, x_sz - 1)
        x_2 = tf.clip_by_value(x_2, 0, x_sz - 1)

        # assemble interpolation indices
        interp_pts_1 = tf.stack([b_1, y_1, x_1, u_1], -1)
        interp_pts_2 = tf.stack([b_1, y_2, x_1, u_1], -1)
        interp_pts_3 = tf.stack([b_1, y_1, x_2, u_1], -1)
        interp_pts_4 = tf.stack([b_1, y_2, x_2, u_1], -1)

        # gather light fields to be interpolated
        lf_1 = tf.gather_nd(central, interp_pts_1)
        lf_2 = tf.gather_nd(central, interp_pts_2)
        lf_3 = tf.gather_nd(central, interp_pts_3)
        lf_4 = tf.gather_nd(central, interp_pts_4)

        # calculate interpolation weights
        y_1_f = tf.to_float(y_1)
        x_1_f = tf.to_float(x_1)
        d_y_1 = 1.0 - (y_t - y_1_f)
        d_y_2 = 1.0 - d_y_1
        d_x_1 = 1.0 - (x_t - x_1_f)
        d_x_2 = 1.0 - d_x_1

        w1 = d_y_1 * d_x_1
        w2 = d_y_2 * d_x_1
        w3 = d_y_1 * d_x_2
        w4 = d_y_2 * d_x_2

        lf = tf.add_n([w1 * lf_1, w2 * lf_2, w3 * lf_3, w4 * lf_4])

    return lf

# Warping layer in multi-view attention

def salmap_u_rendering(salmap_u_lens, ray_u_depths, lfsize):
    with tf.variable_scope('salmap_u_rendering') as scope:
        b_sz = tf.shape(salmap_u_lens)[0]
        y_sz = tf.shape(salmap_u_lens)[1]
        x_sz = tf.shape(salmap_u_lens)[2]
        u_sz = lfsize[2]

        # create and reparameterize light field grid
        b_vals = tf.to_float(tf.range(b_sz))
        u_vals = -1 * (tf.to_float(tf.range(u_sz)) - tf.to_float(u_sz - 1) / 2.0)
        y_vals = tf.to_float(tf.range(y_sz))
        x_vals = tf.to_float(tf.range(x_sz))

        b, y, x, u = tf.meshgrid(b_vals, y_vals, x_vals, u_vals, indexing='ij')

        # warp coordinates by ray depths
        y_t = y
        x_t = x - u * ray_u_depths
        u_r = -1 * u + tf.to_float(u_sz - 1) / 2.0

        # indices for linear interpolation
        b_1 = tf.to_int32(b)
        y_1 = tf.to_int32(tf.floor(y_t))
        y_2 = y_1 + 1
        x_1 = tf.to_int32(tf.floor(x_t))
        x_2 = x_1 + 1
        u_1 = tf.to_int32(u_r)

        y_1 = tf.clip_by_value(y_1, 0, y_sz - 1)
        y_2 = tf.clip_by_value(y_2, 0, y_sz - 1)
        x_1 = tf.clip_by_value(x_1, 0, x_sz - 1)
        x_2 = tf.clip_by_value(x_2, 0, x_sz - 1)

        # assemble interpolation indices
        interp_pts_1 = tf.stack([b_1, y_1, x_1, u_1], -1)
        interp_pts_2 = tf.stack([b_1, y_2, x_1, u_1], -1)
        interp_pts_3 = tf.stack([b_1, y_1, x_2, u_1], -1)
        interp_pts_4 = tf.stack([b_1, y_2, x_2, u_1], -1)

        # gather light fields to be interpolated
        lf_1 = tf.gather_nd(salmap_u_lens, interp_pts_1)
        lf_2 = tf.gather_nd(salmap_u_lens, interp_pts_2)
        lf_3 = tf.gather_nd(salmap_u_lens, interp_pts_3)
        lf_4 = tf.gather_nd(salmap_u_lens, interp_pts_4)

        # calculate interpolation weights
        y_1_f = tf.to_float(y_1)
        x_1_f = tf.to_float(x_1)
        d_y_1 = 1.0 - (y_t - y_1_f)
        d_y_2 = 1.0 - d_y_1
        d_x_1 = 1.0 - (x_t - x_1_f)
        d_x_2 = 1.0 - d_x_1

        w1 = d_y_1 * d_x_1
        w2 = d_y_2 * d_x_1
        w3 = d_y_1 * d_x_2
        w4 = d_y_2 * d_x_2

        lf = tf.add_n([w1 * lf_1, w2 * lf_2, w3 * lf_3, w4 * lf_4])

    return lf

def salmap_v_rendering(salmap_v_lens, ray_v_depths, lfsize):
    with tf.variable_scope('depth_v_rendering') as scope:
        b_sz = tf.shape(salmap_v_lens)[0]
        y_sz = tf.shape(salmap_v_lens)[1]
        x_sz = tf.shape(salmap_v_lens)[2]
        v_sz = lfsize[3]

        # create and reparameterize light field grid
        b_vals = tf.to_float(tf.range(b_sz))
        v_vals = tf.to_float(tf.range(v_sz)) - tf.to_float(v_sz - 1) / 2.0
        y_vals = tf.to_float(tf.range(y_sz))
        x_vals = tf.to_float(tf.range(x_sz))

        b, y, x, v = tf.meshgrid(b_vals, y_vals, x_vals, v_vals, indexing='ij')

        # warp coordinates by ray depths
        y_t = y - v * ray_v_depths
        x_t = x
        v_r = v + tf.to_float(v_sz - 1) / 2.0

        # indices for linear interpolation
        b_1 = tf.to_int32(b)
        y_1 = tf.to_int32(tf.floor(y_t))
        y_2 = y_1 + 1
        x_1 = tf.to_int32(tf.floor(x_t))
        x_2 = x_1 + 1
        v_1 = tf.to_int32(v_r)

        y_1 = tf.clip_by_value(y_1, 0, y_sz - 1)
        y_2 = tf.clip_by_value(y_2, 0, y_sz - 1)
        x_1 = tf.clip_by_value(x_1, 0, x_sz - 1)
        x_2 = tf.clip_by_value(x_2, 0, x_sz - 1)

        # assemble interpolation indices
        interp_pts_1 = tf.stack([b_1, y_1, x_1, v_1], -1)
        interp_pts_2 = tf.stack([b_1, y_2, x_1, v_1], -1)
        interp_pts_3 = tf.stack([b_1, y_1, x_2, v_1], -1)
        interp_pts_4 = tf.stack([b_1, y_2, x_2, v_1], -1)

        # gather light fields to be interpolated
        lf_1 = tf.gather_nd(salmap_v_lens, interp_pts_1)
        lf_2 = tf.gather_nd(salmap_v_lens, interp_pts_2)
        lf_3 = tf.gather_nd(salmap_v_lens, interp_pts_3)
        lf_4 = tf.gather_nd(salmap_v_lens, interp_pts_4)

        # calculate interpolation weights
        y_1_f = tf.to_float(y_1)
        x_1_f = tf.to_float(x_1)
        d_y_1 = 1.0 - (y_t - y_1_f)
        d_y_2 = 1.0 - d_y_1
        d_x_1 = 1.0 - (x_t - x_1_f)
        d_x_2 = 1.0 - d_x_1

        w1 = d_y_1 * d_x_1
        w2 = d_y_2 * d_x_1
        w3 = d_y_1 * d_x_2
        w4 = d_y_2 * d_x_2

        lf = tf.add_n([w1 * lf_1, w2 * lf_2, w3 * lf_3, w4 * lf_4])

    return lf

# resample ray depths for depth consistency regularization

def transform_ray_u_depths(ray_depths, u_step, lfsize):
    with tf.variable_scope('transform_ray_u_depths') as scope:
        b_sz = tf.shape(ray_depths)[0]
        y_sz = tf.shape(ray_depths)[1]
        x_sz = tf.shape(ray_depths)[2]
        u_sz = lfsize[2]

        # create and reparameterize light field grid
        b_vals = tf.to_float(tf.range(b_sz))
        u_vals = -1 * (tf.to_float(tf.range(u_sz)) - tf.to_float(u_sz - 1) / 2.0)
        y_vals = tf.to_float(tf.range(y_sz))
        x_vals = tf.to_float(tf.range(x_sz))

        b, y, x, u = tf.meshgrid(b_vals, y_vals, x_vals, u_vals, indexing='ij')

        # warp coordinates by ray depths
        y_t = y
        x_t = x + u_step * ray_depths
        u_t = u - u_step + tf.to_float(u_sz - 1) / 2.0

        # indices for linear interpolation
        b_1 = tf.to_int32(b)
        y_1 = tf.to_int32(tf.floor(y_t))
        y_2 = y_1 + 1
        x_1 = tf.to_int32(tf.floor(x_t))
        x_2 = x_1 + 1
        u_1 = tf.to_int32(u_t)

        y_1 = tf.clip_by_value(y_1, 0, y_sz - 1)
        y_2 = tf.clip_by_value(y_2, 0, y_sz - 1)
        x_1 = tf.clip_by_value(x_1, 0, x_sz - 1)
        x_2 = tf.clip_by_value(x_2, 0, x_sz - 1)
        u_1 = tf.clip_by_value(u_1, 0, u_sz - 1)

        # assemble interpolation indices
        interp_pts_1 = tf.stack([b_1, y_1, x_1, u_1], -1)
        interp_pts_2 = tf.stack([b_1, y_2, x_1, u_1], -1)
        interp_pts_3 = tf.stack([b_1, y_1, x_2, u_1], -1)
        interp_pts_4 = tf.stack([b_1, y_2, x_2, u_1], -1)

        # gather light fields to be interpolated
        lf_1 = tf.gather_nd(ray_depths, interp_pts_1)
        lf_2 = tf.gather_nd(ray_depths, interp_pts_2)
        lf_3 = tf.gather_nd(ray_depths, interp_pts_3)
        lf_4 = tf.gather_nd(ray_depths, interp_pts_4)

        # calculate interpolation weights
        y_1_f = tf.to_float(y_1)
        x_1_f = tf.to_float(x_1)
        d_y_1 = 1.0 - (y_t - y_1_f)
        d_y_2 = 1.0 - d_y_1
        d_x_1 = 1.0 - (x_t - x_1_f)
        d_x_2 = 1.0 - d_x_1

        w1 = d_y_1 * d_x_1
        w2 = d_y_2 * d_x_1
        w3 = d_y_1 * d_x_2
        w4 = d_y_2 * d_x_2

        lf = tf.add_n([w1 * lf_1, w2 * lf_2, w3 * lf_3, w4 * lf_4])

    return lf

def transform_ray_v_depths(ray_depths, v_step, lfsize):
    with tf.variable_scope('transform_ray_v_depths') as scope:
        b_sz = tf.shape(ray_depths)[0]
        y_sz = tf.shape(ray_depths)[1]
        x_sz = tf.shape(ray_depths)[2]
        v_sz = lfsize[3]

        # create and reparameterize light field grid
        b_vals = tf.to_float(tf.range(b_sz))
        v_vals = tf.to_float(tf.range(v_sz)) - tf.to_float(v_sz - 1) / 2.0
        y_vals = tf.to_float(tf.range(y_sz))
        x_vals = tf.to_float(tf.range(x_sz))

        b, y, x, v = tf.meshgrid(b_vals, y_vals, x_vals, v_vals, indexing='ij')

        # warp coordinates by ray depths
        y_t = y + v_step * ray_depths
        x_t = x

        v_t = v - v_step + tf.to_float(v_sz - 1) / 2.0

        # indices for linear interpolation
        b_1 = tf.to_int32(b)
        y_1 = tf.to_int32(tf.floor(y_t))
        y_2 = y_1 + 1
        x_1 = tf.to_int32(tf.floor(x_t))
        x_2 = x_1 + 1
        v_1 = tf.to_int32(v_t)

        y_1 = tf.clip_by_value(y_1, 0, y_sz - 1)
        y_2 = tf.clip_by_value(y_2, 0, y_sz - 1)
        x_1 = tf.clip_by_value(x_1, 0, x_sz - 1)
        x_2 = tf.clip_by_value(x_2, 0, x_sz - 1)
        v_1 = tf.clip_by_value(v_1, 0, v_sz - 1)

        # assemble interpolation indices
        interp_pts_1 = tf.stack([b_1, y_1, x_1, v_1], -1)
        interp_pts_2 = tf.stack([b_1, y_2, x_1, v_1], -1)
        interp_pts_3 = tf.stack([b_1, y_1, x_2, v_1], -1)
        interp_pts_4 = tf.stack([b_1, y_2, x_2, v_1], -1)

        # gather light fields to be interpolated
        lf_1 = tf.gather_nd(ray_depths, interp_pts_1)
        lf_2 = tf.gather_nd(ray_depths, interp_pts_2)
        lf_3 = tf.gather_nd(ray_depths, interp_pts_3)
        lf_4 = tf.gather_nd(ray_depths, interp_pts_4)

        # calculate interpolation weights
        y_1_f = tf.to_float(y_1)
        x_1_f = tf.to_float(x_1)
        d_y_1 = 1.0 - (y_t - y_1_f)
        d_y_2 = 1.0 - d_y_1
        d_x_1 = 1.0 - (x_t - x_1_f)
        d_x_2 = 1.0 - d_x_1

        w1 = d_y_1 * d_x_1
        w2 = d_y_2 * d_x_1
        w3 = d_y_1 * d_x_2
        w4 = d_y_2 * d_x_2

        lf = tf.add_n([w1 * lf_1, w2 * lf_2, w3 * lf_3, w4 * lf_4])

    return lf

# loss to encourage consistency of ray depths corresponding to same scene point

def depth_consistency_loss(d_u, d_v, lfsize):
    x_u = transform_ray_u_depths(d_u, 1.0, lfsize)
    x_v = transform_ray_v_depths(d_v, 1.0, lfsize)
    d1 = (d_u[:, :, :, 1:]-x_u[:, :, :, 1:])
    d2 = (d_v[:, :, :, 1:]-x_v[:, :, :, 1:])
    l1 = tf.reduce_mean(tf.abs(d1)+tf.abs(d2))
    return l1

# spatial TV loss (l1 of spatial derivatives)

def image_derivs(x, nc):
    dy = tf.nn.depthwise_conv2d(x, tf.tile(tf.expand_dims(tf.expand_dims([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], 2), 3), [1, 1, nc, 1]), strides=[1, 1, 1, 1], padding='VALID')
    dx = tf.nn.depthwise_conv2d(x, tf.tile(tf.expand_dims(tf.expand_dims([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], 2), 3), [1, 1, nc, 1]), strides=[1, 1, 1, 1], padding='VALID')
    return dy, dx

def tv_loss(u, v):
    b_sz = tf.shape(u)[0]
    y_sz = tf.shape(u)[1]
    x_sz = tf.shape(u)[2]
    u_sz = lfsize[2]
    v_sz = lfsize[3]
    temp1 = tf.reshape(u, [b_sz, y_sz, x_sz, u_sz])
    temp2 = tf.reshape(v, [b_sz, y_sz, x_sz, v_sz])
    dy1, dx1 = image_derivs(temp1, u_sz)
    l1 = tf.reduce_mean(tf.abs(dy1) + tf.abs(dx1))
    dy2, dx2 = image_derivs(temp2, v_sz)
    l2 = tf.reduce_mean(tf.abs(dy2) + tf.abs(dx2))
    l = l1 + l2
    return l

# cross entropy loss for saliency maps

def cross_entropy2d(input, mask):      # imput b*256*256*2

    mask = tf.reshape(mask, [-1])    # x*y*1
    mask = tf.to_float(mask)
    # mask = tf.one_hot(mask, depth=2, on_value=1, off_value=0)
    # mask = tf.tile(mask, [7, 1])   # (b*7*x*y)*2
    input = tf.reshape(input, [-1])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=input))

    return loss

# image preprocessing

def process_im(center, gt, lens_u_im, lens_v_im):

    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]

    center = tf.to_float(center)/255.0
    gt = tf.reshape(gt, [-1])
    gt = tf.reshape(tf.one_hot(tf.to_int32(gt), depth=1, on_value=0, off_value=1), [lfsize[0], lfsize[1], 1])

    center -= mean_rgb
    center /= std_rgb

    lens_u_list = tf.to_float(lens_u_im) / 255.0
    lens_v_list = tf.to_float(lens_v_im) / 255.0

    lens_u_list -= mean_rgb
    lens_u_list /= std_rgb
    lens_v_list -= mean_rgb
    lens_v_list /= std_rgb

    lens_u_list = tf.transpose(tf.reshape(lens_u_list, [lfsize[0], 7, lfsize[1], 3]), [0, 2, 1, 3])
    lens_v_list = tf.transpose(tf.transpose(tf.reshape(lens_v_list, [7, lfsize[0], lfsize[1], 3]), [1, 0, 2, 3]), [0, 2, 1, 3])

    return center, gt, lens_u_list, lens_v_list


def read_lf(filename_center_queue, filename_lens_u_queue, filename_lens_v_queue, filename_gt_queue):
    center = tf.read_file(filename_center_queue)
    lens_u = tf.read_file(filename_lens_u_queue)
    lens_v = tf.read_file(filename_lens_v_queue)
    gt = tf.read_file(filename_gt_queue)

    center_im = tf.image.decode_png(center, channels=3)   # [height, width, num_channels]
    lens_u_im = tf.image.decode_png(lens_u, channels=3)
    lens_v_im = tf.image.decode_png(lens_v, channels=3)
    gt_im = tf.image.decode_png(gt, channels=1)

    center_resize = tf.image.resize_images(center_im, [256, 256])
    gt_resize = tf.image.resize_images(gt_im, [256, 256])

    center_list, gt_list, lens_u_list, lens_v_list = process_im(center_resize, gt_resize, lens_u_im, lens_v_im)

    return center_list, lens_u_list, lens_v_list, gt_list


def input_pipeline(center_filenames, lens_u_filenames, lens_v_filenames, gt_filenames, lfsize, center_size, batchsize):
    filename_center_queue, filename_lens_u_queue, filename_lens_v_queue, filename_gt_queue = \
        tf.train.slice_input_producer([center_filenames, lens_u_filenames, lens_v_filenames, gt_filenames], shuffle=True)
    example_list = [read_lf(filename_center_queue, filename_lens_u_queue, filename_lens_v_queue, filename_gt_queue)]
    # number of threads for populating queue
    min_after_dequeue = 0
    capacity = 8
    center_batch, lens_u_batch, lens_v_batch, gt_batch = tf.train.shuffle_batch_join(example_list, batch_size=batchsize, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue, enqueue_many=False,
                                                      shapes=[[center_size[0], center_size[1], 3],
                                                              [lfsize[0], lfsize[1], lfsize[2], 3],
                                                              [lfsize[0], lfsize[1], lfsize[3], 3],
                                                              [center_size[0], center_size[1], 1]])
    return center_batch, lens_u_batch, lens_v_batch, gt_batch

# full forward model

def forward_model(center, lfsize, disp_mult):
    with tf.variable_scope('forward_model') as scope:
        # predict ray depths from input image
        ray_v_depths = depth_v_network(center, lfsize, disp_mult, 'ray_v_depths')  # b*y*x*u
        # shear input image by predicted ray depths to render Lambertian light field
        lf_shear_v_r = depth_v_rendering(center[:, :, :, 0], ray_v_depths, lfsize)
        lf_shear_v_g = depth_v_rendering(center[:, :, :, 1], ray_v_depths, lfsize)
        lf_shear_v_b = depth_v_rendering(center[:, :, :, 2], ray_v_depths, lfsize)
        lf_shear_v = tf.stack([lf_shear_v_r, lf_shear_v_g, lf_shear_v_b], axis=4)   # b*y*x*v*3

        ray_u_depths = depth_u_network(center, lfsize, disp_mult, 'ray_u_depths')  # b*y*x*v
        # shear input image by predicted ray depths to render Lambertian light field
        lf_shear_u_r = depth_u_rendering(center[:, :, :, 0], ray_u_depths, lfsize)
        lf_shear_u_g = depth_u_rendering(center[:, :, :, 1], ray_u_depths, lfsize)
        lf_shear_u_b = depth_u_rendering(center[:, :, :, 2], ray_u_depths, lfsize)
        lf_shear_u = tf.stack([lf_shear_u_r, lf_shear_u_g, lf_shear_u_b], axis=4)   # b*y*x*u*3

        lens_shear = tf.concat([lf_shear_u, lf_shear_v], axis=3)   # b*y*x*(u+v)*3

        lens_shear = tf.concat(tf.split(lens_shear, (lfsize[2] + lfsize[3]), axis=3), axis=0)   # (b*(u+v))*y*x*1*3
        lens_shear = tf.reshape(lens_shear, [(batchsize*(lfsize[2] + lfsize[3])), lfsize[0], lfsize[1], 3])   # (b*(u+v))*y*x*3
        salmap_p5, salmap_p4, salmap_p3, salmap_p2, salmap_p1 = Saliency_Prediction(tf.stop_gradient(lens_shear), 'salmaps')    # (b*(u+v))*y*x*2

        salmap_p5 = tf.expand_dims(salmap_p5, 3)   # (b*(u+v))*y*x*1*2
        salmap_p5 = tf.concat(tf.split(salmap_p5, (lfsize[2] + lfsize[3]), axis=0), axis=3)   # b*y*x*(u+v)*2
        salmap_p5_u = salmap_p5[:, :, :, 0:7, :]
        salmap_p5_v = salmap_p5[:, :, :, 7:14, :]

        salmap_p4 = tf.expand_dims(salmap_p4, 3)  # (b*(u+v-1))*y*x*1*2
        salmap_p4 = tf.concat(tf.split(salmap_p4, (lfsize[2] + lfsize[3]), axis=0), axis=3)  # b*y*x*(u+v)*2
        salmap_p4_u = salmap_p4[:, :, :, 0:7, :]
        salmap_p4_v = salmap_p4[:, :, :, 7:14, :]

        salmap_p3 = tf.expand_dims(salmap_p3, 3)  # (b*(u+v-1))*y*x*1*2
        salmap_p3 = tf.concat(tf.split(salmap_p3, (lfsize[2] + lfsize[3]), axis=0), axis=3)  # b*y*x*(u+v)*2
        salmap_p3_u = salmap_p3[:, :, :, 0:7, :]
        salmap_p3_v = salmap_p3[:, :, :, 7:14, :]

        salmap_p2 = tf.expand_dims(salmap_p2, 3)  # (b*(u+v-1))*y*x*1*2
        salmap_p2 = tf.concat(tf.split(salmap_p2, (lfsize[2] + lfsize[3]), axis=0), axis=3)  # b*y*x*(u+v)*2
        salmap_p2_u = salmap_p2[:, :, :, 0:7, :]
        salmap_p2_v = salmap_p2[:, :, :, 7:14, :]

        salmap_p1 = tf.expand_dims(salmap_p1, 3)  # (b*(u+v-1))*y*x*1*2
        salmap_p1 = tf.concat(tf.split(salmap_p1, (lfsize[2] + lfsize[3]), axis=0), axis=3)  # b*y*x*(u+v)*2
        salmap_p1_u = salmap_p1[:, :, :, 0:7, :]
        salmap_p1_v = salmap_p1[:, :, :, 7:14, :]

        depths_u_256 = tf.reshape(tf.concat(tf.split(ray_u_depths, batchsize, axis=0), axis=3), [lfsize[0], lfsize[1], (batchsize*lfsize[2]), ])
        depths_u_128 = tf.image.resize_images(depths_u_256, [128, 128])
        depths_u_128 = tf.concat(tf.split(tf.expand_dims(depths_u_128, axis=0), batchsize, axis=3), axis=0)
        depths_u_64 = tf.image.resize_images(depths_u_256, [64, 64])
        depths_u_64 = tf.concat(tf.split(tf.expand_dims(depths_u_64, axis=0), batchsize, axis=3), axis=0)
        depths_u_32 = tf.image.resize_images(depths_u_256, [32, 32])
        depths_u_32 = tf.concat(tf.split(tf.expand_dims(depths_u_32, axis=0), batchsize, axis=3), axis=0)
        depths_u_16 = tf.image.resize_images(depths_u_256, [16, 16])
        depths_u_16 = tf.concat(tf.split(tf.expand_dims(depths_u_16, axis=0), batchsize, axis=3), axis=0)

        depths_v_256 = tf.reshape(tf.concat(tf.split(ray_v_depths, batchsize, axis=0), axis=3), [lfsize[0], lfsize[1], (batchsize*lfsize[3])])
        depths_v_128 = tf.image.resize_images(depths_v_256, [128, 128])
        depths_v_128 = tf.concat(tf.split(tf.expand_dims(depths_v_128, axis=0), batchsize, axis=3), axis=0)
        depths_v_64 = tf.image.resize_images(depths_v_256, [64, 64])
        depths_v_64 = tf.concat(tf.split(tf.expand_dims(depths_v_64, axis=0), batchsize, axis=3), axis=0)
        depths_v_32 = tf.image.resize_images(depths_v_256, [32, 32])
        depths_v_32 = tf.concat(tf.split(tf.expand_dims(depths_v_32, axis=0), batchsize, axis=3), axis=0)
        depths_v_16 = tf.image.resize_images(depths_v_256, [16, 16])
        depths_v_16 = tf.concat(tf.split(tf.expand_dims(depths_v_16, axis=0), batchsize, axis=3), axis=0)

        depth_16 = tf.concat([depths_u_16, depths_v_16], axis=3)
        depth_32 = tf.concat([depths_u_32, depths_v_32], axis=3)
        depth_64 = tf.concat([depths_u_64, depths_v_64], axis=3)
        depth_128 = tf.concat([depths_u_128, depths_v_128], axis=3)
        depth_256 = tf.concat([ray_u_depths, ray_v_depths], axis=3)

        # Warping operation in multi-view attention module

        shear_u_p5 = salmap_u_rendering(salmap_p5_u[:, :, :, :, 0], tf.stop_gradient(depths_u_16), lfsize_16)
        shear_v_p5 = salmap_v_rendering(salmap_p5_v[:, :, :, :, 0], tf.stop_gradient(depths_v_16), lfsize_16)
        p5 = tf.concat([shear_u_p5, shear_v_p5], axis=3)

        shear_u_p4 = salmap_u_rendering(salmap_p4_u[:, :, :, :, 0], tf.stop_gradient(depths_u_32), lfsize_32)
        shear_v_p4 = salmap_v_rendering(salmap_p4_v[:, :, :, :, 0], tf.stop_gradient(depths_v_32), lfsize_32)
        p4 = tf.concat([shear_u_p4, shear_v_p4], axis=3)

        shear_u_p3 = salmap_u_rendering(salmap_p3_u[:, :, :, :, 0], tf.stop_gradient(depths_u_64), lfsize_64)
        shear_v_p3 = salmap_v_rendering(salmap_p3_v[:, :, :, :, 0], tf.stop_gradient(depths_v_64), lfsize_64)
        p3 = tf.concat([shear_u_p3, shear_v_p3], axis=3)

        shear_u_p2 = salmap_u_rendering(salmap_p2_u[:, :, :, :, 0], tf.stop_gradient(depths_u_128), lfsize_128)
        shear_v_p2 = salmap_v_rendering(salmap_p2_v[:, :, :, :, 0], tf.stop_gradient(depths_v_128), lfsize_128)
        p2 = tf.concat([shear_u_p2, shear_v_p2], axis=3)

        shear_u_p1 = salmap_u_rendering(salmap_p1_u[:, :, :, :, 0], tf.stop_gradient(ray_u_depths), lfsize)
        shear_v_p1 = salmap_v_rendering(salmap_p1_v[:, :, :, :, 0], tf.stop_gradient(ray_v_depths), lfsize)
        p1 = tf.concat([shear_u_p1, shear_v_p1], axis=3)

        pre5, pre4, pre3, pre2, pre1 = multi_view_attention(p1, p2, p3, p4, p5, depth_16, depth_32, depth_64, depth_128, depth_256, 'pre')

        return ray_u_depths, ray_v_depths, lf_shear_u, lf_shear_v, salmap_p1_u, salmap_p1_v, salmap_p2_u, salmap_p2_v, salmap_p3_u, salmap_p3_v, salmap_p4_u, salmap_p4_v, \
    salmap_p5_u, salmap_p5_v, shear_u_p5, shear_v_p5, shear_u_p4, shear_v_p4, shear_u_p3, shear_v_p3, shear_u_p2, shear_v_p2, shear_u_p1, shear_v_p1, pre5, pre4, pre3, pre2, pre1

if is_training:

    train_center_path = 'H:\LF_Lens\LF_Lens_256(7)\Train\center'  # path to training examples
    train_lens_u_path = 'H:\LF_Lens\LF_Lens_256(7)\Train\Lens_L_Concat'  # path to training examples
    train_lens_v_path = 'H:\LF_Lens\LF_Lens_256(7)\Train\Lens_V_Concat'  # path to training examples
    train_gt_path = 'H:\LF_Lens\LF_Lens_256(7)\Train\mask'  # path to training examples

    train_center_filenames = [os.path.join(train_center_path, f) for f in os.listdir(train_center_path) if
                              f.endswith('jpg')]
    train_lens_u_filenames = [os.path.join(train_lens_u_path, f) for f in os.listdir(train_lens_u_path) if
                              f.endswith('png')]
    train_lens_v_filenames = [os.path.join(train_lens_v_path, f) for f in os.listdir(train_lens_v_path) if
                              f.endswith('png')]
    train_gt_filenames = [os.path.join(train_gt_path, f) for f in os.listdir(train_gt_path) if f.endswith('png')]

    center_batch, lens_u_batch, lens_v_batch, gt_batch = input_pipeline(train_center_filenames,
                                                                                     train_lens_u_filenames,
                                                                                     train_lens_v_filenames,
                                                                                     train_gt_filenames, lfsize,
                                                                                     center_size, batchsize)

    # forward model
    ray_u_depths, ray_v_depths, lf_shear_u, lf_shear_v, salmap_p1_u, salmap_p1_v, salmap_p2_u, salmap_p2_v, salmap_p3_u, salmap_p3_v, salmap_p4_u, salmap_p4_v, \
    salmap_p5_u, salmap_p5_v, shear_u_p5, shear_v_p5, shear_u_p4, shear_v_p4, shear_u_p3, shear_v_p3, shear_u_p2, shear_v_p2, shear_u_p1, shear_v_p1, pre5, pre4, pre3, pre2, pre1 \
        = forward_model(center_batch, lfsize, disp_mult)

    gt_128 = tf.to_int32(tf.image.resize_images(gt_batch, [128, 128]))
    gt_64 = tf.to_int32(tf.image.resize_images(gt_batch, [64, 64]))
    gt_32 = tf.to_int32(tf.image.resize_images(gt_batch, [32, 32]))
    gt_16 = tf.to_int32(tf.image.resize_images(gt_batch, [16, 16]))

    # training losses to minimize
    lam_tv = 0.001
    lam_dc = 0.01
    with tf.name_scope('loss'):
        salmapcenter_p1Loss = cross_entropy2d(pre1, gt_batch)
        salmapLoss = cross_entropy2d(pre1, gt_batch) + cross_entropy2d(pre2, gt_128) + cross_entropy2d(pre3, gt_64) + cross_entropy2d(pre4, gt_32) + cross_entropy2d(pre5, gt_16)
        shear_u_loss = tf.reduce_mean(tf.abs(lf_shear_u - lens_u_batch))
        shear_v_loss = tf.reduce_mean(tf.abs(lf_shear_v - lens_v_batch))
        shear_loss = (shear_v_loss + shear_u_loss) / 2
        tv_loss = lam_tv * tv_loss(ray_u_depths, ray_v_depths)
        depth_consistency_loss = lam_dc * depth_consistency_loss(ray_u_depths, ray_v_depths, lfsize)
        depth_loss = shear_loss + depth_consistency_loss + tv_loss

    # Optimizer
    with tf.name_scope('train'):
        train_step_depth = tf.train.AdamOptimizer(learning_rate=learning_rate_depth).minimize(depth_loss)
        train_step_salmap = tf.train.AdamOptimizer(learning_rate=learning_rate_salmap).minimize(salmapLoss)

    # tensorboard summaries
    tf.summary.scalar('shear_loss', shear_loss)

    tf.summary.scalar('salmapLoss', salmapLoss)
    tf.summary.scalar('salmapcenter_p1Loss', salmapcenter_p1Loss)
    tf.summary.scalar('tv_loss', tv_loss)
    tf.summary.scalar('depth_consistency_loss', depth_consistency_loss)
    tf.summary.scalar('depth_loss', depth_loss)


    tf.summary.histogram('ray_u_depths', ray_u_depths)
    tf.summary.histogram('ray_v_depths', ray_v_depths)

    tf.summary.image('input_image', center_batch)

    tf.summary.image('salamp_c', tf.nn.sigmoid(pre1))

    tf.summary.image('lf_shear_u', tf.reshape(tf.transpose(lf_shear_u, perm=[0, 1, 3, 2, 4]),
                                              [batchsize, lfsize[0], lfsize[1] * lfsize[3], 3]))

    tf.summary.image('lf_shear_v', tf.reshape(tf.transpose(lf_shear_v, perm=[0, 3, 1, 2, 4]),
                                              [batchsize, lfsize[0] * lfsize[2], lfsize[1], 3]))

    tf.summary.image('ray_u_depths', tf.reshape(tf.transpose(ray_u_depths, perm=[0, 1, 3, 2]),
                                                [batchsize, lfsize[0], lfsize[1] * lfsize[3], 1]))

    tf.summary.image('ray_v_depths', tf.reshape(tf.transpose(ray_v_depths, perm=[0, 3, 1, 2]),
                                                [batchsize, lfsize[0] * lfsize[2], lfsize[1], 1]))

    tf.summary.image('salmap_u_lens', tf.reshape(tf.transpose(tf.sigmoid(salmap_p1_u), perm=[0, 1, 3, 2, 4]),
                                                  [batchsize, lfsize[0], lfsize[1] * lfsize[3], 1]))

    tf.summary.image('salmap_v_lens', tf.reshape(tf.transpose(tf.sigmoid(salmap_p1_v), perm=[0, 3, 1, 2, 4]),
                                                  [batchsize, lfsize[0] * lfsize[2], lfsize[1], 1]))

    tf.summary.image('salmap_shear_u',
                     tf.reshape(tf.transpose(tf.nn.sigmoid(tf.expand_dims(shear_u_p1, axis=4)), perm=[0, 1, 3, 2, 4]),
                                [batchsize, lfsize[0], lfsize[1] * lfsize[3], 1]))

    tf.summary.image('salmap_shear_v',
                     tf.reshape(tf.transpose(tf.nn.sigmoid(tf.expand_dims(shear_v_p1, axis=4)), perm=[0, 3, 1, 2, 4]),
                                [batchsize, lfsize[0] * lfsize[2], lfsize[1], 1]))


    merged = tf.summary.merge_all()

    logdir = 'logs/train/'  # path to store logs
    checkpointdir = 'checkpoints/'  # path to store checkpoints

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logdir, sess.graph)
        saver = tf.train.Saver(max_to_keep=40)
        sess.run(tf.global_variables_initializer())  # initialize variables

        coord = tf.train.Coordinator()  # coordinator for input queue threads
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # start input queue threads

        for i in range(train_iters):
        # training training step
            _ = sess.run(train_step_depth)
            _ = sess.run(train_step_salmap)
        # save training summaries
            if (i + 1) % 1 == 0:  # can change the frequency of writing summaries if desired
                print('training step: ', i)
                trainsummary = sess.run(merged)
                train_writer.add_summary(trainsummary, i)
            # save checkpoint
            if (i + 1) % 1000 == 0:
                saver.save(sess, checkpointdir + 'model.ckpt', global_step=i)

    # cleanup
        train_writer.close()
        coord.request_stop()
        coord.join(threads)

else:   # Test

    def process_test_im(center):

        mean_rgb = [0.485, 0.456, 0.406]
        std_rgb = [0.229, 0.224, 0.225]

        center = tf.to_float(center) / 255.0
        center -= mean_rgb
        center /= std_rgb

        return center


    def read_test_lf(filename_center_queue):
        center = tf.read_file(filename_center_queue[0])

        center_im = tf.image.decode_png(center, channels=3)  # [height, width, num_channels]

        center_resize = tf.image.resize_images(center_im, [256, 256])

        center_list = process_test_im(center_resize)

        return center_list


    def input_test_pipeline(center_filenames, center_size, batchsize):
        filename_center_queue = tf.train.slice_input_producer([center_filenames], shuffle=False)
        example_list = [read_test_lf(filename_center_queue)]
        # number of threads for populating queue
        capacity = 8
        center_batch = tf.train.batch(example_list, batch_size=batchsize, capacity=capacity,
                                      enqueue_many=False, shapes=[[center_size[0], center_size[1], 3]])
        return center_batch

    MapRoot = './Saliency maps'
    if not os.path.exists(MapRoot):
        os.mkdir(MapRoot)

    depth_u_Root = './Test/depth_u'
    if not os.path.exists(depth_u_Root):
        os.mkdir(depth_u_Root)

    depth_v_Root = './Test/depth_v'
    if not os.path.exists(depth_v_Root):
        os.mkdir(depth_v_Root)

    salmap_u_Root = './Test/salmap_u'
    if not os.path.exists(salmap_u_Root):
        os.mkdir(salmap_u_Root)

    salmap_v_Root = './Test/salmap_v'
    if not os.path.exists(salmap_v_Root):
        os.mkdir(salmap_v_Root)

    shear_u_Root = './Test/shear_u'
    if not os.path.exists(shear_u_Root):
        os.mkdir(shear_u_Root)

    shear_v_Root = './Test/shear_v'
    if not os.path.exists(shear_v_Root):
        os.mkdir(shear_v_Root)

    shear_u_lf_Root = './Test/shear_u_lf'
    if not os.path.exists(shear_u_lf_Root):
        os.mkdir(shear_u_lf_Root)

    shear_v_lf_Root = './Test/shear_v_lf'
    if not os.path.exists(shear_v_lf_Root):
        os.mkdir(shear_v_lf_Root)

    test_center_path = 'H:\LF_Lens\LF_Lens_256(7)\Test\center'  # path to training examples

    test_center_filenames = [os.path.join(test_center_path, f) for f in os.listdir(test_center_path) if
                             f.endswith('jpg')]

    center_test = input_test_pipeline(test_center_filenames, center_size, batchsize)

    center_im = tf.placeholder(tf.float32, shape=[None, center_size[0], center_size[1], 3])

    ray_u_depths, ray_v_depths, lf_shear_u, lf_shear_v, salmap_p1_u, salmap_p1_v, salmap_p2_u, salmap_p2_v, salmap_p3_u, salmap_p3_v, salmap_p4_u, salmap_p4_v, \
    salmap_p5_u, salmap_p5_v, shear_u_p5, shear_v_p5, shear_u_p4, shear_v_p4, shear_u_p3, shear_v_p3, shear_u_p2, shear_v_p2, shear_u_p1, shear_v_p1, pre5, pre4, pre3, pre2, pre1 \
        = forward_model(center_im, lfsize, disp_mult)

    pred_salmap = tf.nn.sigmoid(pre1)

    ray_u_depths = tf.reshape(tf.transpose(ray_u_depths, perm=[0, 3, 1, 2]), [batchsize * lfsize[2], lfsize[0], lfsize[1], 1])
    ray_v_depths = tf.reshape(tf.transpose(ray_v_depths, perm=[0, 3, 1, 2]), [batchsize * lfsize[3], lfsize[0], lfsize[1], 1])
    salmap_p1_u = tf.reshape(tf.transpose(tf.expand_dims(tf.nn.softmax(salmap_p1_u, axis=4)[:, :, :, :, 1], 4), perm=[0, 3, 1, 2, 4]),
                            [batchsize * lfsize[2], lfsize[0], lfsize[1], 1])
    salmap_p1_v = tf.reshape(tf.transpose(tf.expand_dims(tf.nn.softmax(salmap_p1_v, axis=4)[:, :, :, :, 1], 4), perm=[0, 3, 1, 2, 4]),
                            [batchsize * lfsize[3], lfsize[0], lfsize[1], 1])
    shear_u_p1 = tf.reshape(tf.transpose(tf.expand_dims(tf.nn.softmax(shear_u_p1, axis=4)[:, :, :, :, 1], axis=4), perm=[0, 3, 1, 2, 4]),
                           [batchsize * lfsize[2], lfsize[0], lfsize[1], 1])
    shear_v_p1 = tf.reshape(tf.transpose(tf.expand_dims(tf.nn.softmax(shear_v_p1, axis=4)[:, :, :, :, 1], axis=4), perm=[0, 3, 1, 2, 4]),
                           [batchsize * lfsize[3], lfsize[0], lfsize[1], 1])
    lf_shear_u = tf.reshape(tf.transpose(lf_shear_u, perm=[0, 3, 1, 2, 4]),
                           [batchsize * lfsize[2], lfsize[0], lfsize[1], 3])
    lf_shear_v = tf.reshape(tf.transpose(lf_shear_v, perm=[0, 3, 1, 2, 4]),
                          [batchsize * lfsize[3], lfsize[0], lfsize[1], 3])

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, './checkpoints/model.ckpt-129999')

        coord = tf.train.Coordinator()  # coordinator for input queue threads
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # start input queue threads

        for i in range(480):
            print(i)
            filename = test_center_filenames[i]
            imgname = filename[1:4]

            c_test = sess.run(center_test)

            feed_dict = {center_im: c_test}

            depth_u, depth_v, salmap_u, salmap_v, shear_u, shear_v, shear_u_lf, shear_v_lf =\
                sess.run([ray_u_depths, ray_v_depths, salmap_p1_u, salmap_p1_v, shear_u_p1, shear_v_p1, lf_shear_u, lf_shear_v], feed_dict=feed_dict)
            salmap = sess.run(pred_salmap, feed_dict=feed_dict)
            salmap = np.reshape(salmap, [center_size[0], center_size[1]])
            salmap = np.float32(salmap)
            salmap_name = "%s/%s.png" % (MapRoot, imgname)
            salmap = imresize(salmap, [256, 256], interp='bilinear')
            plt.imsave(salmap_name, salmap, cmap='gray')


            for j in range(7):

                salmap_u_j = np.reshape(salmap_u[j, :, :, :], [center_size[0], center_size[1]])
                salmap_u_j = np.float32(salmap_u_j)
                salmap_u_j_name = "%s/%s_u_%s.png" % (salmap_u_Root, imgname, j)
                salmap_u_j = imresize(salmap_u_j, [400, 590], interp='bilinear')
                plt.imsave(salmap_u_j_name, salmap_u_j, cmap='gray')

                salmap_v_j = np.reshape(salmap_v[j, :, :, :], [center_size[0], center_size[1]])
                salmap_v_j = np.float32(salmap_v_j)
                salmap_v_j_name = "%s/%s_v_%s.png" % (salmap_v_Root, imgname, j)
                salmap_v_j = imresize(salmap_v_j, [400, 590], interp='bilinear')
                plt.imsave(salmap_v_j_name, salmap_v_j, cmap='gray')

                depth_v_j = np.reshape(depth_v[j, :, :, :], [center_size[0], center_size[1]])
                depth_v_j = np.float32(depth_v_j)
                depth_v_j_name = "%s/%s_v_%s.png" % (depth_v_Root, imgname, j)
                depth_v_j = imresize(depth_v_j, [400, 590], interp='bilinear')
                plt.imsave(depth_v_j_name, depth_v_j, cmap='gray')

                depth_u_j = np.reshape(depth_u[j, :, :, :], [center_size[0], center_size[1]])
                depth_u_j = np.float32(depth_u_j)
                depth_u_j_name = "%s/%s_u_%s.png" % (depth_u_Root, imgname, j)
                depth_u_j = imresize(depth_u_j, [400, 590], interp='bilinear')
                plt.imsave(depth_u_j_name, depth_u_j, cmap='gray')

                shear_u_j = np.reshape(shear_u[j, :, :, :], [center_size[0], center_size[1]])
                shear_u_j = np.float32(shear_u_j)
                shear_u_j_name = "%s/%s_u_%s.png" % (shear_u_Root, imgname, j)
                shear_u_j = imresize(shear_u_j, [400, 590], interp='bilinear')
                plt.imsave(shear_u_j_name, shear_u_j, cmap='gray')

                shear_v_j = np.reshape(shear_v[j, :, :, :], [center_size[0], center_size[1]])
                shear_v_j = np.float32(shear_v_j)
                shear_v_j_name = "%s/%s_v_%s.png" % (shear_v_Root, imgname, j)
                shear_v_j = imresize(shear_v_j, [400, 590], interp='bilinear')
                plt.imsave(shear_v_j_name, shear_v_j, cmap='gray')

                shear_u_lf_j = np.reshape(shear_u_lf[j, :, :, :], [center_size[0], center_size[1], 3])
                shear_u_lf_j = np.float32(shear_u_lf_j)
                shear_u_lf_j_name = "%s/%s_u_%s.png" % (shear_u_lf_Root, imgname, j)
                shear_u_lf_j = imresize(shear_u_lf_j, [256, 256], interp='bilinear')
                plt.imsave(shear_u_lf_j_name, shear_u_lf_j, format='png')

                shear_v_lf_j = np.reshape(shear_v_lf[j, :, :, :], [center_size[0], center_size[1], 3])
                shear_v_lf_j = np.float32(shear_v_lf_j)
                shear_v_lf_j_name = "%s/%s_u_%s.png" % (shear_v_lf_Root, imgname, j)
                shear_v_lf_j = imresize(shear_v_lf_j, [256, 256], interp='bilinear')
                plt.imsave(shear_v_lf_j_name, shear_v_lf_j, format='png')

        coord.request_stop()
        coord.join(threads)

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from utils import *
import ipdb as pdb

def batch_norm(x, name="batch_norm"):
	return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input, name="instance_norm"):
	with tf.variable_scope(name):
		depth = input.get_shape()[-1]
		scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
		offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
		mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
		epsilon = 1e-5
		inv = tf.rsqrt(variance + epsilon)
		normalized = (input-mean)*inv
		return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
	with tf.variable_scope(name):
		return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
							weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
							biases_initializer=None)



def conv3d(input_, output_dim,
           k_t=4, k_h=4, k_w=4, d_t=2, d_h=2, d_w=2, pad_t=1, pad_h=1, pad_w=1, stddev=0.01,
           name="conv3d",padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_t, k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_t, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        
        conv = tf.nn.bias_add(conv, biases)

        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv
        
def l1Penalty(x, scale=0.1, name="L1Penalty"):
    l1P = tf.contrib.layers.l1_regularizer(scale)
    return l1P(x)

def deconv3d(input_, output_shape,
             k_t=4, k_h=4, k_w=4, d_t=2, d_h=2, d_w=2, pad_t=1, pad_h=1, pad_w=1, stddev=0.01,
             name="deconv3d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_t, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_t, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def deconv3d_old(input_, output_shape,
        k_t=4, k_h=4, k_w=4, d_t=2, d_h=2, d_w=2, pad_t=1, pad_h=1, pad_w=1, stddev=0.01,
        name="deconv3d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_t, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                    initializer=tf.random_normal_initializer(stddev=stddev))
        #pdb.set_trace()

#        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=[output_shape[0],tt,hh,ww,output_shape[-1]],
#                                       strides=[1, d_t, d_h, d_w, 1], padding='SAME')

        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_t, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #deconv = tf.reshape(tf.nn.bias_add(deconv, biases),output_shape)

        deconv = tf.nn.bias_add(deconv, biases)

        #deconv = tf.nn.bias_add(deconv, biases)
        if with_w:
            return deconv, w, biases
        else:
            return deconv



def deconv3d_me(input_, output_dim,
			 k_t=4, k_h=4, k_w=4, d_t=2, d_h=2, d_w=2, pad_t=1, pad_h=1, pad_w=1, stddev=0.01,
			 name="deconv3d", with_w=False):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		# w = tf.get_variable('w', [k_t, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
		# 					initializer=tf.random_normal_initializer(stddev=stddev))
		
		# print 1/0
		# deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
		# 						strides=[1, d_t, d_h, d_w, 1], padding='SAME')

		deconv = tf.layers.conv3d_transpose(input_, output_dim,[k_t, k_h, k_w],[d_t,d_h,d_w],padding='valid',kernel_initializer=tf.random_normal_initializer(stddev=stddev))

		# biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		# deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

		if with_w:
			print 1/0
			# return deconv, w, biases
			return deconv
		else:
			return deconv



def linear(input_, output_size, scope=None, stddev=0.01, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def deconv2d_v0(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #pdb.set_trace()
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
	with tf.variable_scope(name):
		return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
									weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
									biases_initializer=None)

# def conv3d(input_, output_dim, ks=3, s=1, stddev = 0.02, padding='VALID', name="conv3d"):
# 	with tf.variable_scope(name) as scope:
# 	        out_filters = output_dim
# 	        kernel = _weight_variable('weights', [3, 3, 3, 3, out_filters])
# 	        conv = tf.nn.conv3d(input_, kernel, [1, 1, 1, 1, 1], padding='VALID')
# 	        biases = _bias_variable('biases', [out_filters])
# 	        bias = tf.nn.bias_add(conv, biases)
# 	        prev_layer = tf.nn.relu(bias, name=scope.name)
# 	        in_filters = out_filters

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias

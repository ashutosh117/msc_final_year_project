## CONV BLOCK for resnet
from __future__ import division,print_function
import numpy as np
import tensorflow as tf  
import matplotlib.pyplot as plt


class ConvBlock(object):
    
    #layers contains parameter for each layer in the format
    #layers = [(),(),..,()]
    #where () => (input_height/width,#input channels,#output channels,#stride,padding)
    def __init__(self,layers,block_name):
        self.block_name = block_name
        self.layers = layers
        self.block_params = []
        self.n = len(layers)
        #main branch 
        for i in range(self.n):
            layer = layers[i]
            filter_sz = [layer[0],layer[0],layer[1],layer[2]]
            print('layer : ',i,'filter sz : ',filter_sz)
            W = tf.Variable(initial_value=tf.random_normal(filter_sz,dtype = tf.float32))
            self.block_params.append(W)
            print(self.block_params)
        #shortcut branch 
        first_input_size = self.layers[0][1]
        last_output_size = self.layers[-1][2]
        self.Ws = tf.Variable(initial_value=tf.random_normal([1,1,first_input_size,last_output_size],dtype = tf.float32))
        
        
    def predict(self,X):
        output = X
        short_output = tf.nn.conv2d(output,self.Ws,strides = [1,1,1,1],padding = "VALID")
        short_output = tf.layers.batch_normalization(short_output)
        for i in range(self.n):
            W = self.block_params[i]
            layer = self.layers[i]
            stride = [1,layer[3],layer[3],1]
            output = tf.nn.conv2d(output,W,strides=stride,padding = layer[4])
            output = tf.layers.batch_normalization(output)
            print(output.shape)
            if i != (self.n-1):
                output = tf.nn.relu(output)
        output = tf.add(output,short_output)
        return output
    
    
if __name__ == '__main__':
    layers = [(1,3,12,1,'VALID'),(3,12,12,1,'SAME'),(1,12,24,1,'VALID')]
    conv_block = ConvBlock(layers,'Test block')
    
    X = np.random.random((1,224,224,3)).astype(np.float32)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        output = conv_block.predict(X)
        print('Output shape : ',output.shape)

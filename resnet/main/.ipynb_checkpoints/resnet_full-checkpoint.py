#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:37:37 2019
@author: t1
"""

from __future__ import division,print_function
import numpy as np
import tensorflow as tf

from conv_block import ConvLayer,BatchNormLayer,ConvBlock
from identity_block import IdentityBlock

def init_weights(*shape):
    return (np.random.randn(*shape)*np.sqrt(2.0/(np.prod(shape[:-1])))).astype(np.float32)

#ReLu layer to have the same interface as keras 
class ReluLayer(object):
    def forward(self,X):
        return tf.nn.relu(X)
    
    def get_params(self):
        return []
    
class MaxPoolLayer(object):
    def __init__(self,dim,stride=2):
        self.dim = dim
        self.stride = stride
        
    def forward(self,X):
        return tf.nn.max_pool(X,ksize = [1,self.dim,self.dim,1],
                              strides = [1,self.stride,self.stride,1]
                              ,padding = 'VALID')
        
    def get_params(self):
        return []
        
class AvgPoolLayer(object):
    def __init__(self,dim,stride = 7):
        self.stride = stride
        self.dim = dim
        
    def forward(self,X):
        return tf.nn.avg_pool(X,ksize = [1,self.dim,self.dim,1],
                              strides = [1,self.stride,self.stride,1],
                              padding = 'VALID')
        
    def get_params(self):
        return []

class FlattenLayer(object):
    def forward(self,X):
        return tf.layers.flatten(X)
    
    def get_params(self):
        return []
    
    

class DenseLayer(object):
    def __init__(self,mi,mo):
        self.W = tf.Variable(init_weights(mi,mo))
        self.b = tf.Variable(np.zeros(mo,dtype = np.float32))
    
    def forward(self,X):
        return tf.matmul(X,self.W)+self.b
    
    def get_params(self):
        return [self.W,self.b]
    
    def copyFromKerasLayers(self,layer):
        W,b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1,op2))
    
class TfResNet(object):
    def __init__(self):
        #1st block
        
        self.conv1 = ConvLayer(7,3,64,stride = 2,padding = 'SAME')
        self.bn1 = BatchNormLayer(64)
        self.activation1 = ReluLayer()
        self.max_pool1 = MaxPoolLayer(3,stride=2)
        
        #2nd Block 
        self.conv_block2a = ConvBlock(64,[64,64,256],stride=1)
        self.identity_block2b = IdentityBlock(256,[64,64,256])
        self.identity_block2c = IdentityBlock(256,[64,64,256])
        
        #3rd Block 
        self.conv_block3a = ConvBlock(256,[128,128,512],stride=2)
        self.identity_block3b = IdentityBlock(512,[128,128,512])
        self.identity_block3c = IdentityBlock(512,[128,128,512])
        self.identity_block3d = IdentityBlock(512,[128,128,512])
        
        #4th Block 
        self.conv_block4a = ConvBlock(512,[256,256,1024],stride = 2)
        self.identity_block4b = IdentityBlock(1024,[256,256,1024])
        self.identity_block4c = IdentityBlock(1024,[256,256,1024])
        self.identity_block4d = IdentityBlock(1024,[256,256,1024])
        self.identity_block4e = IdentityBlock(1024,[256,256,1024])
        self.identity_block4f = IdentityBlock(1024,[256,256,1024])
        
        #5th Block
        self.conv_block5a = ConvBlock(1024,[512,512,2048],stride = 2)
        self.identity_block5b = IdentityBlock(2048,[512,512,2048])
        self.identity_block5c = IdentityBlock(2048,[512,512,2048])
        
        
        #Final block
        self.avg_poolf = AvgPoolLayer(7,stride=7)
        self.flattenf =  FlattenLayer()
        self.dense_layerf = DenseLayer(2048,1000)
        
    def forward(self,X):
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.activation1.forward(FX)
        FX = self.max_pool1.forward(FX)
        
        FX = self.conv_block2a.forward(FX)
        FX = self.identity_block2b.forward(FX)
        FX = self.identity_block2c.forward(FX)
        
        FX = self.conv_block3a.forward(FX)
        FX = self.identity_block3b.forward(FX)
        FX = self.identity_block3c.forward(FX)
        FX = self.identity_block3d.forward(FX)
        
        FX = self.conv_block4a.forward(FX)
        FX = self.identity_block4b.forward(FX)
        FX = self.identity_block4c.forward(FX)
        FX = self.identity_block4d.forward(FX)
        FX = self.identity_block4e.forward(FX)
        FX = self.identity_block4f.forward(FX)
        
        FX = self.conv_block5a.forward(FX)
        FX = self.identity_block5b.forward(FX)
        FX = self.identity_block5c.forward(FX)
        
        FX = self.avg_poolf.forward(FX)
        FX = self.flattenf.forward(FX)
        FX = self.dense_layerf.forward(FX)
        
        return FX
    
    def get_params(self):
        params = []
        params += self.conv1.get_params()
        params += self.bn1.get_params()
        
        params += self.conv_block2a.get_params()
        params += self.identity_block2b.get_params()
        params += self.identity_block2c.get_params()
        
        params += self.conv_block3a.get_params()
        params += self.identity_block3b.get_params()
        params += self.identity_block3c.get_params()
        params += self.identity_block3d.get_params()
        
        params += self.conv_block4a.get_params()
        params += self.identity_block4b.get_params()
        params += self.identity_block4c.get_params()
        params += self.identity_block4d.get_params()
        params += self.identity_block4e.get_params()
        params += self.identity_block4f.get_params()
        
        params += self.conv_block5a.get_params()
        params += self.identity_block5b.get_params()
        params += self.identity_block5c.get_params()
        
        params += self.dense_layerf.get_params()
        
        return params
    
    def set_session(self,session):
        
        self.conv1.session = session
        self.bn1.session = session
        
        self.conv_block2a.set_session(session)
        self.identity_block2b.set_session(session)
        self.identity_block2c.set_session(session)
        
        self.conv_block3a.set_session(session)
        self.identity_block3b.set_session(session)
        self.identity_block3c.set_session(session)
        self.identity_block3d.set_session(session)
        
        self.conv_block4a.set_session(session)
        self.identity_block4b.set_session(session)
        self.identity_block4c.set_session(session)
        self.identity_block4d.set_session(session)
        self.identity_block4e.set_session(session)
        self.identity_block4f.set_session(session)
        
        self.conv_block5a.set_session(session)
        self.identity_block5b.set_session(session)
        self.identity_block5c.set_session(session)
        
        self.dense_layerf.session = session
        
        
    def copyFromKerasLayers(self,layers):
        self.conv1.copyFromKerasLayers(layers[1])
        self.bn1.copyFromKerasLayers(layers[2])
        
        self.conv_block2a.copyFromKerasLayers(layers[5:17])
        self.identity_block2b.copyFromKerasLayers(layers[17:27])
        self.identity_block2c.copyFromKerasLayers(layers[27:37])
        
        self.conv_block3a.copyFromKerasLayers(layers[37:49])
        self.identity_block3b.copyFromKerasLayers(layers[49:59])
        self.identity_block3c.copyFromKerasLayers(layers[59:69])
        self.identity_block3d.copyFromKerasLayers(layers[69:79])
        
        self.conv_block4a.copyFromKerasLayers(layers[79:91])
        self.identity_block4b.copyFromKerasLayers(layers[91:101])
        self.identity_block4c.copyFromKerasLayers(layers[101:111])
        self.identity_block4d.copyFromKerasLayers(layers[111:121])
        self.identity_block4e.copyFromKerasLayers(layers[121:131])
        self.identity_block4f.copyFromKerasLayers(layers[131:141])
        
        self.conv_block5a.copyFromKerasLayers(layers[141:153])
        self.identity_block5b.copyFromKerasLayers(layers[153:163])
        self.identity_block5c.copyFromKerasLayers(layers[163:173])
        
        self.dense_layerf.copyFromKerasLayers(layers[175])

        
        
        
        
        
        
        
        
        
        
        

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:15:01 2019
@author: t1
"""

from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from conv_block import ConvLayer,BatchNormLayer,ConvBlock 
from identity_block import IdentityBlock

import keras
from keras.applications.resnet50 import ResNet50

## comparing the output first few layers of 
## manual resnet with keras resnet  

class MaxPoolLayer(object):
    def __init__(self,dim):
        self.dim = dim
        
    def forward(self,X):
        return tf.nn.max_pool(X,ksize = [1,self.dim,self.dim,1],strides = [1,2,2,1]
                              ,padding='VALID')
    
    def get_params(self):
        return []
    
class ReluLayer(object):
    def forward(self,X):
        return tf.nn.relu(X)
    
    def get_params(self):
        return []


## X -> CL -> BN -> relu -> MP -> CB -> output
class PartialResNet(object):
    def __init__(self):
        
        self.conv_layer1 = ConvLayer(7,3,64,stride = 2,padding = 'SAME')
        self.batch_norm1 = BatchNormLayer(64)
        self.relu_layer1 = ReluLayer()
        self.max_pool1 = MaxPoolLayer(3)
        self.conv_block1 = ConvBlock(64,mo=[64,64,256],stride=1)
        
        self.layers = [self.conv_layer1,
                       self.batch_norm1,
                       self.relu_layer1,
                       self.max_pool1,
                       self.conv_block1 
                       ]
        
    def forward(self,X):
        FX = self.conv_layer1.forward(X)
        FX = self.batch_norm1.forward(FX)
        FX = self.relu_layer1.forward(FX)
        FX = self.max_pool1.forward(FX)
        FX = self.conv_block1.forward(FX)
        return FX
    
    def get_params(self):
        all_params = []
        all_params += self.conv_layer1.get_params()
        all_params += self.batch_norm1.get_params()
        all_params += self.conv_block1.get_params()
        return all_params
    
    def set_session(self,session):
        self.session = session
        self.conv_layer1.session = session
        self.batch_norm1.session = session
        self.conv_block1.set_session(session)
        
    def copyFromKerasLayers(self,layers):
        self.conv_layer1.copyFromKerasLayers(layers[1])
        self.batch_norm1.copyFromKerasLayers(layers[2])
        self.conv_block1.copyFromKerasLayers(layers[5:])
        
    
        
        
        
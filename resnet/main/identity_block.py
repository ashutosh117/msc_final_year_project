#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:23:03 2019
@author: t1
"""

from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 
from conv_block import ConvLayer,BatchNormLayer

## Identity Block For resnet 
## tensorflow version : 1.4.0

class IdentityBlock(object):
    # X -> CL -> BN -> relu -> CL -> BN -> relu -> CL -> BN_final
    # return relu(BN_final + X) 
    def __init__(self,mi,mo):
        #mi => input no of filters
        #mo => output filters for each layer
        self.f = tf.nn.relu
        self.session = None
        
        #define layers 
        self.conv1 = ConvLayer(1,mi,mo[0],stride = 1,padding = 'VALID')
        self.bn1 = BatchNormLayer(mo[0])
        self.conv2 = ConvLayer(3,mo[0],mo[1],stride = 1,padding = 'SAME')
        self.bn2 = BatchNormLayer(mo[1])
        self.conv3 = ConvLayer(1,mo[1],mo[2],stride = 1,padding = 'VALID')
        self.bn3 = BatchNormLayer(mo[2])
        
        self.layers = [self.conv1,self.bn1,
                       self.conv2,self.bn2,
                       self.conv3,self.bn3]
        
    def forward(self,X):
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.f(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.f(FX)
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)
        
        return self.f(X + FX)
    
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params
        
    def set_session(self,session):
        self.session = session
        self.conv1.session = session
        self.bn1.session = session
        self.conv2.session = session
        self.bn2.session = session
        self.conv3.session = session
        self.bn3.session = session
        
    def copyFromKerasLayers(self,layers):
        self.conv1.copyFromKerasLayers(layers[0])
        self.bn1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.bn2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.bn3.copyFromKerasLayers(layers[7])
        
        
        
        
        
        
        
        
        
        
        
        
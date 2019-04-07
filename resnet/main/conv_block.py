#CONVBLOCK FOR RESNET using tensorflow 1.4.0
from __future__ import division,print_function
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

def init_filters(d,mi,mo):
    return (np.random.randn(d,d,mi,mo)*np.sqrt(2.0/(d*d*mi))).astype(np.float32)

## CONVLAYER
class ConvLayer(object):
    def __init__(self,d,mi,mo,stride=1,padding='SAME'):
        #d,mi,mo => filter_size,input_channels,output_channels
        self.W = tf.Variable(init_filters(d,mi,mo))
        self.b = tf.Variable(np.zeros(mo,dtype = np.float32))
        self.stride = stride
        self.padding = padding
        
    def forward(self,X):
        X = tf.nn.conv2d(X,self.W,strides = [1,self.stride,self.stride,1],padding = self.padding)
        return X+self.b
    
    def get_params(self):
        return [self.W,self.b]
    
    def copyFromKerasLayers(self,layer):
        W,b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1,op2))
        
## BATCH NORMALIZATION LAYER       
class BatchNormLayer(object):
    def __init__(self,D):
        #D => number of channels 
        self.running_mean = tf.Variable(np.zeros(D,dtype = np.float32),trainable = False)
        self.running_var = tf.Variable(np.ones(D,dtype = np.float32),trainable = False)
        self.beta = tf.Variable(np.zeros(D,dtype = np.float32)) #offset
        self.gamma = tf.Variable(np.ones(D,dtype = np.float32)) #scale
        
    def forward(self,X):
        return tf.nn.batch_normalization(X,self.running_mean,self.running_var,self.beta,self.gamma,1e-3)
    
    def get_params(self):
        return [self.running_mean,self.running_var,self.beta,self.gamma]
    
    def copyFromKerasLayers(self,layer):
        gamma,beta,running_mean,running_var = layer.get_weights()
        op1 = self.running_mean.assign(running_mean)
        op2 = self.running_var.assign(running_var)
        op3 = self.beta.assign(beta)
        op4 = self.gamma.assign(gamma)
        self.session.run((op1,op2,op3,op4))
        
##CONV BLOCK 
class ConvBlock(object):
    #1) (Main branch) X -> CL -> BN -> relu -> CL -> BN -> relu -> CL ->BN_m
    #2) (Shortcut Branch) X -> CL -> BN_s
    #3) return relu(Add(BN_m,BN_s))
    def __init__(self,mi,mo,stride = 2):
        #mi => input number of filters
        #mo => ouput number of filters for each layer
        self.stride = stride
        self.f = tf.nn.relu
        self.session = None
        # MAIN branch 
        self.conv1 = ConvLayer(1,mi,mo[0],stride = self.stride,padding = 'VALID')
        self.bn1 = BatchNormLayer(mo[0])
        self.conv2 = ConvLayer(3,mo[0],mo[1],stride = 1,padding = 'SAME')
        self.bn2 = BatchNormLayer(mo[1])
        self.conv3 = ConvLayer(1,mo[1],mo[2],stride = 1,padding = 'VALID')
        self.bn3 = BatchNormLayer(mo[2])
        
        #SHORTCUT branch
        self.convs = ConvLayer(1,mi,mo[2],stride = self.stride,padding = 'VALID')
        self.bns = BatchNormLayer(mo[2])
        
        self.layers = [
            self.conv1,self.bn1,
            self.conv2,self.bn2,
            self.conv3,self.bn3,
            self.convs,self.bns
        ]
        
    def forward(self,X):
        #MAIN branch
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.f(FX)
        
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.f(FX)
        
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)
        
        #SHORTCUT branch
        SX = self.convs.forward(X)
        SX = self.bns.forward(SX)
        
        return self.f(FX + SX)
    
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params
    
    def copyFromKerasLayers(self,layers):
        self.conv1.copyFromKerasLayers(layers[0])
        self.bn1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.bn2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.bn3.copyFromKerasLayers(layers[8])
        self.convs.copyFromKerasLayers(layers[7])
        self.bns.copyFromKerasLayers(layers[9])
        
    def set_session(self,session):
        self.session = session
        self.conv1.session = session
        self.bn1.session = session
        self.conv2.session = session
        self.bn2.session = session
        self.conv3.session = session
        self.bn3.session = session
        self.convs.session = session
        self.bns.session = session
        
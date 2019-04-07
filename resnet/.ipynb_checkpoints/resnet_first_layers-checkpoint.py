#First few  layers of resnet
from __future__ import print_function,division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input,decode_predictions
from keras.models import Model
from keras.preprocessing import image
from conv_block import ConvLayer,BatchNormLayer,ConvBlock 

# def init_filters(d,mi,mo):
#     return (np.random.randn(d,d,mi,mo)*np.sqrt(2.0/(d*d*mo))).astype(np.float32)

# class ConvLayer(object):
#     def __init__(self,d,mi,mo,stride,padding):
#         self.W = tf.Variable(init_filters(d,mi,mo))
#         self.b = tf.Variable(np.zeros(mo,dtype = np.float32))
#         self.stride = stride
#         self.padding = padding
        
#     def forward(self,X):
#         X = tf.nn.conv2d(X,self.W,strides = [1,self.stride,self.stride,1],padding = self.padding)
#         return X + self.b
    
#     def get_params(self):
#         return [self.W,self.b]
    
#     def copyFromKerasLayers(self,layer):
#         W,b = layer.get_weights()
#         op1 = self.W.assign(W)
#         op2 = self.b.assign(b)
#         self.session.run((op1,op2))
        
        
class ReluLayer(object):
        
    def forward(self,X):
        return tf.nn.relu(X)
    
    def get_params(self):
        return []
    
class MaxPoolLayer(object):
    def __init__(self,dim):
        self.dim = dim
        
    def forward(self,X):
        return tf.nn.max_pool(
            X,
            ksize = [1,self.dim,self.dim,1],
            strides = [1,2,2,1],
            padding = 'VALID'
        )
    
    def get_params(self):
        return []

class PartialResNet(object):
    def __init__(self):
        #layers for the partial resnet
        self.layers = [
            ConvLayer(d = 7,mi=3,mo=64,stride = 2,padding='SAME'),
            BatchNormLayer(64),
            ReluLayer(),
            MaxPoolLayer(3),
            ConvBlock(mi = 64,fm_sizes = [64,64,256],stride = 1)
        ]
        
        self.input_1 = tf.placeholder(dtype = tf.float32,shape = [None,224,224,3])
        self.output1 = self.forward(self.input_1)
        
    def forward(self,X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
        
    def copyFromKerasLayers(self,layers):
        self.layers[0].copyFromKerasLayers(layers[1])
        self.layers[1].copyFromKerasLayers(layers[2])
        self.layers[4].copyFromKerasLayers(layers[5:])
        
    def predict(self,X):
        assert(self.session is not None)
        return self.session.run(
            self.output1,
            feed_dict = {self.input_1 : X}
        )
    
    def set_session(self,session):
        self.session = session
        self.layers[0].session = session
        self.layers[1].session = session
        self.layers[4].set_session(session)
        
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params
    
    
if __name__ == '__main__':
    resnet = ResNet50(weights='imagenet')
    
    partial_resnet_keras = Model(
        inputs = resnet.inputs,
        outputs = resnet.layers[16].output
    )
    
    partial_resnet = PartialResNet()
    
    #random image data
    X = np.random.random((1,224,224,3)).astype(np.float32)
    #keras output
    keras_output = partial_resnet_keras.predict(X)
    
    #get output from my partial resnet model
    
    #starting a new session messes up keras so we use the one already started by keras
    session = keras.backend.get_session()
    partial_resnet.set_session(session)
    
    init = tf.variables_initializer(partial_resnet.get_params())
    session.run(init)
    
    partial_resnet_output = partial_resnet.predict(X)
    print('My Partial resnet output shape : ',partial_resnet_output.shape)
    
    #copy weights from keras model
    partial_resnet.copyFromKerasLayers(partial_resnet_keras.layers)
    
    #compare the outputs from two models
    diff = np.abs(partial_resnet.predict(X) - keras_output).sum()
    
    if diff < 1e-10:
        print('Everything is great!')
    else:
        print('diff = {:.10f}'.format(diff))
    
        
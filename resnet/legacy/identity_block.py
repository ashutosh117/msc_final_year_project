#resnet Identity block in tensorflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#filter initialization
def init_filter(d,mi,mo):
    return (np.random.randn(d,d,mi,mo)*np.sqrt(2.0/d*d*mi)).astype(np.float32)

class ConvLayer(object):
    def __init__(self,d,mi,mo,stride=1,padding = 'VALID'):
        #d,mi,mo => kernel size,input channel size,output channel size
        
        #filter and bias
        self.W = tf.Variable(init_filter(d,mi,mo),dtype = tf.float32)
        self.b = tf.Variable(np.zeros(mo,dtype =  np.float32))
        self.stride = stride
        self.padding = padding
        
    def forward(self,X):
        output = tf.nn.conv2d(
            X,
            self.W,
            strides = [1,self.stride,self.stride,1],
            padding = self.padding
        )
        return output + self.b
    
    def copyFromKerasLayers(self,layer):
        W,b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1,op2))
        
    def get_params(self):
        return [self.W,self.b]
    
#Batch Normalization class tensorflow
class BatchNormLayer(object):
    def __init__(self,D):
        #running mean and variance
        self.running_mean = tf.Variable(np.zeros(D,dtype = np.float32),trainable = False)
        self.running_var = tf.Variable(np.ones(D,dtype = np.float32),trainable = False)
        #offset and scale of new distribution
        self.beta = tf.Variable(np.zeros(D,dtype = np.float32))
        self.gamma = tf.Variable(np.ones(D,dtype = np.float32))
        
    def forward(self,X):
        #forward propagation 
        return tf.nn.batch_normalization(
            X,
            self.running_mean,
            self.running_var,
            self.beta,
            self.gamma,
            1e-3
        )
    
    def copyFromKerasLayers(self,layer):
        gamma,beta,running_mean,running_var = layer.get_weights()
        op1 = self.running_mean.assign(running_mean)
        op2 = self.running_var.assign(running_var)
        op3 = self.gamma.assign(gamma)
        op4 = self.beta.assign(beta)
        self.session.run((op1,op2,op3,op4))
        
    def get_params(self):
        return [self.running_mean,self.running_var,self.beta,self.gamma]
    

class IdentityBlock(object):
    def __init__(self,mi,fm_sizes):
        #mi => number of input channe; size
        #fm_sizes => list of  output channel sizes
        #sride = stride for the first layer of the block
        self.f = tf.nn.relu
        self.session = None
        assert(len(fm_sizes) == 3)
        
        #main branch layers
        self.conv1 = ConvLayer(1,mi,fm_sizes[0],stride = 1,padding = 'VALID')
        self.bn1 = BatchNormLayer(fm_sizes[0])
        self.conv2 = ConvLayer(3,fm_sizes[0],fm_sizes[1],stride=1,padding='SAME')
        self.bn2 = BatchNormLayer(fm_sizes[1])
        self.conv3 = ConvLayer(1,fm_sizes[1],fm_sizes[2],stride=1,padding='VALID')
        self.bn3 = BatchNormLayer(fm_sizes[2])
        
        self.layers = [
            self.conv1,self.bn1,
            self.conv2,self.bn2,
            self.conv3,self.bn3
        ]
        
        #for testing the shape of output
        #this will not be used when input is recieved from previous layers
        self.input_ = tf.placeholder(dtype = tf.float32,shape = [1,224,224,mi])
        self.output = self.forward(self.input_)
        
    def forward(self,X):
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.f(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.f(FX)
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)
        
        return self.f(FX+X)
        
    def predict(self,X):
        assert(self.session is not None)
        return self.session.run(self.output,feed_dict = {self.input_ : X})
        
    def set_session(self,session):
        self.session = session
        self.conv1.session = session
        self.bn1.session = session
        self.conv2.session = session
        self.bn2.session = session
        self.conv3.session = session
        self.bn3.session = session
        
    def copyFromKerasLayer(self,layers):
        self.conv1.copyFromKerasLayers(layers[0])
        self.bn1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.bn2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.bn3.copyFromKerasLayers(layers[7])
        
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params
    
if __name__ == '__main__':
    identity_block = IdentityBlock(256,[64,64,256])
    #test image randomly generated
    X = np.random.random((1,224,224,256)).astype(np.float32)
    
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        print('Global variables initialized')
        identity_block.set_session(session)
        output = identity_block.predict(X)
        print(output.shape)
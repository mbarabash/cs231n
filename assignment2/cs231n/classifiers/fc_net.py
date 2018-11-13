from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        first, cache = affine_forward(X, self.params['W1'], self.params['b1'])
        relu, cacherelu = relu_forward(first)
        scores, cache2 =  affine_forward(relu, self.params['W2'], self.params['b2'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, softmax = softmax_loss(scores, y)
        loss += (np.sum(self.params['W2']**2)+np.sum(self.params['W1']**2))*self.reg*.5
        dx_2, dw_2, db_2 = affine_backward(softmax, cache2)
        drelu = dx_2
        daffine2 = relu_backward(drelu, cacherelu)
        dx_1, dw_1, db_1 = affine_backward(daffine2, cache)
        grads['W2'] = dw_2+self.params['W2']*self.reg
        grads['b2'] = db_2
        grads['W1'] = dw_1+self.params['W1']*self.reg
        grads['b1'] = db_1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        for i,dim in enumerate(hidden_dims):
            if i==0:
                self.params['W'+str(i+1)] = weight_scale * np.random.randn(input_dim,dim)
            else: 
                #hidden dimensions
                self.params['W'+str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1],dim)
            
            self.params['b' +str(i+1)] = np.zeros(dim)
            if self.use_batchnorm:
                self.params['gamma'+str(i+1)]=np.ones(dim)
                self.params['beta'+str(i+1)]=np.zeros(dim)
                
        self.params['W'+str(1+len(hidden_dims))] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        self.params['b'+str(1+len(hidden_dims))] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        
        debug_print = False
        if debug_print:
            print("d: entering loss")
        
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
                
                #debug:
                #bn_param['eps']= 1e-6

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        storage={}
        #for x in self.params:
        #    print (x)
                
        for i in range(self.num_layers-1):
            
            if debug_print:
                print("d:", i )
            if i==0:
                #print(X.shape,self.params['W'+str(i+1)].shape, self.params['b'+str(i+1)].shape)
                storage['first'+str(i)], storage['cache'+str(i)] = affine_forward( X, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
                
                if debug_print:
                    print('d: storage[', 'first'+str(i), '], storage['
                      'cache'+str(i), '] = af_fwd( X, self.params[',
                      'W'+str(i+1), '], self.params[', 'b'+str(i+1),']')
                
                
            #other than first itteration
            else:
                #print(self.params['W'+str(i+1)].shape, self.params['b'+str(i+1)].shape)
                
                storage['first'+str(i)], storage['cache'+str(i)] = affine_forward(storage['relu'+str(i-1)], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
                
                if debug_print:
                    print ('d: storage[','first'+str(i),'], storage[',
                       'cache'+str(i),'] = af_fwd(storage[',
                       'relu'+str(i-1), '], self.params[','W'+str(i+1),
                       '], self.params[','b'+str(i+1),']')
            
            #batch norm true
            if self.use_batchnorm:
                
                storage['batchnorm'+str(i)], storage['batchcache'+str(i)] = batchnorm_forward(storage['first'+str(i)], self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
                #print(i)
                #relu
                storage['relu'+str(i)], storage['cacherelu'+str(i)] = relu_forward(storage['batchnorm'+str(i)])
                if debug_print:
                    print('d: storage[','batchnorm'+str(i),
                      '],  storage[', 'batchcache'+str(i),
                      '] = bnfwd(storage[', 'first'+str(i), 
                      '], self.params[', 'gamma'+str(i+1),
                      '], self.params[', 'beta'+str(i+1), ']')
                    print('d: storage[','relu'+str(i), 
                      '], storage[', 'cacherelu'+str(i),
                      '] = rlfwd(storage[', 'batchnorm'+str(i), ']')
            #batch norm false
            else:
                
                #relu
                storage['relu'+str(i)], storage['cacherelu'+str(i)] = relu_forward(storage['first'+str(i)])
                if debug_print:
                    print('d: shouldnt get here with batchnorm')
            #uno, dos = storage['cache'+str(i)][0], storage['cache'+str(i)][1]
            #print("d:",str(i), uno.shape, dos.shape, storage['cacherelu'+str(i)].shape,"\n")
        
        storage['first'+str(self.num_layers-1)], storage['cache'+str(self.num_layers-1)] = affine_forward(storage['relu'+str(self.num_layers-2)], self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])    
                
        #uno, dos = storage['cache'+str(self.num_layers-1)][0], storage['cache'+str(self.num_layers-1)][1]
        #print("d:",str(i), uno.shape, dos.shape, storage['cacherelu'+str(i)].shape,"\n")
        scores = storage['first'+str(self.num_layers-1)]
        
        
        if debug_print:
            print('d:* storage[', 'first'+str(self.num_layers-1),
              '], storage[', 'cache'+str(self.num_layers-1),
              '] = af_fwd(storage[','relu'+str(self.num_layers-2),
              '], self.params[','W'+str(self.num_layers),
              '], self.params[','b'+str(self.num_layers),']') 
            print("d: scores sum=", np.sum(scores) )

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, softmax = softmax_loss(scores, y)
        for i in range(self.num_layers):
            loss += np.sum(self.params['W'+str(i+1)]**2)*self.reg*.5
        #print("d: hello")
        for i in range(self.num_layers):
                #print("d: hi", i, self.params['W1'].shape, self.params['W2'].shape, self.params['W3'].shape, )
            if debug_print:
                print('d: bckwd i=',i)
            if i == 0: 
                dx, dw, db = affine_backward(softmax, storage['cache'+str((self.num_layers-i)-1)])
                grads['W'+(str(self.num_layers))] = dw+self.params['W'+(str(self.num_layers))]*self.reg
                grads['b'+(str(self.num_layers))] = db
                
                if debug_print:
                    print('d:* dx, dw, db = af_bwd(softmax,storage[',
                      'cache'+str((self.num_layers-i)-1),']')
                    print('d:  grads[', 'W'+(str(self.num_layers)), '] = dw+REG')
                    print("d: sum(dx), sum(dW) = ", np.sum(dx), np.sum(dw))
                
            else: 
                backrelu = relu_backward(dx, storage['cacherelu'+str(self.num_layers-i-1)])
                
                if debug_print:
                    print('d: backrelu=rlbkwd(dx, storage[', 'cacherelu'+str(self.num_layers-i-1), ']')
                    print("d: sum(backrelu)=", np.sum(backrelu))
                if self.use_batchnorm:
                    #print(self.num_layers-i-1)
                    batchback, grads['gamma'+str(self.num_layers-i)], grads['beta'+str(self.num_layers-i)] = batchnorm_backward(backrelu, storage['batchcache'+str((self.num_layers-i)-1)])
                    
                    if debug_print:
                        print('d: batchback, grads[', 'gamma'+str(self.num_layers-i),
                          '], grads[', 'beta'+str(self.num_layers-i),
                          ']=btnrmbkwd(backrelu, storage[',
                          'batchcache'+str((self.num_layers-i)-1),']')
                        print("d: sum/std/shape/[0,0](batchback) =", np.sum(batchback),np.std(batchback),batchback.shape,batchback[0,0])
                    
                    dx, dw, db = affine_backward(batchback, storage['cache'+str((self.num_layers-i)-1)])
                    
                    if debug_print:
                        print('d: dx, dw, db = af_bwd(batchback, storage[',
                          'cache'+str((self.num_layers-i)-1),']')
                    
                    
                else:   
                    dx, dw, db = affine_backward(backrelu, storage['cache'+str((self.num_layers-i)-1)])
            
                grads['W'+(str(self.num_layers-i))] = dw+self.params['W'+(str(self.num_layers-i))]*self.reg
                grads['b'+(str(self.num_layers-i))] = db
            
                if debug_print:
                    print('d: grads[','W'+(str(self.num_layers-i)), '] = dw +REG')
                    print("d: sum(dx), sum(dW) = ", np.sum(dx), np.sum(dw))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
    
#         for i in range(self.num_layers):
#             print("hi", i)
#             if i == 0:
#                 storage['dx_'+(str(self.num_layers-i-1))], storage['dw_'+(str(self.num_layers-i-1))], storage['db_'+(str(self.num_layers-i-1))] = affine_backward(softmax, storage['cache'+str(self.num_layers-i-1)])
#                 #print((str(self.num_layers-i-1)))
#             else:
#                 storage['dx_'+(str(self.num_layers-i))], storage['dw_'+(str(self.num_layers-i))], storage['db_'+(str(self.num_layers-i))] = affine_backward(storage['daffine'+(str(self.num_layers-i+1))], cache+str(self.num_layers-i-1))
            
#             #relu pass
#             storage['drelu'+(str(self.num_layers-i))] = storage['dw_'+(str(self.num_layers-i-1))]
                
#             print(storage['drelu'+(str(self.num_layers-i))].shape, (str(self.num_layers-i)))
#             print(storage['cacherelu0'].shape, storage['cacherelu1'].shape, storage['cacherelu2'].shape, )
            
#             storage['daffine'+(str(num_layers-i-1))] = relu_backward(storage['drelu'+(str(self.num_layers-i))], storage['cacherelu'+(str(self.num_layers-i-1))])
#             grads['W'+(str(self.num_layers-i+1))] = storage['dw_'+(str(self.num_layers-i))]+self.params['W'+(str(self.num_layers-i+1))]*self.reg
#             grads['b'+(str(self.num_layers-i+1))] = storage['db_'+(str(self.num_layers-i))]
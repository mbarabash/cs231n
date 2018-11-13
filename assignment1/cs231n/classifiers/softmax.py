import numpy as np
from random import shuffle
#from past.builtins import xrange
xrange=range

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for image_i in range(num_train):
    scores=X[image_i].dot(W)
    numerator = np.exp(scores[y[image_i]])
    denominator = 0
    for class_i in range(num_classes):
      denominator += np.exp(scores[class_i])
    fraction = numerator/denominator
    lfraction = np.log(fraction)
    norm_lfraction = lfraction/num_train
    loss += -norm_lfraction
    
    #backpass
    dnorm_lfraction = -1.
    dlfraction=1./num_train*dnorm_lfraction
    dfraction = 1./fraction*dlfraction
    dnumerator = 1./denominator*dfraction
    ddenominator = (-numerator/denominator**2)*dfraction
    dscores = np.zeros_like(scores)
    for class_i in range(num_classes):
      dscores[class_i] = np.exp(scores[class_i])*ddenominator
    dscores[y[image_i]] += np.exp(scores[y[image_i]]) * dnumerator
    for class_i in range(num_classes):
      dW[:,class_i] += X[image_i,:]*dscores[class_i]
  #reg
  loss += np.sum(W*W)*reg
  dW += 2*W*reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
      # regularization!                                                           #
  #############################################################################
    
  #forward pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W) #(N,C)
  exponential = np.exp(scores)
  numerator = exponential[np.arange(num_train),y] #size num_train || N
  denominator = np.sum(exponential, axis=1) #size  N
  fraction = numerator/denominator
  lfraction = np.log(fraction)
  norm_lfraction = lfraction/num_train
  loss += -np.sum(norm_lfraction)

  #backward pass
  dnorm_lfraction = -np.ones_like(norm_lfraction)
  dlfraction = 1./num_train*dnorm_lfraction
  dfraction = 1./fraction*dlfraction
  dnumerator = 1./denominator*dfraction
  ddenominator = (-numerator/denominator**2)*dfraction
  dexponential = np.ones_like(exponential)*ddenominator[:,np.newaxis]
  dexponential[np.arange(num_train),y] += dnumerator
  dscores = exponential * dexponential
  #print("dW",dW.shape,"\nX",X.shape,"\n dscores", dscores.shape)
  dW += np.tensordot(X,dscores, axes=([0],[0]))   #X (N,D)    dscores (N,C)
 
    
  loss += np.sum(W*W)*reg
  dW += 2*W*reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


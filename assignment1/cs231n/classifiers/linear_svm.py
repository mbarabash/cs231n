import numpy as np
from random import shuffle
#from past.builtins import xrange
xrange=range

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += (1./num_train)*X[i,:]
        dW[:,y[i]] += (1./num_train)*(-X[i,:])
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss *= 1./num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*W*reg
    
    
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW

def svm_loss_naive_copy(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss *= 1./num_train
  # Add regularization to the loss.
  W2=np.sum(W*W)
  loss+=reg * W2
  ##########
  ##########
  dW2=reg
  dW+=2*W*reg
  dloss2unnormalized=1./num_train
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    dcorrect_class_score = 0
    dscores = np.zeros_like(scores)
    for j in xrange(num_classes):
        if j== y[i]:
            continue
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if margin > 0:
            dmargin=dloss2unnormalized
            dscores[j]=dmargin
            dcorrect_class_score += -dmargin
    dscores[y[i]]=dcorrect_class_score
    for j in xrange(num_classes):
        dW[:,j] += X[i,:]*dscores[j]

  return loss, dW    

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train=X.shape[0]
  scores= X.dot(W) #size (N,C)
  correct_class_score = scores[np.arange(num_train),y] #size (N,)

  margin = scores - correct_class_score[:,None] + 1
 # margin = margin[margin>0]
  maskmargin = margin<0
  margin[maskmargin]=0
 # dW += (1./num_train)*X
 # dW += (1./num_train)*(-X[np.arange(num_train),y])*2
  margin[np.arange(num_train),y]=0
  loss = np.sum(margin)
  loss *= 1./num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              # 
  ############################################################################# 
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.              (D,C)  (N,D)                                           #
  #############################################################################

  dW += 2*W*reg
  dlossunnormalized = 1./num_train
  dmargin = np.ones_like(margin)*dlossunnormalized
  dmargin[np.arange(num_train),y] = 0 
  dmargin[maskmargin]=0
  dscores = 1*dmargin
  dcorrect_class_scores = -1*np.sum(dmargin, axis=1)
  dscores[np.arange(num_train),y] = dcorrect_class_scores
  dW += np.tensordot(X,dscores, axes=([0],[0]))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

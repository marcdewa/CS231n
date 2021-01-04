import numpy as np
from random import shuffle
from past.builtins import xrange

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
  for i in xrange(num_train):
    scores = X[i].dot(W)
    temp = np.zeros_like(scores)
    temp = (np.exp(scores))
    temp /= np.sum(temp)
    loss += -np.log(temp[y[i]])

    dW[:, y[i]] += (-1) * (np.sum(temp) - temp[y[i]]) / np.sum(temp) * X[i]
    for j in xrange(num_classes):
      if j == y[i]:
          continue
      dW[:, j] += temp[j] / np.sum(temp) * X[i]
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # for i in xrange(num_train):
  #   scores = X[i].dot(W)
  #   temp = np.zeros_like(scores)
  #   temp = (np.exp(scores))
  #   temp /= np.sum(temp)
  #   loss += -np.log(temp[y[i]])


  # Loss
  scores = X.dot(W)
  scores = np.exp(scores)
  sum_scores = np.sum(scores,axis=1)
  corr_scores = scores[list(range(num_train)),y]
  loss = -np.log(corr_scores/sum_scores)
  loss = np.sum(loss)

  loss /= num_train
  loss += reg * np.sum(W * W)

  # Gradient


  s = np.divide(scores, sum_scores.reshape(num_train, 1))
  s[range(num_train), y] = - (sum_scores - corr_scores) / sum_scores
  dW = X.T.dot(s)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


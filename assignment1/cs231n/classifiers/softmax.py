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
  f = X.dot(W)
  for i in range(f.shape[0]):
      fmax = np.max(f[i, :])
      sum = 0.0
      f[i, :] = f[i, :] - fmax
      for j in range(f.shape[1]):
          sum += np.exp(f[i, j])
      loss += -np.log(np.exp(f[i, y[i]])/sum)

      dW[:, y[i]] -= (sum - np.exp(f[i, y[i]]))/sum * X[i, :]
      for j in range(f.shape[1]):
          if j == y[i]:
              continue
          dW[:, j] += np.exp(f[i, j]) / sum * X[i, :]
  loss /= X.shape[0]
  dW /= X.shape[0]
  dW += 2 * reg * W
  loss += reg*np.sum(W*W)
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
  f = X.dot(W)
  fr = f - np.reshape(np.max(f, axis = 1), (f.shape[0], -1))
  sums = np.sum(np.exp(fr), axis = 1)

  losses = np.exp(fr[range(X.shape[0]), y]) / sums
  losses = -np.log(losses)
  loss = np.mean(losses) + (reg * np.sum(W*W))

  #gradient
  p_exp = np.exp(fr) / np.reshape(sums, (sums.shape[0], -1))
  p_exp[range(X.shape[0]), y] -= 1
  dW = X.T.dot(p_exp)
  dW /= X.shape[0]
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

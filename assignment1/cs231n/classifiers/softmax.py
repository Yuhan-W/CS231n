from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        dw1 = X[i]
        dw1 = dw1.reshape(dw1.shape[0], 1)
        scores = X[i].dot(W)
        robscores = scores - np.max(scores) # numeric robustness
        scores = np.exp(scores)
        dw2 = scores
        dw3 = np.zeros(num_classes) + 1 / np.sum(scores)
        correct_score = np.exp(robscores[y[i]]) / np.sum(np.exp(robscores))
        dw3[y[i]] -= 1 / dw2[y[i]]
        loss += -np.log(correct_score)
        delta = dw2 * dw3
        delta = delta.reshape(1, delta.shape[0])
        dw = dw1.dot(delta)
        dW += dw

    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    dW /= num_train
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    stabscores = scores - np.max(scores, axis=1).reshape(-1, 1)
    stabscores = np.exp(stabscores)
    sum_stabscores = np.sum(stabscores, axis=1).reshape(-1, 1)
    out = stabscores / sum_stabscores
    correct_scores = stabscores[np.arange(num_train), y].reshape(-1, 1) / sum_stabscores
    loss = np.sum(-np.log(correct_scores))
    
    loss /= num_train 
    loss += 0.5 * reg * np.sum(W*W)

    dO = out.copy()
    dO[np.arange(num_train), y] -= 1
    dW = (X.T).dot(dO)
    dW /= num_train
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

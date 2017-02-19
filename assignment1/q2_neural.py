import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    H1 = np.dot(data, W1) + b1
    A1 = sigmoid(H1)

    cost = np.sum(A1)

    # hidden = sigmoid(np.dot(data, W1) + b1)
    # scores = np.dot(hidden, W2) + b2
    # probs = softmax(scores)
    # cost = -np.sum(labels * np.log(probs)) / labels.shape[0]
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    dcost_dA1 = np.ones(shape = A1.shape)

    dA1_dH1 = sigmoid_grad(A1)

    dcost_dH1 = dcost_dA1 * dA1_dH1
    assert dcost_dH1.shape == H1.shape

    dH1_dW1 = data
    dH1_db1 = np.ones(shape = (data.shape[0], 1))

    gradW1 = np.dot(dcost_dH1.T, dH1_dW1).T
    gradb1 = np.dot(dcost_dH1.T, dH1_db1).T

    # dL_dscores = probs.copy()
    # dL_dscores -= labels
    # dL_dscores /= labels.shape[0]

    # dscores_dW2 = hidden
    # dscores_dhidden = W2

    # gradW2 = np.dot(dL_dscores.T, dscores_dW2).T
    # gradb2 = np.sum(dL_dscores, axis = 0).reshape(1, -1)

    # dL_dhidden = np.dot(dscores_dhidden, dL_dscores.T)

    # dhidden_dW1 = np.dot(sigmoid_grad(hidden), W1.T)
    # dhidden_db1 = sigmoid_grad(hidden)

    # gradW1 = np.dot(dL_dhidden, dhidden_dW1).T
    # gradb1 = np.sum(np.dot(dL_dhidden, dhidden_db1), axis = 1).reshape(1, -1)

    gradW2 = W2 * 0.
    gradb2 = b2 * 0.

    assert gradW1.shape == W1.shape, str(gradW1.shape) + " != " + str(W1.shape)
    assert gradb1.shape == b1.shape, str(gradb1.shape) + " != " + str(b1.shape)
    assert gradW2.shape == W2.shape, str(gradW2.shape) + " != " + str(W2.shape)
    assert gradb2.shape == b2.shape, str(gradb2.shape) + " != " + str(b2.shape)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 7]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    for i in range(10):
       data = np.random.randn(1, 1)
       labels = np.array([[1, 0]])
       dims = [1, 1, 2]
       params = np.random.randn((dims[0] + 1) * dims[1] + (dims[1] + 1) * dims[2])
       gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dims), params)
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()

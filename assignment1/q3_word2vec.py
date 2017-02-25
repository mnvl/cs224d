import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length

    ### YOUR CODE HERE
    x /= np.sqrt(np.sum(x ** 2, axis = 1, keepdims = True))
    ### END YOUR CODE

    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    assert (np.abs(x - np.array([[0.6, 0.8], [0.4472, 0.8944]])) < 1e-3).all()
    print

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, assuming the softmax prediction function and cross
    # entropy loss.

    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    # - dataset: needed for negative sampling, unused here.

    # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors

    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    yhat = softmax(np.dot(outputVectors, predicted))

    cost = -np.log(yhat[target])

    yhat_y = yhat.copy()
    yhat_y[target] -= 1

    gradPred = np.dot(yhat_y, outputVectors)

    grad = yhat_y[:, np.newaxis] * np.tile(predicted, (yhat_y.shape[0], 1))
    ### END YOUR CODE

    return cost, gradPred, grad

def test_softmaxCostAndGradient():
  print "Testing softmax cost and gradient..."

  D = 3
  N = 4

  predicted = np.linspace(0.2, 1, D)
  target = 2
  outputVectors = np.linspace(0.2, 1, N * D).reshape(-1, D)

  (cost, gradPred, grad) = softmaxCostAndGradient(predicted, target, outputVectors, None)

  assert np.fabs(cost - 1.28430031543) < 1e-6, cost
  assert np.amax(np.fabs(gradPred - np.array([-0.00640406] * D))) < 1e-6, gradPred
  assert np.amax(np.fabs(grad - np.array(
    [[ 0.02524334,  0.07573003,  0.12621672],
     [ 0.03738576,  0.11215727,  0.18692878],
     [-0.14463116, -0.43389347, -0.72315578],
     [ 0.08200206,  0.24600617,  0.41001028]]))) < 1e-6, grad

  print

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.
    #
    # Note: See test_word2vec below for dataset's initialization.
    #
    # Input/Output Specifications: same as softmaxCostAndGradient
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    D = predicted.shape[0]

    randomVectors = np.zeros((K, D))
    for i in range(K):
      token_index = dataset.sampleTokenIdx()
      while token_index == target:
        token_index = dataset.sampleTokenIdx()

      randomVectors[i, :] = outputVectors[token_index, :]

    cost = -np.log(sigmoid(np.dot(outputVectors[target], predicted))) - \
           np.sum(np.log(sigmoid(-np.dot(randomVectors, predicted))))

    gradPred = -(1. - sigmoid(np.dot(outputVectors[target], predicted))) * outputVectors[target] + \
               np.sum((1. - sigmoid(-np.dot(randomVectors, predicted)))[:, np.newaxis] * randomVectors, axis = 0)

    grad = 0
    ### END YOUR CODE

    return cost, gradPred, grad

def test_negSamplingCostAndGradient(dataset):
  print "Testing negative sampling cost and gradient..."

  random.seed(1)
  np.random.seed(1)

  D = 3
  N = 5

  predicted = np.linspace(0.2, 1, D)
  target = 2
  outputVectors = np.linspace(0.2, 1, N * D).reshape(-1, D)

  (cost, gradPred, grad) = negSamplingCostAndGradient(predicted, target, outputVectors, dataset)

  assert np.fabs(cost - 14.8025361347) < 1e-6, cost
  assert np.amax(np.fabs(gradPred - np.array([ 4.31425363, 4.72879439, 5.14333514]))) < 1e-6, gradPred
  assert np.amax(np.fabs(grad - np.array(
    [[ 0.37468291,  1.12404874,  1.87341457],
     [ 0.13872590,  0.41617771,  0.69362951],
     [-0.04899058, -0.14697173, -0.24495288],
     [ 0.64605456,  1.93816368,  3.23027281],
     [ 0.34041984,  1.02125953,  1.70209922]]))) < 1e-6, grad

  print

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    # Outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.
    # Input/Output specifications: same as the skip-gram model
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #
    #################################################################

    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)
    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    test_softmaxCostAndGradient()
    test_negSamplingCostAndGradient(dataset)

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()

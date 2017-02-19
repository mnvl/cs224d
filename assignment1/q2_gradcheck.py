import numpy as np
import random

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-3

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    passed = True
    while not it.finished:
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later
        ### YOUR CODE HERE:
        old = x[ix]

        x[ix] = old + h
        random.setstate(rndstate)
        fx1, grad1 = f(x)

        x[ix] = old - h
        random.setstate(rndstate)
        fx2, grad2 = f(x)

        numgrad = (fx1 - fx2) / (2. * h)

        x[ix] = old
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-6:
            print "Gradient check failed at %s of %s, grad = %f, num_grad = %f, reldiff = %f." % (
              str(ix), str(x.shape), grad[ix], numgrad, reldiff)
            passed = False

        it.iternext() # Step to next dimension

    print 'Gradient check %s.' % (passed and 'passed' or 'failed')

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    f = lambda x: (x[:,0] * 2 + x[:,1] * 3 + x[:, 2] * 5, np.array([[2, 3, 5]]))
    gradcheck_naive(f, np.random.randn(1, 3))
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()

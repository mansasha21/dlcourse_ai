import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    if predictions.ndim == 1:
        z = predictions - np.max(predictions)
        numerator = np.exp(z)
        denom = np.sum(numerator)
        return numerator/denom
    z = predictions - np.max(predictions, axis=len(predictions.shape) - 1)[:,None]
    numerator = np.exp(z)
    denom = np.sum(numerator, axis=len(predictions.shape) - 1)
    return numerator/denom[:,None]
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
   # raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    length = probs.shape[0]
    logs = np.log(probs)
    return -logs[target_index] if probs.ndim == 1 else np.mean(-logs[np.arange(length),target_index])
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    probs = softmax(predictions) 
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()
    if len(dprediction.shape) == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = probs.shape[0]
        dprediction[range(batch_size), target_index.T] -= 1
        dprediction /= batch_size
                
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W**2)
    grad = 2 * W * reg_strength
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss, grad = softmax_with_cross_entropy(predictions, target_index)
    return loss, X.T@grad

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            loss = 0

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for it,sec in enumerate(sections):
                batch_X = X[batches_indices[it]]
                batch_y = y[batches_indices[it]]
                batch_loss, batch_grad = linear_softmax(batch_X, self.W,batch_y)
                l2_loss, l2_grad = l2_regularization(self.W, reg)
                loss += l2_loss + batch_loss 
                self.W -= (batch_grad * learning_rate + l2_grad*learning_rate)
            loss /= sections.shape[0]
            #print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)
            
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        y_pred = X@self.W
        y_pred = np.argmax(y_pred, axis=1)
    
        return y_pred



                
                                                          

            

                

import numpy as np 
np.set_printoptions(suppress=True) # dont print every number in scientific form
import matplotlib.pyplot as plt

from utils import get_regression_data # function to create dummy data for regression
 
X, Y = get_regression_data() # get dummy regression data
print(X.shape)
print(Y.shape)
print(X)
print(Y)

class LinearHypothesis:
    def __init__(self): # initalize parameters 
        self.w = np.random.randn() ## randomly initialise weight
        self.b = np.random.randn() ## randomly initialise bias
        
    def __call__(self, X): # how do we calculate output from an input in our model?
        ypred = self.w * X + self.b ## make a prediction using a linear hypothesis
        return ypred # return prediction
    
    def update_params(self, new_w, new_b):
        self.w = new_w ## set this instance's weights to the new weight value passed to the function
        self.b = new_b ## do the same for the bias

H = LinearHypothesis() # instantiate our linear model
y_hat = H(X) # make prediction
print('Input:',X, '\n')
print('W:', H.w, 'B:', H.b, '\n')
print('Prediction:', y_hat, '\n')

def plot_h_vs_y(X, y_hat, Y):
    plt.figure()
    plt.scatter(X, Y, c='r', label='Label')
    plt.scatter(X, y_hat, c='b', label='Hypothesis', marker='x')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

plot_h_vs_y(X, y_hat, Y)

def L(y_hat, labels): # define our criterion (loss function)
    errors = y_hat - labels ## calculate errors
    squared_errors = errors ** 2 ## square errors
    mean_squared_error = sum(squared_errors) / len(squared_errors) ## calculate mean 
    return mean_squared_error # return loss

cost = L(y_hat, Y)
print(cost)

def random_search(n_samples, limit=20):
    """Try out n_samples of random parameter pairs and return the best ones"""
    best_weights = None ## no best weight found yet
    best_bias = None ## no best bias found yet
    lowest_cost = float('inf') ## initialize it very high (how high can it be?)
    for i in range(0, n_samples): ## try this many different parameterisations
        w = np.random.uniform(-limit, limit) ## randomly sample a weight within the limits of the search
        b = np.random.uniform(-limit, limit) ## randomly sample a bias within the limits of the search
        #print(w, b)
        H.update_params(w, b) ## update our model with these random parameters
        y_hat = H(X) ## make prediction
        cost = L(y_hat, Y) ## calculate loss
        if cost < lowest_cost: ## if this is the best parameterisation so far
            lowest_cost = cost ## update the lowest running cost to the cost for this parameterisation
            best_weights = w ## get best weights so far from the model
            best_bias = b ## get best bias so far from the model
    print('Lowest cost of', lowest_cost, 'achieved with weight of', best_weights, 'and bias of', best_bias)
    return best_weights, best_bias ## return the best weight and best bias

def grid_search(n_samples, limit=0.2):
    """Try out n_samples of random parameter pairs and return the best ones"""
    best_weights = None ## no best weight found yet
    best_bias = None ## no best bias found yet
    lowest_cost = float('inf') ## initialize it very high (how high can it be?)
    for i in range(0, n_samples):
        for j in range(0,n_samples):## try this many different parameterisations
            w = -20+2*limit*i ## randomly sample a weight within the limits of the search
            b = -20+2*limit*j ## randomly sample a bias within the limits of the search
            #print(w, b)
            H.update_params(w, b) ## update our model with these random parameters
            y_hat = H(X) ## make prediction
            cost = L(y_hat, Y) ## calculate loss
            if cost < lowest_cost: ## if this is the best parameterisation so far
                lowest_cost = cost ## update the lowest running cost to the cost for this parameterisation
                best_weights = w ## get best weights so far from the model
                best_bias = b ## get best bias so far from the model
    print('Lowest cost of', lowest_cost, 'achieved with weight of', best_weights, 'and bias of', best_bias)
    return best_weights, best_bias ## return the best weight and best bias

best_weights, best_bias = random_search(10000) # do 10000 samples in a random search 
H.update_params(best_weights, best_bias) # make sure to set our model's weights to the best values we found
plot_h_vs_y(X, H(X), Y) # plot model predictions agains labels

best_weights, best_bias = grid_search(100) # do 10000 samples in a random search 
H.update_params(best_weights, best_bias) # make sure to set our model's weights to the best values we found
plot_h_vs_y(X, H(X), Y) # plot model predictions agains labels
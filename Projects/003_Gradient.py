# DON'T WORRY ABOUT THIS CELL, IT JUST SETS SOME STUFF UP
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..') # add utils location to python path
from utils import get_regression_data, visualise_regression_data

X, Y = get_regression_data(m=30)
print(X.shape,Y.shape)
visualise_regression_data(X, Y)

# DEFINE MEAN SQUARED ERROR LOSS FUNCTION
def L(y_hat, labels):
    errors = y_hat - labels # calculate errors
    squared_errors = np.square(errors) # square errors
    mean_squared_error = np.sum(squared_errors) / (len(y_hat)) # calculate mean 
    return mean_squared_error # return loss

class LinearHypothesis:
    def __init__(self): 
        self.w = np.random.randn() ## weight
        self.b = np.random.randn() ## bias
    
    def __call__(self, X): ## how do we calculate output from an input in our model?
        y_hat = self.w*X + self.b ## make linear prediction
        return y_hat
    
    def update_params(self, new_w, new_b):
        self.w = new_w ## set this instance's w to the new w
        self.b = new_b ## set this instance's b to the new b
        
    def calc_deriv(self, X, y_hat, labels):
        m = len(Y) ## m = number of examples
        diffs = y_hat - labels ## calculate errors
        dLdw = 2*np.array(np.sum(diffs*X) / m) ## calculate derivative of loss with respect to weights
        dLdb = 2*np.array(np.sum(diffs)/m) ## calculate derivative of loss with respect to bias
        return dLdw, dLdb ## return rate of change of loss wrt w and wrt b
    
H = LinearHypothesis() ## initialise our model
y_hat = H(X) ## make prediction
dLdw, dLdb = H.calc_deriv(X, y_hat, Y) ## calculate gradient of current loss with respect to model parameters

print(dLdw, dLdb)

num_epochs = 40
learning_rate = 0.1
H = LinearHypothesis()
Gamma=.9
first_w = H.w
first_b = H.b


def plot_loss(losses):
    plt.figure() # make a figure
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(losses) # plot costs

def train(num_epochs, X, Y, H, L, plot_cost_curve=False):
    all_costs = [] ## initialise empty list of costs to plot later
    for e in range(num_epochs): ## for this many complete runs through the dataset
        y_hat = H(X) ## make predictions
        cost = L(y_hat, Y) ## compute loss 
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) ## calculate gradient of current loss with respect to model parameters
        new_w = H.w - learning_rate * dLdw ## compute new model weight using gradient descent update rule
        new_b = H.b - learning_rate * dLdb ## compute new model bias using gradient descent update rule
        H.update_params(new_w, new_b) ## update model weight and bias
        all_costs.append(cost) ## add cost for this batch of examples to the list of costs (for plotting)
        if e % (num_epochs//4) == 0:
            print(e,":",cost)
    if plot_cost_curve: ## plot stuff
       plot_loss(all_costs)
    print('Final cost:', cost)
    print('Weight values:', H.w)
    print('Bias values:', H.b)

def train_Momentum(num_epochs, X, Y, H, L, plot_cost_curve=False):
    all_costs = [] ## initialise empty list of costs to plot later
    Velocity_w=0
    Velocity_b=0
    for e in range(num_epochs): ## for this many complete runs through the dataset
        y_hat = H(X) ## make predictions
        cost = L(y_hat, Y) ## compute loss 
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) ## calculate gradient of current loss with respect to model parameters
        Velocity_w = Velocity_w * Gamma + learning_rate * dLdw 
        new_w = H.w - Velocity_w ## compute new model weight using gradient descent update rule
        Velocity_b = Velocity_b * Gamma + learning_rate * dLdb
        new_b = H.b - Velocity_b ## compute new model bias using gradient descent update rule
        H.update_params(new_w, new_b) ## update model weight and bias
        all_costs.append(cost) ## add cost for this batch of examples to the list of costs (for plotting)
        if e % (num_epochs//4) == 0:
            print(e,":",cost)
    if plot_cost_curve: ## plot stuff
       plot_loss(all_costs)
    print('Final cost:', cost)
    print('Weight values:', H.w)
    print('Bias values:', H.b)  

def train_Nesterov(num_epochs, X, Y, H, L, plot_cost_curve=False):
    all_costs = [] ## initialise empty list of costs to plot later
    Velocity_w=0
    Velocity_b=0
    for e in range(num_epochs): ## for this many complete runs through the dataset
        y_hat = H(X) ## make predictions
        cost = L(y_hat, Y) ## compute loss 
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) ## calculate gradient of current loss with respect to model parameters
        Velocity_w = Velocity_w/2 * Gamma + learning_rate * dLdw 
        new_w = H.w - Velocity_w ## compute new model weight using gradient descent update rule
        Velocity_b = Velocity_b/2 * Gamma + learning_rate * dLdb
        new_b = H.b - Velocity_b ## compute new model bias using gradient descent update rule
        H.update_params(new_w, new_b) ## update model weight and bias
        all_costs.append(cost) ## add cost for this batch of examples to the list of costs (for plotting)
        if e % (num_epochs//4) == 0:
            print(e,":",cost)
    if plot_cost_curve: ## plot stuff
       plot_loss(all_costs)
    print('Final cost:', cost)
    print('Weight values:', H.w)
    print('Bias values:', H.b)

train(num_epochs, X, Y, H, L, plot_cost_curve=True) # train model and plot cost curve
visualise_regression_data(X, Y, H(X)) # plot predictions and true data

H.w = first_w
H.b = first_b


train_Momentum(num_epochs, X, Y, H, L, plot_cost_curve=True) # train model and plot cost curve
visualise_regression_data(X, Y, H(X)) # plot predictions and true data

H.w = first_w
H.b = first_b


train_Nesterov(num_epochs, X, Y, H, L, plot_cost_curve=True) # train model and plot cost curve
visualise_regression_data(X, Y, H(X)) # plot predictions and true data
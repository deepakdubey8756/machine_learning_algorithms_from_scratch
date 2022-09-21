#implementing Linear Regresssion from scratch
class LinearReression:

    def __init__(self):
        self.theta = None
        self.bias = None

    #predicting value of x
    def predict(self, x):
        """X: one dim matrix featurs
           Theta: our weights"""

        #length of our features
        features = len(x)

        #predicted value
        y_predicted = self.bias

        #Iterating throught every data instances
        for i in range(features):
            y_predicted +=  self.theta[i] * x[i]
            
        return y_predicted


    # this function will return initial coeffients
    def initialize_coeff(self, features):
        coeffs = []
        for i in range(features):
            coeffs.append(0)
        
        return coeffs



    #Calculate cost;
    def cost(self, y, x):
        return y - self.predict(x)


    #Calculating mean squared error
    def mean_squared_error(self, y, x):
        """y: list of labels
           y_predicted: list of total predicted values"""

        #total data points
        n = len(y)

        #our total cost
        error = 0
        
        for i in range(n):
            error += self.cost(y[i], x[i])** 2
    
        return error/n
    

    def updateTheta(self, theta_i, grad_coeff_i, alpha):
        return theta_i - grad_coeff_i * alpha

    #this function will calculate gradients
    def gradient(self, alpha, x, y):
        n = len(y)
        features = len(x[0])
        gradient_coeff = self.initialize_coeff(features)
        gradient_bias = 0
        #computing gradient coefficients of entire dataset
        for i in range(n):
            diff = self.cost(y[i], x[i])
            gradient_bias += -(2/n) * diff
            for j in range(features):
                gradient_coeff[j] += -(2/n) * x[i][j] * diff

        #updating our theta parameter
        self.bias = self.updateTheta(self.bias, gradient_bias, alpha)
        for i in range(features):
            self.theta[i] = self.updateTheta(self.theta[i], gradient_coeff[i], alpha)

        return

    #fiting entire dataset
    def fit(self, x, y, alpha=0.001, epochs=10000):
        #Total Features
        m = len(x[0])
        self.theta = self.initialize_coeff(m)
        self.bias = 0
        for i in range(epochs):
            self.gradient(alpha, x, y)


if __name__ == "__main__":


    # x_multi our features
    x_multi = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]]

    # our labels
    y = [2, 4, 6, 8, 10]

    #importing our linear Regression model
    leg  = LinearReression()

    #fiting our datapoints
    leg.fit(x_multi, y)
    
    #printing it's weiths
    print(leg.theta)
    print(leg.bias)
    print(leg.predict([45, 90]))
    print(leg.mean_squared_error(y,  x_multi))

#implementing Linear Regresssion from scratch
class LinearReression:

    def __init__(self):
        self.theta = []


    #predicting value of x
    def predict(self, x):
        """X: one dim matrix featurs
           Theta: our weights"""

        #length of our features
        features = len(x)

        #predicted value
        y_predicted = self.theta[0]

        #Iterating throught every data instances
        for i in range(features):
            y_predicted +=  self.theta[i + 1] * x[i]
            
        return y_predicted



    #Calculating mean squared error
    def mean_squared_error(self, y, y_predicted):
        """y: list of labels
           y_predicted: list of total predicted values"""

        #total data points
        n = len(y)
        #our total cost
        cost = 0
        
        for i in range(n):
            cost += (y[i] - y_predicted[i]) ** 2
    
        return cost/n



    #this function will calculate gradients
    def gradient(self, alpha, x, y):
        initial_gradient  = []
        n = len(y)
        features = len(x[0])

        for i in range(features+1):
            initial_gradient.append(0)

        #computing gradient coefficients of entire dataset
        for i in range(n):
            diff = y[i] - self.predict(x[i])
            initial_gradient[0] += -(2/n) * diff
            for j in range(features):
                initial_gradient[j+1] += -(2/n) * x[i][j] * diff

        #updating our theta parameter
        self.theta[0] = self.theta[0] - initial_gradient[0] * alpha
        for i in range(1, features):
            self.theta[i] = self.theta[i] - initial_gradient[i] * alpha

        return

    #fiting entire dataset
    def fit(self, x, y):
        #Total Features
        m = len(x[0])
        for i in range(m+1):
            self.theta.append(0)
        alpha = 0.001
        epochs = 10000
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

    #Let's predict our output
    y_predicted = []
    n = len(y)
    for i in range(n):
        y_predicted.append(leg.predict(x_multi[i]))

    print(y_predicted)

    print(leg.mean_squared_error(y,  y_predicted))

import numpy as np
# Each row contains length and width of Sepal and Label
# 0: Iris-Setosa | 1: Iris-Versicolor
data_frame = np.loadtxt('iris-data.csv', delimiter=',')
#print(data_frame)

#Define Training and Testing Sets
from sklearn.model_selection import train_test_split

target_value = data_frame[:, -1]
#print (target_value)
data = data_frame[:, :-1]
#print(data)

X_training_data , X_test_data, Y_training_data, Y_test_data = train_test_split(data, target_value, test_size=0.25, random_state=42)

#print(X_testing_data)


#Training of Perceptron
from sklearn.metrics import mean_squared_error

np.random.seed(93)

class Perceptron(object):
    def __init__(self, alpha=0.01, no_of_epochs=20):
        self.alpha = alpha
        self.no_of_epochs = no_of_epochs

    def predict(self, X):
        pred = np.dot(X, self.weight_) + self.theeta_
        return 1.0 if pred >= 0.0 else 0.0
    
    def fit(self, X, Y):
        # intilizing the weights and bias
        self.weight_ = np.random.uniform(0, 1, X.shape[1])
        print(self.weight_)
        self.theeta_ = np.random.uniform(-1, -1, 1)
        print(self.theeta_)
        self.costList_ = []

        for ep in range(self.no_of_epochs):
            cost_epoch = 0
            for xi, target in zip(X, Y):
                # cost function
                predicted_value = self.predict(xi)
                cost = np.square(target - predicted_value)
                cost_epoch += float(cost/len(X))
                
                # update weights and bias
                update = self.alpha * (target - predicted_value)
                self.weight_ += update * xi
            
            # store Mean Square Error through every epoch iteration
            self.costList_.append(cost_epoch)
            
            #Results
            print("Epoch No: {:04}\tLoss: {:06.6f}".format((ep+1), cost_epoch))
            print("\t\tSummation: {:.2f}(X1) + {:.2f}(X2)  + ({:.2f})".format(self.weight_[0],self.weight_[1],float(self.theeta_)))
        return self


clf = Perceptron()
clf.fit(X_training_data, Y_training_data)

#Test
petal_length = 5.70
petal_width = 2.80

# 0: Iris-setosa | 1: Iris-versicolor
print('Iris-versicolor' if clf.predict([petal_length, petal_width]) else 'Iris-setosa')
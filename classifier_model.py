from torch import nn
#import catboost
from catboost import CatBoostClassifier, Pool

""" 
Binary Classifier Model for predicting is the reward for a blackjack bet is positive or negative based on the state of the deck
"""

class BetsClassifierCatBoost():
    def __init__(self, iterations=1000, learning_rate=0.1, depth=6, loss_function='Logloss', verbose=False):
        self.model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, depth=depth, loss_function='Logloss', verbose=False)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    

class BetsClassifierLinear(nn.Module):
    def __init__(self):
        super(BetsClassifierLinear, self).__init__()
        self.fc1 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.sigmoid(x)
        return x

class BetsClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super(BetsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.functional.sigmoid(x)
        return x
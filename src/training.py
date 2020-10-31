import numpy as np
import torch 
import torch.nn as nn
""" In this class we construct a classification multi-task FFNN that inputs a number of feature vectors and combines the output of
        all the different features into a single fully connected layer
"""

class FeedForward(torch.nn.Module):

        def __init__(self, input_sizes, hidden_size=32):
            """ All the layers and sizes are initialized. 

            Args:
                input_sizes (list): List of the feature sizes. Note that the length of ths list
                corresponds to the number of features present
                hidden_size (int, optional): Number of neurons in the hidden layer. Defaults to 32.
            """
            super().__init__()
            
            self.input_sizes = input_sizes
            self.hidden_size  = hidden_size
            self.num_features  = len(input_sizes)
            
            self.fc1 = [torch.nn.Linear(i, self.hidden_size) for i in self.input_sizes]
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            
            self.combine = torch.nn.Linear(self.num_features, self.hidden_size)
            self.final = torch.nn.Linear(self.hidden_size, 5)
            
            self.sigmoid = torch.nn.Sigmoid()    
            self.softmax = torch.nn.Softmax(dim=1)  
                       
        def forward(self, x):
            """[summary]

            Args:
                x (list(torch.Tensor)): List of tensors containing the input features

            Returns:
                (list): List of classification labels
            """
            
            hidden = [i(k) for i, k in zip(self.fc1, x)]
            relus = [self.sigmoid(h) for h in hidden]
            outputs = [self.fc2(r) for r in relus]
            total_output = torch.cat([i for i in outputs],axis=1)
            
            hidden_final = self.combine(total_output)
            sigmoid_last = self.sigmoid(hidden_final)
            output_final = self.final(sigmoid_last)
            output_final = self.softmax(output_final)
            
            return output_final
        
class linnetwork(nn.Module):
    def __init__(self, input_size, hidden_size = 32):
        super().__init__()
        torch.manual_seed(0)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lin1 = nn.Linear(input_size, hidden_size) 
        self.lin2 = nn.Linear(hidden_size, 5)
        self.softmax = torch.nn.Softmax(dim=1)  
        
    def forward(self, X):
        a1 = self.lin1(X) #computes the dot product and adds bias
        h1 = a1.sigmoid()
        a2 = self.lin2(h1) #computes dot product and adds bias
        output_final = self.softmax(a2)
        return output_final
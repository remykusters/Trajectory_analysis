import numpy as np
import torch 


class FeedForward(torch.nn.Module):
        '''
        In this class we construct a classification multi-task FFNN that inputs a number of feature vectors and combines the output of
        all the different features into a single fully connected layer
        '''
        def __init__(self, input_sizes, hidden_size=10):
            super().__init__()
            
            self.input_sizes = input_sizes
            self.hidden_size  = hidden_size
            self.num_features  = len(input_sizes)
            
            self.fc1 = [torch.nn.Linear(i, self.hidden_size) for i in self.input_sizes]
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            
            self.combine = torch.nn.Linear(self.num_features, self.hidden_size)
            self.final = torch.nn.Linear(self.hidden_size, 4)
            
            self.sigmoid = torch.nn.Sigmoid()    
            #self.softmax = torch.nn.Softmax(dim=1)  
                       
        def forward(self, x):
            
            hidden = [i(k) for i, k in zip(self.fc1, x)]
            relus = [self.sigmoid(h) for h in hidden]
            outputs = [self.fc2(r) for r in relus]
            total_output = torch.cat([i for i in outputs],axis=1)
            
            hidden_final = self.combine(total_output)
            sigmoid_last = self.sigmoid(hidden_final)
            output_final = self.final(sigmoid_last)
            output_final = self.sigmoid(output_final)
            
            return output_final
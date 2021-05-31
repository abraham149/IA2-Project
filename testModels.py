import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#needed data
class LogisticRegression() :
    def __init__( self, lr, iters ) :        
        self.lr = lr
        self.iters = iters
        self.costs_train = []
        self.costs_val = []
        self.acc_train = []
        self.acc_val = []
          
    def sigmoid(self, Z):
        return 1/(1+np.exp(Z*-1))
    
    def fit( self, X_train, y_train, X_val, y_val,  print_cost = True ) :              
        self.m, self.n = X_train.shape        
        self.w = np.zeros( self.n )        
        self.b = 0        
        self.X = X_train        
        self.y = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.print_cost = print_cost      
        self.optimize()            
        return self
      
    # Return the last metrics for the cost and accuracies
    def last_metrics(self):
        return [self.costs_train[-1], self.costs_val[-1], self.acc_train[-1], self.acc_val[-1]]
        
    def optimize( self ):           
        for i in range(self.iters):
            # Calculate grads
            A = self.sigmoid( (self.X @ self.w)+self.b)
            dA = (A-self.y.T)
            dA = np.reshape(dA,self.m)        
            dW = (self.X.T @ dA) /self.m         
            db = np.sum(dA)/self.m
            # Optimize
            self.w -= self.lr*dW    
            self.b -= self.lr*db
            # Calculate trainning metrics
            cost = (-1/self.m) * (np.sum(self.y*np.log(A) + (1-self.y)*np.log(1-A)))
            training_acc = accuracy_score(y_train, self.predict(X_train))
            # Calculate val metrics
            A_val = self.sigmoid( (self.X_val @ self.w)+self.b)
            cost_val = cost = (-1/self.m) * (np.sum(self.y_val*np.log(A_val) + (1-self.y_val)*np.log(1-A_val)))
            val_acc = accuracy_score(y_val, self.predict(X_val))
            if i % 100 == 0:
                self.costs_train.append(cost)
                self.costs_val.append(cost_val)
                self.acc_train.append(training_acc)
                self.acc_val.append(val_acc)
                
            if self.print_cost and i % 100 == 0:
                print ("Iter {}: train cost: {:.5f}, train acc: {:.5f}, val cost: {:.5f}, val acc: {:.5f} "
                       .format(i,cost,training_acc,cost_val, val_acc ) )
        return self

    #Predict function  
    def predict( self, X ) :    
        A = self.sigmoid((X @ self.w) +self.b)
        Y = np.where(A>0.5,1,0)        
        return Y

class FFNN(nn.Module):
    def __init__(self, embedding_dim, n_cont, out_sz, layers, p=0.5):
        """
        Params:
            embedding_dim: list of embedding dimensions, one item per categorical feature
            n_cont: number of continous features
            out_sz: size of output layer, usually 1 as it classification
            layers: array with different hidden layer sizes
            p: dropout betw 0-1
        """
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dim]) #Generate the embedding layers, iterate dims
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((out for inp,out in embedding_dim))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        layerlist.append(nn.Sigmoid())
            
        self.layers = nn.Sequential(*layerlist)
        
        """        
        n_emb = sum((out for inp,out in embedding_dim)) #get total size of embeddings
        n_in = n_emb + n_cont #get the total numberof features after embeddings
        
        self.fc1 = nn.Linear(n_in,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,out_sz)
        """
    
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i])) # Apply each embedd to its feature
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1) # Unite the embedding with the cont features
        
        x = self.layers(x)

        return x
"""
Calificacion_Promedio Tareas_Puntuales Total_Tareas
Dias_Conectado Minutos_Promedio Año Mes CICLO tareas_score
"""

Calificacion_Promedio = float(input("AVG grade "))
Tareas_Puntuales = int(input("Punctual HWs "))
Total_Tareas = int(input("Total HWs "))
Dias_Conectado = int(input("Days connected "))
Minutos_Promedio = float(input("Avg mins "))
Año = int(input("year "))
Mes = int(input("Month "))

print("Orgs: ",['Mktg', 'Finance', 'Admin', 'EDU', 'HR', 'IT', 'BSN', 'Law', 'AH'])
Org = int(input("Org"))

Año = min(Año - 2017,2)
Ciclo = Año + Mes/8
tareas_score = Tareas_Puntuales / Total_Tareas

model_to_choose = int(input("which model"))

Graduado = ["No","Si"]

if model_to_choose == 1: #numpy model
    model = FFNN([(3, 2), (8, 4), (9, 5)], 9, 1, layers = [100,100])
    model.load_state_dict(torch.load("pytorchFFNN.pt"))
    model.eval()
else:
    with open('LogReg_ScratchModel.m', 'rb') as input:
        model = pickle.load(input)

if model_to_choose == 1:
    row_cat = [Año, Mes, Org]
    row_cont = [Calificacion_Promedio, Tareas_Puntuales,0,0, Total_Tareas, Dias_Conectado, Minutos_Promedio, Minutos_Promedio * Dias_Conectado, Ciclo]
    cat = torch.Tensor(row_cat).reshape(1,-1)
    cat = cat.to(torch.long)
    cont = torch.Tensor(row_cont).reshape(1,-1)
    print("Graduado: ")
    print(model(cat, cont).item())
    
else:
    row = [Calificacion_Promedio, Tareas_Puntuales, Total_Tareas, Dias_Conectado, Minutos_Promedio, Año, Mes, Ciclo , tareas_score]
    row = np.array(row)
    print("Graduado: ")
    pred = model.predict(row)
    print(Graduado[pred])
    


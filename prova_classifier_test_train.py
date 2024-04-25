from __future__ import annotations
from TREEplus import *
import pandas as pd

#d = dict(features, **n_features)  #merges the two dicts
#df = pd.DataFrame(data=d)         #creates the dataframe
import time
import csv

############################ y categorical ####################################### df=pd.read_csv('Carseats.csv') df=df.iloc[:,1:]  features_names=list(df.columns)  colonne=features_names[:6] features_name=features_names[7:9] features_names=colonne + features_name     n_features_names=list(df.columns) columns = [(n_features_names[6])] n_features_name = n_features_names[9:11] n_features_names=columns + n_features_name     features=df.iloc[:,0:6] features2=df.iloc[:,7:9]  features=dict(features) features2=dict(features2)   n_features=df.iloc[:,6:7] n_features2=df.iloc[:,9:11]   n_features=dict(n_features) n_features2=dict(n_features2)  features = dict(features, features2) n_features = dict(n_features, n_features2)  High=[] for i in features['Sales']:     if i < 8:         High.append('NO')     else:         High.append('YES')  High=pd.DataFrame(High) High=dict(High) High['High'] = High.pop(0)  y=High['High']  exclude_keys = ['Sales']  new_d = {k: features[k] for k in set(list(featur
############################ y categorical #######################################
df=pd.read_csv('Carseats_train.csv')
#df=df.iloc[:,1:]

features_names=list(df.columns)

colonne=features_names[:6]
features_name=features_names[7:9]
features_names=colonne + features_name




n_features_names=list(df.columns)
columns = [(n_features_names[6])]
n_features_name = n_features_names[9:11]
n_features_names=columns + n_features_name




features=df.iloc[:,0:6]
features2=df.iloc[:,7:9]

features=dict(features)
features2=dict(features2)


n_features=df.iloc[:,6:7]
n_features2=df.iloc[:,9:11]


n_features=dict(n_features)
n_features2=dict(n_features2)

features = dict(features, **features2)
n_features = dict(n_features, **n_features2)


High=[]
for i in features['Sales']:
    if i < 8:
        High.append('NO')
    else:
        High.append('YES')

High=pd.DataFrame(High)
High=dict(High)
High['High'] = High.pop(0)

y=High['High']


#y = n_features["ShelveLoc"] testing with shelveloc as target variable 

exclude_keys = ['Sales']



new_d = {k: features[k] for k in set(list(features.keys())) - set(exclude_keys)}
features=new_d

#del features["Sales"]   #attempt to quicken removal of target variabkle


features_names=features_names[1:]

indici = np.arange(0, len(y))


###############Prepating Test Set############################################### 
df_test=pd.read_csv('Carseats_test.csv')
#df_test=df_test.iloc[:,1:]

features_test=df_test.iloc[:,0:6]
features2_test=df_test.iloc[:,7:9]

features_test=dict(features_test)
features2_test=dict(features2_test)


n_features_test=df_test.iloc[:,6:7]
n_features2_test=df_test.iloc[:,9:11]


n_features_test=dict(n_features_test)
n_features2_test=dict(n_features2_test)

features_test = dict(features_test, **features2_test)
n_features_test = dict(n_features_test, **n_features2_test)


High=[]
for i in features_test['Sales']:
    if i < 8:
        High.append('NO')
    else:
        High.append('YES')

High=pd.DataFrame(High)
High=dict(High)
High['High'] = High.pop(0)

y_test=High['High']
y_test = y_test.tolist()


#y_test = n_features_test["Shelveloc"]

del features_test["Sales"]   


###########################################################################

#when definining a funcion please be aware we are using purity gain or information gain or greatest difference between variance, all positive aspects 
#adding user_defined as a possible impurity_fn and added user_impur to carry that function 
#user_fn is only defined for the growing stage at this point 

def user_fn(self, node): #impur just takes node in CART
    #example gini
    prom = 0
    c = Counter(self.y[node.indexes]) #Creates a dictionary {"yes":number, "no"}
    c = list(c.items())
    for i in  c:
        prob_i = float((i[1]/len(self.y[node.indexes])))**2 #probability squared
                    
        prom += prob_i*i[1] #original weighted, only looking at purity
                    
    return prom   


start = time.time()
my_tree = NodeClass('n1', indici) 
tree = TREEplus(y,features,features_names,n_features,n_features_names,impurity_fn = "gini",problem="classifier",method = "CART",max_level = 10, min_cases_parent= 10,min_cases_child= 5,min_imp_gain=0.0001) 


tree.growing_tree(my_tree)
##can print single tree 


#alpha = tree.pruning(features_test, n_features_test, y_test)
#tree.print_alpha(alpha) prints the alpha values

treetab = tree.print_tree(table = True)  #print both the pruned and non pruned tree 

treetab

print(time.time() - start)
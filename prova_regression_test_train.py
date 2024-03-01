#from __future__ import annotations
#from tkinter import N

from TREEplus import *

import pandas as pd

#d = dict(features, **n_features)  #merges the two dicts
#df = pd.DataFrame(data=d)         #creates the dataframe

#print(df)
#import csv



#####################################loading the carseats data#########################

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


#############################################################################


#import itertools
#from statistics import mode
'''
print(features,'features',type(features))
print()
print(features_names,'features_names')
print()
print(n_features,'n_features')
print()
print(n_features_names,'n_features_names')
print()
'''
########################### y categorical #######################################
'''
High=[]
for i in features['Sales']:
    if i < 8:
        High.append('NO')
    else:
        High.append('YES')

High=pd.DataFrame(High)
High=dict(High)
High['High'] = High.pop(0)

#y=High['High']

#exclude_keys = ['Sales']

#new_d = {k: features[k] for k in set(list(features.keys())) - set(exclude_keys)}
#features=new_d

#features_names=features_names[1:]
'''

######################y numerical#####################################
y=features['Price']
exclude_keys = ['Price']
new_d = {k: features[k] for k in set(list(features.keys())) - set(exclude_keys)}
features=new_d

indici = np.arange(0, len(y))

features_names3 = features_names[0:5]
features_names4 = features_names[6:]
features_names = features_names3 +  features_names4 

#############Data Prep for prediction ############
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

y_test=features_test['Price']

y_test = y_test.tolist()

del features_test["Price"]   

#impurity = impurity_fn('MSE') # chhose the simplest impurity functin (for regression tree)

#impurity = Impurity ("MSE")
# start a tree structure by instantiating its root
#print("features", features)
#print("features_names", features_names)
#print("n_features", n_features)
#print("n_features_names", n_features_names)



############Program Running


###User Defined Function 



#when definining a funcion please be aware we are using purity gain or information gain or greatest difference between variance, all positive aspects 
#adding user_defined as a possible impurity_fn and added user_impur to carry that function 
def user_fn(self, node): #impur just takes node in CART
    
    return (mean(self.y[node.indexes])**2)*len(self.y[node.indexes])


my_tree = MyNodeClass('n1', indici) 

tree = TREEplus(y,features,features_names,n_features,n_features_names, impurity_fn = "pearson", problem="regression", method = "TWO-STAGE", min_cases_parent= 10,min_cases_child= 5) 

tree.growing_tree(my_tree)

#tree.print_tree()


#alpha = tree.pruning(features_test, n_features_test, y_test)

#tree.print_alpha(alpha)


tree.print_tree()

#alpha = tree.pruning()



'''
new = (150, 77, 12, 22, 55, 18)
new_n = ("Good", "Yes", "No") 
d = dict(zip(features_names, new))
dn = dict(zip(n_features_names, new_n))
d.update(dn)





for list in tree.tree:
    for tuple in list:
            for node in tuple:
                if node.name =="n1":
                    tree.pred_x(node, d)
'''

'''
def myfun():
    print("hello")

import dis
dis.dis(myfun)
'''
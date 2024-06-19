#Copyright 2024 Daniel Fynn
#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import itertools #base library
import math #base library
import numpy as np # use numpy arraysfrom
from  statistics import mean,variance,mode #base library
from anytree import Node, RenderTree, NodeMixin
from collections import Counter #base library
import matplotlib.pyplot as plt
import pydot
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import webbrowser #base library 
import random #base library 
import pandas as pd
import gc #base library 
import time #base library 

#rpy2 objects for lba
#kernel crashing could mean needing to set global environmental variable R_HOME = path to directory i.e. C:\Program Files\R\R-4.3.2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.rinterface as rinterface
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #issue searching an empty numpy array 

pd.options.mode.chained_assignment = None #settingwithcopywarning
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 3000)


class NodeClass(NodeMixin):  # Add Node feature #MyBaseClass
    
    children = []
    value_soglia_split = []
    beta = []
    alpha = []
    error = []
    global_predictability = []
    local_predictability = []
    node_prop = []
    node_prop_gain = []

    def __init__(self, name, indexes, split=None, parent=None,node_level= 0,to_pop = False):
        super(NodeClass, self).__init__()
        self.name = name                   # id n_node number
        self.indexes = indexes             # array of indexes of cases
        #self.impurity = impurity          # value in the node of the chosen impurity function
        self.split = split                 # string of the split (if any in the node, None => leaf)
        self.parent = parent               # parent node (if None => root node)
        self.node_level = node_level       # Tiene traccia del livello dei nodi all'interno dell albero in ordine crescente : il root node avrà livello 0
        self.to_pop = to_pop
        self.deviance = 0 
        self.surrogate_splits = [] #[stumps, variables, splits, between_var ]
        self.competitor_splits = []
    
    def get_value(self, y, problem):
        '''
        Returns the value of the node 
        '''
        if problem =='regression':
            return mean(y[self.indexes])
        else:
            response_dict ={}
            for response in y[self.indexes]:        #determing majority in  nodes
                if response in response_dict:
                    response_dict[response] +=1
                else:
                    response_dict[response] =1
        return max(response_dict, key = response_dict.get)


    def get_name_as_number(self):
        '''
        new name's node defination with integer
        '''
        return int(self.get_name()[1:])
    

    def get_children(self):
        '''
        ritorna il figlio se esiste altrimenti none
        returns the children of the node if they exist otherwise none 
        '''
        return self.children
    

    def get_value_thresh(self):
        '''
        returns the value and the threshold for alpha printing
        '''
        return self.value_soglia_split[0][0:2] + [self.value_soglia_split[0][3]]
        

    def set_to_pop(self):
        '''
        Durante il growing tiene traccia dei nodi da potare.
        During the growing, it tracks which nodes to remove
        '''
        self.to_pop = True 


    def get_name(self):
        '''
        returns the name of the node
        '''
        return self.name
    

    def get_level(self):
        '''
        returns the level of the node
        '''
        return self.node_level
    

    def set_features(self,features):
        '''
        sets the features of the node
        '''
        self.features = features
    

    def get_parent(self):
        '''
        return the parent node, if the the parent node is None it is the root.
        '''
        return self.parent
    

    def set_children(self,lista:list):#lista di nodi    
        '''
        sets the children of the node
        '''
        for i in lista:
            self.children.append(i)
    
    def set_split2(self, split):
        '''
        sets the split of the node to none, looks to do nothing
        '''
        self.split = split

    def set_split(self,value_soglia):
        '''
        sets the split of the node
        '''
        self.value_soglia_split = value_soglia
    
    def set_beta(self, beta):
        '''
        sets the beta values of the node for LBT
        '''
        self.beta = beta
    
    def set_alpha(self,alpha):
        '''
        sets the alpha values of the node for LBT
        '''
        self.alpha = alpha

    def set_error(self, error):
        '''
        sets the error values of the node for LBT
        '''
        self.error = error
    
    def set_global_predictability(self, gp, combination_split = False):
        '''
        sets the global_predictability values of the node for LBT
        '''
        if combination_split:
            self.global_predictability = [round(gp[0],3) , round(gp[1],3)]
        else:
            self.global_predictability = round(gp,3) 
    
    def set_local_predictability(self, varian):
        '''
        sets the loval_predictability values of the node for LBT
        '''
        self.local_predictability = round(varian,3)

    def set_node_prop(self,node_prop):
        '''
        sets the node_prop of the node 
        '''
        self.node_prop = node_prop

    def set_node_prop_gain(self,node_prop_gain):
        '''
        Sets the node proportion gain for the node
        '''
        self.node_prop_gain = node_prop_gain

    # define binary split mechanics (for numerical variables)
    def bin_split(self, feat, feat_nominal, var_name, soglia):
        '''
        Performs the birnary splitting of the indices 
        '''
        #_self_ is the node object, feat and feature_names (these could be better implemented via a *dict*)
        # var_name the string name and soglia the sogliashold
        
        if isinstance(feat, np.ndarray) and isinstance(feat_nominal, np.ndarray): #numpy input, not true continues if and or
            if feat.shape[0] >0:
                feat_names = feat.dtype.names
            else:
                feat_names = []
            
            if feat_nominal.shape[0] > 0:
                feat_nominal_names = feat_nominal.dtype.names
            else:
                feat_nominal_names = []


            if var_name in feat_names:         #is_numeric(var) :      # split for numerical variables
                self.split = var_name + " > " + str(round(soglia,2)) # compose the split string (just for numerical features)
                parent = self.name
                select = self.features[var_name][self.indexes] > soglia              # split cases belonging to the parent node
            elif var_name in feat_nominal_names:        #is_numeric(var) :      # split for nominal variables
                if type(soglia) is tuple:
                    self.split = var_name + " in " + str(soglia) # compose the split string (just for numerical features)
                elif isinstance(soglia, bytes):
                    self.split = var_name + " in " + str((soglia))
                else:
                    self.split = var_name + " in " + "'" +str(soglia)+"'" 

                parent = self.name
                select = np.array([i in soglia for i in feat_nominal[var_name][self.indexes]]) # split cases belonging to the parent node

            else :
                print("Var name is not among the supplied features!")
                return
        else:
            if var_name in feat:         #is_numeric(var) :      # split for numerical variables
                self.split = var_name + " > " + str(round(soglia,2)) # compose the split string (just for numerical features)
                parent = self.name
                select = self.features[var_name][self.indexes] > soglia              # split cases belonging to the parent node
            elif var_name in feat_nominal:         #is_numeric(var) :      # split for nominal variables
                #TODO may need to write more to allow for classes with a single char
                if type(soglia) is tuple:
                    self.split = var_name + " in " + str(soglia) # compose the split string (just for numerical features)
                else:
                    self.split = var_name + " in " + "'" +str(soglia)+"'" 

                parent = self.name
                select = np.array([i in soglia for i in feat_nominal[var_name][self.indexes]]) # split cases belonging to the parent node

            else :
                print("Var name is not among the supplied features!")
                return
        
        #to do if its a long index, breaking up the datafram will be faster
        left_i = self.indexes[~select]                      # to the left child criterion FALSE
        right_i = self.indexes[select]                      # to the right child criterion TRUE
        child_l = "n" + str(int(parent.replace("n",""))*2)
        child_r = "n" + str(int(parent.replace("n",""))*2 + 1)         
        return NodeClass(child_l, left_i, None, parent = self,node_level=self.node_level+1), NodeClass(child_r, right_i, None, parent = self,node_level=self.node_level+1)   # instantiate left & right children
            


class TREE4:
    '''
    TREE4 class with methods for tree growing and evaluation 
    '''

    def __init__(self,
                 y,
                 features,
                 features_names,
                 n_features, 
                 n_features_names,
                 impurity_fn = "between_variance",
                 user_impur=None, 
                 problem = "regression",  
                 method = "CART",
                 twoing = False,
                 min_cases_parent = 10, 
                 min_cases_child = 5, 
                 min_imp_gain=0.01, 
                 max_level = 10, 
                 surrogate_split = False):

        self.y = y
        self.features = features #needs to be an object that can be have its elements accessed with features[var] nomenculature
        self.features_names = features_names
        self.n_features = n_features
        self.n_features_names = n_features_names
        self.problem = problem
        self.impurity_fn = impurity_fn
        self.method = method
        self.user_impur = user_impur
        self.max_level = max_level
        self.twoing = twoing
        self.surrogate_split = surrogate_split

        self.dict_to_dataframe()

        self.combination_split = False            

        #accessing R libraries needed for running Latent Budget Tree 
        if self.method == "LATENT-BUDGET-TREE":
            robjects.r("library(utils, quietly = TRUE)")
            robjects.r("library(base, quietly = TRUE)")
            #robjects.r("suppressWarnings(install.packages('lba', quiet = TRUE))")
            robjects.r("suppressWarnings(suppressMessages(library(lba, quietly = TRUE)))")
        
        #setting the deviance in the response class pre-partitioning 
        if problem =="regression":
            self.devian_y = len(self.y)* self.RSS(y) # impurity function will equal between variance 
            #self.devian_y = len(self.y)*variance(self.y)
        elif problem == "classifier":
            pro = []
            c = Counter(y) 
            c = list(c.items())
            p = len(self.y)
            for i in  c:
                #prob = i[1]/len(self.y) #entrop
                prob = (i[1]/p)**2
                #pro.append(math.log(i[1]/len(y),2)*i[1]/len(y)) #entrop
                pro.append(prob)
            pro = np.array(pro)

            #self.devian_y = len(y)*np.sum(pro)*(-1) entrop
            self.devian_y = 1 - np.sum(pro) #gini
        
        self.grow_rules = dict({'min_cases_parent':min_cases_parent \
                                     ,'min_cases_child':min_cases_child \
                                     ,'min_imp_gain':min_imp_gain})

        #lists and objects used to store information / acess information while growin 
        self.bigtree =  []
        self.nsplit = 0
        self.father = []
        self.root = []
        self.tree = [] #made up of the parents and children [[(p,c,c)]] like this per object 
        self.father_to_pop = []
        self.node_prop_list = []
        self.node_prop_dict = {}
        #self.grow_rules = {}
        self.leaf = []
        self.all_node = []
        self.prediction_cat = []
        self.prediction_reg = []
        self.twoing_c1 = {}
        self.twoing_c2 = {}
        self.twoing_y = pd.DataFrame()
        self.pred_node = []

        #all objects with categorise_num are to do with creating categories when using two-stage methods to create bins using trees for numerical variable before calculating the pearson correlation value 
        self.categorise_num_start = False
        if self.method == "TWO-STAGE":
            self.categorise_num_start = True
        self.catergorise_num1 = False
        self.catergorise_num_big = []
        self.catergorise_num_father = []
        self.catergorise_num_np = []

        #some timing artefacts used for deciding which parts may need to be written in C
        #self.time = pd.DataFrame(columns = ["Function", "Lines", "Time"])
        #new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split, "n":len(node.indexes),  "Heterogeneity":node.deviance, "Explained Heterogeneity": exp_dev,"Class Probabilities":class_node,"Alpha":[node.alpha],"Beta":[node.beta],"LS Error":[node.error]})
        #tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
        #self.start = time.time()

    def user_impur_fn(self, func, node):
        '''user defined impurity fn'''
        return func(self, node)
    
    def impur(self,node, display = False):
        '''impurity calculator depending on choice of fn'''
        if self.problem =='regression':

            if self.impurity_fn =="between_variance":
                return (mean(self.y[node.indexes])**2)*len(self.y[node.indexes]) 
            
            elif self.impurity_fn == "pearson":
                #df = self.dict_to_dataframe()
                df = self.df.iloc[node.indexes]
                wss = 0 
                mean_y = mean(df["y"])
                for j in range(len(df["y"])):
                    wss += (df["y"].iloc[j] - mean_y)**2
                return wss

            elif self.impurity_fn =="user_defined":
                if self.user_impur:
                    return self.user_impur_fn(self.user_impur, node)
                else:
                    print("Must define 'user_impur' if selecting 'user_defined' for 'impur_fn'")     
            else:
                print("Impurity-fn only defined for between variance for regression problem.")
        
        elif self.problem == 'classifier':
            prom = 0
            c = Counter(self.y[node.indexes]) #Creates a dictionary {"yes":number, "no"}
            c = list(c.items())

            if self.impurity_fn =="gini":
                for i in  c:
                    prob_i = float((i[1]/len(self.y[node.indexes])))**2 
                    
                    if display:
                        prom += prob_i
                    else:
                        prom += prob_i*i[1]#/len(self.y[node.indexes]) #original weighted, only looking at purity # got rid of issues with TREE4 gini
                if display:
                    return 1-prom 
                else:  
                    return prom #for use in maximising fn
            
            elif self.impurity_fn =="entropy":
                for i in c:
                    prob_i = float((i[1]/len(self.y[node.indexes])))
                    prom += prob_i*math.log(prob_i,2) 
                return -prom#/len(c)
            
            elif self.impurity_fn =="user_defined":
                if self.user_impur:
                    return self.user_impur_fn(self.user_impur, node)
                else:
                    print("Must define 'user_impur' if selecting 'user_defined' for 'impur_fn'")
            
            elif self.impurity_fn == "tau": 
                for i in  c:
                    prom += float((i[1]/len(self.y[node.indexes])))**2 
                return prom
            
            else:
                print("For classification problem, impurity_fn must be set to either 'gini' or 'entropy' or 'user_defined'")
        else:
            print("'problem' must be classified as either 'regression' or 'classifier'")     

    def get_number_split(self):
        return self.nsplit
    
    def get_leaf(self):
        '''returns a list of NodeClass objects that make up the leaves of the fully grown tree tree'''
        leaf = [inode for inode in self.bigtree if inode not in self.get_father() ]
        le = []
        for i in leaf:
            if i not in le:
                le.append(i)
        self.leaf = [inode for inode in le if inode.to_pop == False]
        return   [inode for inode in le if inode.to_pop == False]
    
    def get_father(self):
        '''
        return all the node father
        '''
        return [inode for inode in self.father if inode not in self.father_to_pop]


    def get_root(self):
        '''
        returns the root
        '''
        return self.root
    
    def RSS(self, y ):
        '''
        return the RSS of a node this funcion is for only internal uses (private_funcion) pvariance() in statistics 
        '''
        mean_y = mean(y)
        val = []
        for i in y:
            val.append((i - mean_y)**2)
        return sum(val)  / len(val)

    def __get_RSS(self,node):
        '''
        return the RSS of a node this funcion is for only internal uses (private_funcion)
        '''
        mean_y = mean(self.y[node.indexes])
        return (1/len(node.indexes)*sum((self.y[node.indexes] - mean_y)**2))

    def get_all_node(self):
        '''returns a list of NodeClass objects that are the nodes of the fully grown tree'''
        foglie = [nodi for nodi in self.get_leaf()]
        self.all_node = foglie + self.get_father()
        return foglie + self.get_father()
    
    def dict_to_dataframe(self):
        '''Returns a dataframe with all numerical and categorical variables initialised in 
        TREE4, and the feature variable, with column heading "y"'''
        
        df = pd.DataFrame(self.features, columns = self.features_names)
        df2 = pd.DataFrame(self.n_features, columns = self.n_features_names)
        df = pd.concat([df, df2], axis = 1)
        df["y"] = self.y
        self.df = df
        #return df
    
    def gini(self, node): 
        '''Returns gini value for teh node of interest for the response variable'''
        #df = self.dict_to_dataframe()
        df = self.df.iloc[node.indexes]
        gini = 0
        for j in list(set(df["y"])):
            gini += (len(df.loc[df["y"] == j])/len(df))**2
        return gini

    def tau_ordering(self, node):
        '''Returns the predictors ordered based on tau values as per two stage methods'''

        #df = self.dict_to_dataframe()
        df = self.df.iloc[node.indexes]
        gini = self.gini(node)

        tau_list = []
        for var in self.features_names+ self.n_features_names:
            sum_rel_freq = 0 
            for i in list(set(df[var])):
                df2 = df.loc[df[var]==i]
                for j in list(set(df2["y"])):
                    sum_rel_freq += (len(df2.loc[df2["y"] == j])/len(df2))**2 * len(df2)/len(df) 
            tau_list.append(((sum_rel_freq - gini) / (1-gini), var))
        tau_list.sort(reverse = True)
        return tau_list

    def wss(self, listob):
        '''Returns the within sum of squares'''
        wss = 0 
        meanx = mean(listob)
        for i in listob:
            wss += (i - meanx)**2
        return wss*len(listob)

    def catergorise_num(self, node, var):
        '''For categorising the numerical predictors when using the pearson correlation, this is like nearest neighbour, but is partitioned with trees'''

        #setting up
        self.catergorise_num_supervised = True

        self.catergorise_num1 = True
        oldmethod = self.method
        self.method = "CART"
        oldimpur = self.impurity_fn
        self.impurity_fn = "between_variance"
        oldnfeaturesnames = self.n_features_names
        self.n_features_names = []
        oldfeaturesnames = self.features_names
        self.features_names = [var]

        oldy = self.y
        if not self.catergorise_num_supervised:
            self.y = self.df[var]
        
        olddevian = self.devian_y
        self.devian_y = len(self.y[node.indexes])* self.RSS(self.y[node.indexes])
        oldlevel = self.max_level
        self.max_level = 10 #undo user input 
        categorise = np.array(self.df[var])
        
        #growing a new tree based off node1 as the root 
        node1 = NodeClass('n1', node.indexes) 
        self.growing_tree(node1)

        #evaluating tree, appending mean value of node as category
        leaf = 0

        for inode in self.catergorise_num_big:
            if inode not in self.catergorise_num_father:
                categorise[inode.indexes] =  mean(self.features[var][inode.indexes])
                leaf +=1
        
        #undoing previous setting up
        self.catergorise_num1 = False
        self.catergorise_num_big = []
        self.catergorise_num_father = []
        self.catergorise_num_np = []
        self.method = oldmethod
        self.impurity_fn = oldimpur
        self.n_features_names = oldnfeaturesnames
        self.features_names = oldfeaturesnames
        
        self.y = oldy
        self.devian_y = olddevian
        self.max_level = oldlevel

        return categorise
    
    def tss(self, node):
        '''returns the total sum of squares of the response variable for a given node'''
        #df = self.dict_to_dataframe()
        df = self.df.iloc[node.indexes]
        tss = 0
        mean_y = mean(df["y"])
        for j in range(len(df["y"])):
            tss += (df["y"].iloc[j] - mean_y)**2
        return tss

    def pearson_ordering(self, node):
        '''Returns the predictors ordered based on pearson values as per two stage methods'''
        #self.categorise_num_start  = False turns off categorising
        if self.categorise_num_start:
            self.df2 = self.df.copy()
            dfa = np.array(self.df2)
        else:
            dfa = np.array(self.df)
        #self.dict_to_dataframe() #uncommented as y is updated , recomented as only need a tree with full data
        #dfa = np.array(self.df2)#2
        dfa = dfa[node.indexes,:]
        #df = df.iloc[node.indexes]
        tss = self.tss(node)
        self.categorise_num_tss = tss 

        pearson_list = []
        for en, var in enumerate(self.features_names+ self.n_features_names):  #reworked a bit to work with images
            wss = 0 
            if var in self.features_names and len(list(set(dfa[:,int(en)]))) > 10 and self.categorise_num_start:
                categorise= self.catergorise_num(node, var)  #unsupervised
                dfa[:,int(en)] = categorise
                self.df2[var] = categorise
            if len(list(set(dfa[:,int(en)]))) > 1: #dfa[:,int(en)]
                #print("lenth unique values",len(list(set(dfa[:,int(en)]))))
                for i in list(set(dfa[:,int(en)])):
                    #df2 = df.loc[df[var]==i]
                    df2a = dfa[dfa[:,int(en)] == i,: ]
                    #if len(df2a["y"]) > 1: #there is only a within, when theres more than 1, otherwise its 0 
                    if len(df2a[:,-1]) > 1: 
                        mean_y = mean(df2a[:,-1])
                        for j in range(len(df2a[:-1])):
                            wss += (df2a[j,-1]- mean_y)**2 #"y"].iloc[j] 
                pearson_list.append((1- wss/ tss, var))
        pearson_list.sort(reverse = True)
        
        self.dict_to_dataframe() #reset after finishing
        self.categorise_num_start = False
        return pearson_list


    def midway_points(self, var, node):
        '''Find the midway point for continious variables for use when selecting a split'''
        midway_points = []

        uniques = list(set(self.features[str(var)][node.indexes]))
        uniques.sort()
        for i in range(len(uniques)-1):
            midway_points.append((uniques[i]+uniques[i+1])/2)
        
        return midway_points

    def __node_search_split(self,node:NodeClass, max_k, combination_split, max_c):

        '''
        The function return the best split that the node may compute.
        Il calcolo è effettuato effettuando ogni possibile split e 
        calcolando la massima between variance 
        tra i nodi figli creati.
       
       Attenzione: questo è un metodo privato non chiamabile a di fuori della classe.

       The algorithm takes into account each possible split, and calculates the maximum variance between the nodes of the children created (CART). 
       This algotihm also finds the best split in a similar way for TWO-STAGE, FAST and LATENT-BUDGET-TREE. 
       Attention this is a private methd not callable outside the class
        '''
        
        impurities_1=[]
        between_variance=[]
        splits=[]
        variables=[]
        gp = []
        stumps = []
        distinct_values=np.array([])
        t=0
        k = False
        
        node.set_features(self.features)
        
        #checks for node purity
        if Counter(self.y[node.indexes]).most_common(1)[0][1] == len(self.y[node.indexes]):

            print("This split isn't good now i cut it [counter] - node class purity")
            node.get_parent().set_to_pop()
            node.get_parent().set_to_pop()
            self.father_to_pop.append(node)
            node.set_split2(None)
            return None

        if not combination_split:
            max_c = 1

        if len(node.indexes) >= self.grow_rules['min_cases_parent']:
            
            #will implement as two-stage, finidng the best split of the highest tau
            #best split in comp98_sici uses modified aic to choose best split
            #stage 1
            if self.method == "LATENT-BUDGET-TREE": #classification only method     #could pass the impurity fn as the method to use in lba [ls or mle] as not used
                #t = time.time()

                #TODO Currently LATENT-BUDGET-TREE prunes based on misclassification rate, rather than a different metric, that may better encompass the multi-class nature of the problem. 

                #new_df = pd.DataFrame({"Function":"nodesearch b4 order lbt", "Lines":663, "Time": time.time() - self.start}, index = [0])
                #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)

                #ordering predictors at current node
                if self.problem == "classifier":
                    ordered_list = self.tau_ordering(node)  
                else:
                    print("Latent Budget Tree only works with Classifier response variable")
                    return None
                
                #new_df = pd.DataFrame({"Function":"nodesearch after order lbt", "Lines":672, "Time": time.time() - self.start}, index = [0])
                #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)

                df = self.df.iloc[node.indexes].copy()

                betas = []
                alphas = []
                errors = []
                k = -1
                
                #going through the k ordered predictors
                while k < len(ordered_list)-1:
                    k +=1   
                    n = 0                   
                    #for combined variables up to max_c
                    while n <= max_c: 
                        n+=1
                        if combination_split:
                            if k  < len(ordered_list)-n: #combines the ajoined high tau values, as soon as it hits the bottom once it will stop, max_c should be a third of predictors max
                                comb_split = str(ordered_list[k][1])+"__"+str(ordered_list[k+n][1])
                                df[comb_split] = df[ordered_list[k][1]] + df[ordered_list[k+n][1]] 
                                cont = pd.crosstab(index = df[comb_split], columns= df["y"], normalize = 'index')
                            else:
                                if len(splits):
                                    print("Unable to go through max_k (* max_c), only went through: ", len(splits), "time/s")
                                    best_index = between_variance.index(max(between_variance))
                                    node.set_beta(betas[best_index])
                                    node.set_alpha(alphas[best_index])
                                    #node.set_error(errors[best_index])
                                    node.set_global_predictability(gp[best_index])

                                    var1, var2 = variables[best_index].split("__")
                                    self.n_features[variables[best_index]] = self.n_features[var1] + self.n_features[var2]
                                    
                                    self.nss_variables = variables
                                    self.nss_splits = splits
                                    self.nss_between_variance = between_variance
                                    self.nss_stumps = stumps

                                    return variables[best_index], tuple(splits[best_index]), between_variance[best_index], stumps[best_index]
                                else:
                                    print("No splits found")
                                    return None
                        else:
                            #creates crosstable                      
                            cont = pd.crosstab(index = df[ordered_list[k][1]], columns= df["y"], normalize = 'index')
                            if cont.shape[0] <=1: #implemented to avoid errors in lba 
                                continue
                        #converts into an r dataframe
                        with (robjects.default_converter + pandas2ri.converter).context():
                            cont_r = robjects.conversion.get_conversion().py2rpy(cont)

                        #new_df = pd.DataFrame({"Function":"nodesearch before model lbt", "Lines":714, "Time": time.time() - self.start}, index = [0])
                        #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)
                        try:
                            robjects.r.assign("cont_r", cont_r)
                            robjects.r("cont_r <-  as.matrix(cont_r)")
                            robjects.r("set.seed(2)")
                            robjects.r("suppressWarnings(suppressMessages(out <- lba(cont_r, K = 2 , what = 'outer', method = 'ls', trace.lba = FALSE)))") 
                            robjects.r("alpha <- out$A")
                            alpha = robjects.r('alpha')
                            alpha = np.asarray(alpha)

                            robjects.r("beta <- t(out$B)")
                            beta = robjects.r('beta')
                            beta = np.asarray(beta)

                            #robjects.r("error <- out$val_func")
                            #error = robjects.r('error')
                            #error = np.asarray(error)
                            #error = -round(error.item(), 16)
                            #out = lba.lba(base.as_matrix(cont_r), K = 2 , what = 'outer', method = 'ls') #base.trace.lba = 0 doesnt work
                        except:
                            print("Error in LBA function")
                            time.sleep(4) #issue with printing order bewtten python n r 
                            continue
                        
                        #new_df = pd.DataFrame({"Function":"nodesearch after model lbt", "Lines":714, "Time": time.time() - self.start}, index = [0])
                        #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)

                        #assessing results from lba for alpha 
                        split = []
                        for i in range(alpha.shape[0]):
                            if alpha[i][0] >= 0.5:            #threshold point set to 0.5, what if the alphas are less than 0.5 for both groups, i think it gets caught later by teh delta fn
                                split.append(cont.index[i])

                        #determinging whether there has been splits foind from the lba model, if so evaluating the split in terms of impurity like CART
                        
                        if split and len(split) != alpha.shape[0]: #len(set(df[ordered_list[k][1]])):  #looks that at least 1 alpha > 0.5, and not all values

                            if not combination_split:
                                stump = node.bin_split(self.features, self.n_features, str(ordered_list[k][1]),split)
                            else:
                                #print("combination_split", df.columns, comb_split)
                                stump = node.bin_split(self.features, df.copy() , comb_split,split) #may not work later when evaluating in the table

                            #print("stumpcheck", self.y[stump[0].indexes].size, self.y[stump[1].indexes].size )

                            if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child'] \
                                and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:

                                impur0 = self.impur(stump[0])
                                impur1 = self.impur(stump[1])

                                splits.append(split) #had list around it , had -1index
                                stumps.append(stump)
                                if combination_split:
                                    variables.append(comb_split)
                                    gp.append([ordered_list[k][0],ordered_list[k+n][0]])
                                else:
                                    variables.append(ordered_list[k][1])
                                    gp.append(ordered_list[k][0])
                                betas.append(np.around(beta,2).tolist()) #before were still arrays
                                alphas.append(np.around(alpha,2).tolist())
                                #errors.append(np.around(error,2).tolist()) #this stored the error for the lba model 

                                if self.impurity_fn =="entropy":
                                    entropy_parent = self.impur(node)
                                    inf_gain = entropy_parent - ((len(stump[0].indexes) / len(node.indexes)) * impur0 + (len(stump[1].indexes) / len(node.indexes)) * impur1)
                                    
                                    between_variance.append(inf_gain)                                
                                else:
                                    between_variance.append((impur0) + (impur1)) 
                            else:
                                continue
                        else:
                            #when no split is found with alpha greater than 0.5, or len(split) is = len(set(var))
                            continue

                        #max_k = 2 #allows for selecting the first max_k complete splits aka no error from lba
                        if len(splits) >= max_k * max_c: #max k can be a user controlled variable, passed to the TREE4 class , or to growing_tree
                            best_index = between_variance.index(max(between_variance))
                            node.set_beta(betas[best_index])
                            node.set_alpha(alphas[best_index]) #np.around(alphas[best_index],4).tolist())
                            #node.set_error(errors[best_index])
                            node.set_global_predictability(gp[best_index], combination_split)
                            if combination_split:
                                var1, var2 = variables[best_index].split("__")
                                #print("combs", self.n_features[var1][0:5], self.n_features[var2][0:5])
                                self.n_features[variables[best_index]] = self.n_features[var1] + self.n_features[var2]
                            
                            
                            self.nss_variables = variables
                            self.nss_splits = splits
                            self.nss_between_variance = between_variance
                            self.nss_stumps = stumps    
                            
                            return variables[best_index], tuple(splits[best_index]), between_variance[best_index], stumps[best_index]    #"latent_budget_tree doesnt return an error" 
                        else:
                            continue

            elif self.method == "FAST" or self.method == "TWO-STAGE":
                
                #TODO include entropy, and shannon

                #new_df = pd.DataFrame({"Function":"nodesearch b4 order ", "Lines":798, "Time": time.time() - self.start}, index = [0])
                #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)
                
                #ordering predictors according to tau or pearson
                if self.problem == "classifier":
                    ordered_list = self.tau_ordering(node)  
                else:
                    ordered_list = self.pearson_ordering(node) 
                #print("ordered_list",ordered_list)
                
                #new_df = pd.DataFrame({"Function":"nodesearch after order ", "Lines":805, "Time": time.time() - self.start}, index = [0])
                #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)
                
                k = 0 #iterator 
                while k < len(ordered_list)-1:       #stopping rule, iterating through k ordered predictors
                    between_variance_k=[]
                    splits_k=[]
                    variables_k=[]
                    stumps_k = []
                    
                    if ordered_list[k][1] in self.n_features_names:
                        cat_var = [ordered_list[k][1]]
                        num_var = []
                    else:
                        num_var = [ordered_list[k][1]]
                        cat_var = []
                    
                    for var in cat_var:  
                        combinazioni = []
                        distinct_values= []
                        distinct_values.append(list(set(self.n_features[str(var)][node.indexes])))
                        distinct_values = list(itertools.chain(*distinct_values)) #flattens, removed nesting
                        for i in range(1,len(distinct_values)): 
                            combinazioni.append(list(itertools.combinations(distinct_values, i)))
                        combinazioni = combinazioni[1:]
                        combinazioni = list(itertools.chain(*combinazioni))
                        combinazioni = combinazioni +  distinct_values

                        #TODO put everything as nested list?
                        '''  new_comb = []
                            for i in combinazioni:
                                try:
                                    if len(list(i))>1:
                                        small_combs = []
                                        for j in range(len(i)):
                                            small_combs.append(i[j])
                                        new_comb.append(small_combs)
                                except:
                                    new_comb.append([i])
                        '''
                        for i in combinazioni: 
                            stump = node.bin_split(self.features, self.n_features, str(var),i)
                            if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child'] \
                                and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:
                                impur0 = self.impur(stump[0])
                                impur1 = self.impur(stump[1])
                                if self.problem == 'classifier' and self.impurity_fn == "tau":    
                                    gini_parent = self.impur(node)
                                    tau = (impur0 * len(stump[0].indexes) / len(node.indexes) + impur1 * len(stump[1].indexes)/ len(node.indexes) - gini_parent) / (1- gini_parent)
                                    between_variance_k.append(tau)
                                    between_variance.append(tau)
                                elif self.problem == "regression" and self.impurity_fn == "pearson": 
                                    impurities_1.append(impur0)
                                    impurities_1.append(impur1)
                                    between_variance_k.append(1- sum(impurities_1[t:]) / self.tss(node)) #exploratory slides 43
                                    between_variance.append(1- sum(impurities_1[t:]) / self.tss(node))
                                else:
                                    print("Error, Two-Stage and FAST algorithm require impurity_fn as tau for classifier, \
                                          and pearson for regression")
                                    return None
                                splits_k.append(i)
                                splits.append(i)
                                variables_k.append(str(var))
                                variables.append(str(var))
                                gp.append(ordered_list[k][0])
                                stumps_k.append(stump)
                                stumps.append(stump)
                                t+=2
                            else:
                                continue
                        #else:
                        #    print("NaN found in observation")
                        #    continue            
                        
                    for var in num_var:                      

                        for i in self.midway_points(var,node):#range(len(set(self.features[str(var)][node.indexes]))): 
                                stump = node.bin_split(self.features, self.n_features, str(var), i)#self.features[str(var)][i])
                                if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child'] \
                                    and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:
                                    impur0 = self.impur(stump[0])
                                    impur1 = self.impur(stump[1])
                                    if self.problem == 'classifier' and self.impurity_fn == "tau":    
                                        gini_parent = self.impur(node)
                                        tau = (impur0 * len(stump[0].indexes) / len(node.indexes) + impur1 * len(stump[1].indexes)/ len(node.indexes) - gini_parent) / (1- gini_parent)
                                        between_variance_k.append(tau)
                                        between_variance.append(tau)
                                    elif self.problem == "regression" and self.impurity_fn == "pearson": 
                                        impurities_1.append(impur0)
                                        impurities_1.append(impur1)
                                        between_variance_k.append(1- sum(impurities_1[t:])/ self.tss(node))
                                        between_variance.append(1- sum(impurities_1[t:])/ self.tss(node))
                                       
                                    else:
                                        print("Error, Two-Stage and FAST algorithm require impurity_fn as tau for classifier, \
                                          and pearson for regression")
                                        return None
                                    splits_k.append(i)#self.features[str(var)][i])
                                    splits.append(i)
                                    variables_k.append(str(var))
                                    variables.append(str(var))
                                    gp.append(ordered_list[k][0])
                                    stumps_k.append(stump)
                                    stumps.append(stump)
                                    t+=2
                                else: 
                                    continue
                        #else:
                        #    print("NaN found in observation")
                        #    continue 
                    try:                  
                        if k == 0:
                            #evaluation for the current k 
                            s_star_k = max(between_variance_k)  
                            s_star_k_between = between_variance_k[between_variance_k.index(max(between_variance_k))] 
                            s_star_k_split = splits_k[between_variance_k.index(max(between_variance_k))]
                            s_star_k_variable = variables_k[between_variance_k.index(max(between_variance_k))]
                            s_star_k_stump = stumps_k[between_variance_k.index(max(between_variance_k))]
                            if self.method == "TWO-STAGE" and max_k == 1: 
                                
                                self.nss_variables = variables
                                self.nss_splits = splits
                                self.nss_between_variance = between_variance
                                self.nss_stumps = stumps                             
                                node.set_global_predictability(gp[between_variance.index(max(between_variance))])

                                return s_star_k_variable, s_star_k_split, s_star_k_between, s_star_k_stump 
                    except:
                        k += 1
                        s_star_k = 0
                        continue
                    try:
                        #updating best split values
                        if k != 0 and max(between_variance_k) > s_star_k:
                            s_star_k = max(between_variance_k) 
                            s_star_k_between = between_variance_k[between_variance_k.index(max(between_variance_k))]
                            s_star_k_split = splits_k[between_variance_k.index(max(between_variance_k))]
                            s_star_k_variable = variables_k[between_variance_k.index(max(between_variance_k))]
                            s_star_k_stump = stumps_k[between_variance_k.index(max(between_variance_k))]
                    except: 
                        k +=1 #failing minimum child size condition
                        continue
                    
                    
                    if self.method == "TWO-STAGE":
                        if max_k == 1:         ##if initial iteration fails to get a result  #len(s_star_k_between) == 1 had previous, but to get to this point cant have error
                            self.nss_variables = variables
                            self.nss_splits = splits
                            self.nss_between_variance = between_variance
                            self.nss_stumps = stumps
                            node.set_global_predictability(gp[between_variance.index(max(between_variance))])
                            return s_star_k_variable, s_star_k_split, s_star_k_between, s_star_k_stump
                        elif k >= max_k-1:
                            self.nss_variables = variables
                            self.nss_splits = splits
                            self.nss_between_variance = between_variance
                            self.nss_stumps = stumps
                            node.set_global_predictability(gp[between_variance.index(max(between_variance))])
                            return s_star_k_variable, s_star_k_split, s_star_k_between, s_star_k_stump
                        else:
                            k +=1
                    if self.method == "FAST":
                        if s_star_k < ordered_list[k+1][0] :  #termination for FAST algoirthm
                            k += 1
                        else:
                            self.nss_variables = variables
                            self.nss_splits = splits
                            self.nss_between_variance = between_variance
                            self.nss_stumps = stumps
                            node.set_global_predictability(gp[between_variance.index(max(between_variance))])
                            return s_star_k_variable, s_star_k_split, s_star_k_between, s_star_k_stump
                    
                
                try:
                    self.nss_variables = variables
                    self.nss_splits = splits
                    self.nss_between_variance = between_variance
                    self.nss_stumps = stumps
                    node.set_global_predictability(gp[between_variance.index(max(between_variance))])
                    return s_star_k_variable, s_star_k_split, s_star_k_between, s_star_k_stump #if all fails after all variables 
                except:
                    return None
        

            #had issues with having a boolean predictor 
            elif self.method == "CART":
                for var in self.n_features_names:
                    
                    combinazioni = []
                    distinct_values= [] #was np before
                    distinct_values.append(list(set(self.n_features[str(var)])))
                    distinct_values = list(itertools.chain(*distinct_values)) #flattens, removed nesting
                    for i in range(1,len(distinct_values)):
                        combinazioni.append(list(itertools.combinations(distinct_values, i)))
                    combinazioni=combinazioni[1:]
                    combinazioni = list(itertools.chain(*combinazioni))
                    combinazioni = combinazioni +  distinct_values
                    
                    #new_df = pd.DataFrame({"Function":"nodesearch nominal "+ str(var)+" " + str(len(combinazioni)) + " time per iterable object " + str( len(combinazioni) / (time.time()-self.time["Time"].to_list()[-1])), "Lines":947, "Time": time.time() - self.start}, index = [0])
                    #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)
                    
                    for i in combinazioni: 
                        stump = node.bin_split(self.features, self.n_features, str(var),i)
                        
                        if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child'] \
                            and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:
                            
                            impur0 = self.impur(stump[0])
                            impur1 = self.impur(stump[1])
                            if self.problem == 'classifier':    
                                if self.impurity_fn =="entropy":
                                    entropy_parent = self.impur(node)
                                    inf_gain = entropy_parent - ((len(stump[0].indexes) / len(node.indexes)) * impur0 + (len(stump[1].indexes) / len(node.indexes)) * impur1)
                                    between_variance.append(inf_gain)                                
                                else:
                                    between_variance.append((impur0) + (impur1))
                            else: 
                                impurities_1.append(impur0)
                                impurities_1.append(impur1)
                                between_variance.append(sum(impurities_1[t:]))
                                

                            splits.append(i)
                            variables.append(str(var))
                            stumps.append(stump)
                            t+=2
                            #print(splits[-1], variables[-1], between_variance[-1])
                    else:
                        continue
                        

                #print("self",self.features_names)
                for var in self.features_names:
                    mp = self.midway_points(var,node)
                    for i in  mp:#range(len(set(self.features[str(var)][node.indexes]))):
                        stump = node.bin_split(self.features, self.n_features, str(var), i) #self.features[str(var)][i])
                        if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child'] \
                            and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:
                            impur0 = self.impur(stump[0])
                            impur1 = self.impur(stump[1])
                            if self.problem == 'classifier':    
                                if self.impurity_fn =="entropy":
                                    entropy_parent = self.impur(node)
                                    inf_gain = entropy_parent - ((len(stump[0].indexes) / len(node.indexes)) * impur0 + (len(stump[1].indexes) / len(node.indexes)) * impur1)
                                    between_variance.append(inf_gain)       
                                else:
                                    between_variance.append((impur0) + (impur1))
                            
                            else: 
                                impurities_1.append(impur0)
                                impurities_1.append(impur1)
                                between_variance.append(sum(impurities_1[t:]))
                            
                            splits.append(i)#self.features[str(var)][i])
                            variables.append(str(var))
                            stumps.append(stump)
                            t+=2
                        else: 
                            continue

                    #new_df = pd.DataFrame({"Function":"nodesearch nominal "+ str(var)+" " + str(len(mp)) + " time per iterable object " + str( len(mp) / (time.time() -self.time["Time"].to_list()[-1]) ), "Lines":990, "Time": time.time() - self.start}, index = [0])
                    #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)
            else:
                print("Method given is not included")
        try:
            #print("max",max(between_variance))
            if self.method == "LATENT-BUDGET-TREE":
                #print("betweenvar", between_variance)
                self.nss_variables = variables
                self.nss_splits = splits
                self.nss_between_variance = between_variance
                self.nss_stumps = stumps

                return variables[between_variance.index(max(between_variance))],tuple(splits[between_variance.index(max(between_variance))]),between_variance[between_variance.index(max(between_variance))], stumps[between_variance.index(max(between_variance))]
            else:

                self.nss_variables = variables
                self.nss_splits = splits
                self.nss_between_variance = between_variance
                self.nss_stumps = stumps
                return variables[between_variance.index(max(between_variance))],splits[between_variance.index(max(between_variance))],between_variance[between_variance.index(max(between_variance))], stumps[between_variance.index(max(between_variance))]
        except:
            #this is mostly an error where the length is less than min size 
            if not self.catergorise_num1:
                if len(node.indexes) < self.grow_rules['min_cases_parent']:
                    print("Node Search Split Error for: ", node.name, "with obs in node: ", len(node.indexes), " which is less than minimum parent size: ",self.grow_rules['min_cases_parent'] )
                else:
                    if k:
                        print("Node Search Split Error for: ",node.name, "couldn't find an appropiate looking at k variables = ", k)
                    else:
                        print("Node Search Split Error for: ",node.name, "couldn't find an appropiate looking at k variables")
                node.get_parent().set_to_pop()
                #node.get_parent().set_to_pop()
                self.father_to_pop.append(node)
                return None
            else:
                return None

    def find(self, lst, var):
        '''finding indicies for matches in list'''
        return [i for i, x in enumerate(lst) if x == var]

    
    def surrogate_splits(self,node, overlap = 0.65, max_sur = 5):
        '''Attaches the possible surrogate splits to the node object...still a working progress'''

        bestvar = self.nss_variables[self.nss_between_variance.index(max(self.nss_between_variance))]
        beststump = self.nss_stumps[self.nss_between_variance.index(max(self.nss_between_variance))]

        leftind = Counter(beststump[0].indexes) #changes it into a dict
        rightind = Counter(beststump[1].indexes) 

        vars = list(set(self.nss_variables))
        vars.remove(bestvar) #gets rid of best var 
        
        surrogates = []

        for i in vars: #for each unique predictor 
            indexes = self.find(self.nss_variables, i) #find each index that relates to that variable
            possibilities = []
            for j in indexes: #iterate through these
                leftmatch = len(set(leftind).intersection(Counter(self.nss_stumps[j][0].indexes)))  
                rightmatch = len(set(rightind).intersection(Counter(self.nss_stumps[j][1].indexes))) 
                possibilities.append([(leftmatch + rightmatch)/(len(beststump[0].indexes) + len(beststump[1].indexes)),j , leftmatch, rightmatch,len(beststump[0].indexes) , len(beststump[1].indexes) ])
            m = max(possibilities)
            surrogates.append([m[0], m[1], i ])
        surrogates.sort(reverse = True)        

        if len(vars) < max_sur:
            max_sur = len(vars)

        for i in range(max_sur):
            if surrogates[i][2] in self.features_names: #numeric variable
                split = surrogates[i][2] + " > " + str(round(self.nss_splits[surrogates[i][1]], 2))
            else:
                if type(self.nss_splits[surrogates[i][1]]) is tuple:
                    split = surrogates[i][2] + " in " + str(self.nss_splits[surrogates[i][1]]) # compose the split string (just for numerical features)
                else:
                    split = surrogates[i][2] + " in " + "'" +str(self.nss_splits[surrogates[i][1]]) +"'"
                                                       
            node.surrogate_splits.append([ split, self.nss_between_variance[surrogates[i][1]], surrogates[i][0] ]) #var, split, bwteen, overlap%

    def competitor_splits(self,node, max_comp = 5):
        '''Attaches the possible competitor splits to the node object...still a working progress'''

        bestvar = self.nss_variables[self.nss_between_variance.index(max(self.nss_between_variance))]
        beststump = self.nss_stumps[self.nss_between_variance.index(max(self.nss_between_variance))] 

        vars = list(set(self.nss_variables))
        vars.remove(bestvar) #gets rid of best var 

        competitors = []
        for i in vars: #for each unique predictor 
            indexes = self.find(self.nss_variables, i) #find each index that relates to that variable
            
            possibilities = []
            for j in indexes:
                possibilities.append([self.nss_between_variance[j],j])
            m = max(possibilities)
            competitors.append([m[0], m[1], i ])
        competitors.sort(reverse = True)        

        #for i in range(max_comp):
        #    node.competitor_splits.append([competitors[i][2], self.nss_splits[competitors[i][1]], self.nss_between_variance[competitors[i][1]] ]) #var, split, bwteen, overlap%

        if len(vars) < max_comp:
            max_comp = len(vars)

        for i in range(max_comp):
            if competitors[i][2] in self.features_names: #numeric variable
                split = competitors[i][2] + " > " + str(round(self.nss_splits[competitors[i][1]], 2))
            else:
                if type(self.nss_splits[competitors[i][1]]) is tuple:
                    split = competitors[i][2] + " in " + str(self.nss_splits[competitors[i][1]]) # compose the split string (just for numerical features)
                else:
                    split = competitors[i][2] + " in " + "'" +str(self.nss_splits[competitors[i][1]]) +"'"
                                                       
            node.competitor_splits.append([ split, self.nss_between_variance[competitors[i][1]], competitors[i][0] ]) #var, split, bwteen, overlap%

    def control(self):
        '''Checks whetehr there is a pure node '''
        for i in self.get_leaf():
            for j in self.get_leaf():
                if i.get_parent() == j.get_parent():
                    if mode(self.y[i.indexes]) == mode(self.y[j.indexes]):
                        #i.set_to_pop()
                        #set_to_pop()
                        self.father_to_pop.append(i.get_parent)
        
    def deviance_cat(self,node):
        '''Calcuates the deviance for categorical variables, using gini'''
        #entropy
        pro = []
        c = Counter(self.y[node.indexes])
        c = list(c.items())
        p = len(self.y[node.indexes])
        for i in  c:
            #prob = i[1]/p entrop
            prob = (i[1]/p)**2 #gini
            pro.append(prob) #gini
            #pro.append(math.log(prob,2) * prob) entrop
        pro = np.array(pro)
        #ex_deviance = -1*np.sum(pro)  #entropy
        ex_deviance =1- np.sum(pro) #gini
        #print(ex_deviance)
        return ex_deviance
    
    def deviance_cat2(self,node):
        '''Calculated the deviance for categorical vars using fn from MASS pg 256'''
        #MASS page 256
        pro = []
        c = Counter(self.y[node.indexes])
        c = list(c.items())
        p = len(self.y[node.indexes])
        for i in  c:
            prob = i[1]/p
            pro.append(math.log(prob) * i[1])
        pro = np.array(pro)
        ex_deviance = -2*np.sum(pro) 
        return ex_deviance

    def prop_nodo(self,node):
        '''Calculates the proportion of the node'''
        c = Counter(self.y[node.indexes])
        c = list(c.items())
        p = len(self.y[node.indexes])
        xlen = len(self.y)
        somm=  0
        for i in  c:
            prob = i[1]/p
            somm +=prob
        return prob
            
    def growing_tree(self,node:Node,rout='start',propotion_total=0.9, max_k = 1, combination_split = False, max_c = 1):
        '''Main function of TREE4, for growing the tree, aka partitioning the nodes and filling objects to present these nodes'''
        #new_df = pd.DataFrame({"Function":"growing_tree", "Lines":1102, "Time": time.time() - self.start}, index = [0])
        #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)

        value_soglia_variance = []
        mini_tree = [] 

        if self.method == "LATENT-BUDGET-TREE":
            self.combination_split = combination_split

        level = node.get_level()

        if level == 0 and combination_split:
            self.n_features = pd.DataFrame(self.n_features) #for numpy adding combination vars to ndarray is hard 
        #print("level",level, node.name)
        if level > self.max_level:
            return None 

        #twoing, a CART method that can be used for grouping multiclass responses into binary, and also for numerical variables, as an ordered variable
        #goes through each possible comination of the response, and find each respective best split, a bit intensive. 
        if self.twoing:
            
            nss_variables = []
            nss_splits = []
            nss_between_variance = []
            nss_stumps = []
            
            #TODO utilise function P_l*P_r / 4 [sum|p(j|t_l) - p(j|t_r)|]**2 for computational efficiency 
            #TODO a different way to reduce imputations: https://support.minitab.com/en-us/minitab/20/help-and-how-to/statistical-modeling/predictive-analytics/how-to/cart-classification/methods-and-formulas/node-splitting-methods/#twoing-criterion
            if self.problem == "classifier":
                yold = self.y #keep it in local
                combinazioni = []
                distinct_values= []
                distinct_values.append(list(set(yold[node.indexes])))
                distinct_values = list(itertools.chain(*distinct_values)) #flattens, removed nesting
                for i in range(1,len(distinct_values)):
                    combinazioni.append(list(itertools.combinations(distinct_values, i)))
                combinazioni=combinazioni[1:]
                combinazioni = list(itertools.chain(*combinazioni))
                combinazioni = combinazioni +  distinct_values

                c1, c2 = [], []
                for i in combinazioni:
                    if isinstance(i, int) or isinstance(i, np.int64): 
                        if [i] not in c2:           #just increasing efficiency by not doing the same split for c1,c2
                            c1.append([i])
                        if list(set(yold)-{i}) not in c1:
                            c2.append(list(set(yold)-{i}))
                    else:
                        if list(i) not in c2:
                            c1.append(list(i))
                        if list(set(yold) - set(tuple(i))) not in c1:
                            c2.append(list(set(yold) - set(tuple(i))))

                y = pd.DataFrame(yold)
                y.rename(columns = {y.columns[0] : "y"}, inplace= True)
                y["twoing"] = 0#creates twoing column

                twoing_value = []
                twoing_soglia = []
                twoing_varian = [] #either using this to determine best or deviance 

                if len(c1) > 2: #2 classes will cause node purity checker to proc.  
                    for i in range(len(c1)): #can make it a bit more efficient, by not including the remainders of other spits 
                        y["twoing"].loc[y["y"].isin(c1[i])] = "c1"
                        y["twoing"].loc[y["y"].isin(c2[i])] = "c2"
                        
                        self.y = y["twoing"] #continually changing the self.y object , do i need to use set 

                        try:
                            #new_df = pd.DataFrame({"Function":"growing_tree twoing b4 split", "Lines":1190, "Time": time.time() - self.start}, index = [0])
                            #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)

                            value,soglia,varian, stump = self.__node_search_split(node, max_k, combination_split, max_c) 
                            
                        except TypeError:
                            #print("TypeError [Twoing, pure node after new class assignment]")                    
                            if len(node.indexes) >= self.grow_rules['min_cases_parent']:
                                continue
                            else:
                                self.y = yold
                                return None
                        #new_df = pd.DataFrame({"Function":"growing_tree twoing after split", "Lines":1201, "Time": time.time() - self.start}, index = [0])
                        #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)

                        twoing_value.append(value)
                        twoing_soglia.append(soglia)
                        twoing_varian.append(varian)

                        nss_variables.append(self.nss_variables)
                        nss_splits.append(self.nss_splits)
                        nss_between_variance.append(self.nss_between_variance)
                        nss_stumps.append(self.nss_stumps)
                #if a split has been found
                if twoing_varian:
                    #evaluarion of best split from all splits 
                    best_index = twoing_varian.index(max(twoing_varian))
                    value = twoing_value[best_index]                            #gets all values back into a recognizable form 
                    soglia = twoing_soglia[best_index]
                    varian = twoing_varian[best_index]

                    #may need it later 
                    self.twoing_c1[node] = c1[best_index]
                    self.twoing_c2[node] = c2[best_index]
                    y["twoing"].loc[y["y"].isin(c1[best_index])] = "c1"
                    y["twoing"].loc[y["y"].isin(c2[best_index])] = "c2"
                    y= pd.DataFrame(y["twoing"]) 
                    y.rename(columns = {y.columns[0] : node.name}, inplace= True) #that way can find it based on the name of the ndoe of the split 
                    self.twoing_y = pd.concat([self.twoing_y, y])

                    self.y = yold #hopefully this works used set function, maybe needs it 
                    
                    #print("nss",len(nss_variables))

                    self.nss_variables =  list(itertools.chain.from_iterable(nss_variables))
                    self.nss_splits = list(itertools.chain.from_iterable(nss_splits))
                    self.nss_between_variance = list(itertools.chain.from_iterable(nss_between_variance))
                    self.nss_stumps = list(itertools.chain.from_iterable(nss_stumps))

                    self.competitor_splits(node)
                    if self.surrogate_split:
                        self.surrogate_splits(node)
                
                else:
                    self.y = yold
                    return None
                
            elif self.problem == "regression":
                
                yold = self.y
                y = pd.DataFrame(self.y[node.indexes], index = node.indexes) #hopefully no issues if it is passed as a dataframe
       
                y.rename(columns = {y.columns[0] : "y"}, inplace= True)
                y["twoing"] = 0

                distinct_values= [set(y["y"])]
                distinct_values = list(itertools.chain(*distinct_values)) #flattens, removed nesting

                twoing_value = []
                twoing_soglia = []
                twoing_varian = [] #either using this to determine best or deviance 

                #if len(c1) > 2: #2 classes will cause node purity checker to proc.  
                
                #t = time.time()
                #print("dist", len(distinct_values))

                for i in distinct_values: 
                    if len(set(y["y"].loc[y["y"]<= i])) > 2 and len(set( y["y"].loc[y["y"] > i] )) >2: #TODO may be >=2 
                        y["twoing"].loc[y["y"]<= i] = "c1"
                        y["twoing"].loc[y["y"] > i] = "c2"
                        self.y = y["twoing"] #continually changing the self.y object , do i need to use set 

                    else:
                        continue #next iteration
                
                    self.problem = "classifier" #changing to classifier after changing classes 
                    if self.method == "CART":
                        self.impurity_fn = "gini"
                    else:
                        self.impurity_fn = "tau"
                    try:
                        value,soglia,varian, stump = self.__node_search_split(node, max_k, combination_split, max_c) 

                    except TypeError:
                        #print("TypeError [Twoing, pure node after new class assignment]")                    
                        if len(node.indexes) >= self.grow_rules['min_cases_parent']:
                            continue
                        else:
                            self.y = yold
                            self.problem = "regression"
                            if self.method == "CART":
                                self.impurity_fn = "between_variance"
                            else:
                                self.impurity_fn = "pearson"
                            return None

                    twoing_value.append(value)
                    twoing_soglia.append(soglia)
                    twoing_varian.append(varian)

                    nss_variables.append(self.nss_variables)
                    nss_splits.append(self.nss_splits)
                    nss_between_variance.append(self.nss_between_variance)
                    nss_stumps.append(self.nss_stumps)

                if twoing_varian:
                #evaluarion of best split from all splits 
                    best_index = twoing_varian.index(max(twoing_varian))
                    value = twoing_value[best_index]                            #gets all values back into a recognizable form 
                    soglia = twoing_soglia[best_index]
                    varian = twoing_varian[best_index]
                    self.twoing_c1[node] = "<="+str(distinct_values[best_index])
                    self.twoing_c2[node] = ">"+str(distinct_values[best_index])
                    y["twoing"].loc[y["y"]<= distinct_values[best_index]] = "c1"
                    y["twoing"].loc[y["y"] > distinct_values[best_index]] = "c2"
                    y= pd.DataFrame(y["twoing"]) 
                    y.rename(columns = {y.columns[0] : node.name}, inplace= True) #that way can find it based on the name of the ndoe of the split 
                    self.twoing_y = pd.concat([self.twoing_y, y])
                    self.y = yold #hopefully this works used set function, maybe needs it 
                    
                    self.problem = "regression"
                    if self.method == "CART":
                        self.impurity_fn = "between_variance"
                    else:
                        self.impurity_fn = "pearson"

                    #flattens teh list of lists created above 
                    self.nss_variables = list(itertools.chain.from_iterable(nss_variables))
                    self.nss_splits = list(itertools.chain.from_iterable(nss_splits))
                    self.nss_between_variance = list(itertools.chain.from_iterable(nss_between_variance))
                    self.nss_stumps = list(itertools.chain.from_iterable(nss_stumps))

                    self.competitor_splits(node)
                    if self.surrogate_split:
                        self.surrogate_splits(node)
                
                else:
                    self.y = yold
                    self.problem = "regression"
                    if self.method == "CART":
                        self.impurity_fn = "between_variance"
                    else:
                        self.impurity_fn = "pearson"
                    return None 

            else:
                print("Problem must either be classifier or regression")
                return None
        
        #normal tree growing protocol without twoing
        else:
            try:
                #self.start2 = time.time()
                #new_df = pd.DataFrame({"Function":"growing_tree b4 split", "Lines":1351, "Time": time.time() - self.start}, index = [0])
                #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)
                
                value,soglia,varian,stump = self.__node_search_split(node, max_k, combination_split, max_c)  
                #if self.method == "CART": #!= "LATENT-BUDGET-TREE" or self.method != "FAST" or self.method != "TWO-STAGE":
                self.competitor_splits(node)
                if self.surrogate_split:
                    self.surrogate_splits(node)

            except TypeError:
                if not self.catergorise_num1:
                    print("TypeError: Node search split (CART) failure")
                return None
            
            #if self.method == "LATENT-BUDGET-TREE":
                #varian = -varian #change put in place to worth with infracture, but want the correct ls value from lba to be printed 

        if not self.catergorise_num1:

            #new_df = pd.DataFrame({"Function":"growing_tree after split", "Lines":1362, "Time": time.time() - self.start}, index = [0])
            #self.time = pd.concat([self.time, new_df], ignore_index=True, sort=False)

            value_soglia_variance.append([value,soglia,varian,level])
            self.root.append((value_soglia_variance,rout))

        node.set_local_predictability(varian) #the max(between_variance) from best_split, local predictability using the the given impurity measure

        #recreate split from node search split 
        left_node,right_node = node.bin_split(self.features, self.n_features, str(value),soglia)
        #left_node,right_node  = stump[0],stump[1] #the last bin_split sets teh node.split variable so have to run though again at the end, not use pre-existing. 
                
        node.set_children((left_node,right_node))
        node.set_split(value_soglia_variance)
        mini_tree.append((node,left_node,right_node))

        if not self.catergorise_num1:
            self.tree.append(mini_tree) 
        
        if rout != 'start': 
            if not self.catergorise_num1:
                self.father.append(node) # may be redundant with the same appending happenign below
                    
        if rout == "start":
            if not self.catergorise_num1:
                self.bigtree.append(node)#append nodo padre
            else:
                self.catergorise_num_big.append(node)

        if not self.catergorise_num1:
            self.bigtree.append(left_node)#append nodo figlio sinistro
            self.bigtree.append(right_node)#append nodo figlio desto
            print("Split Found: ",node.name, value_soglia_variance,rout, node, node.split)

        ###### Calcolo della deviance nel nodo  
        if rout == 'start':
            if not self.catergorise_num1:
                self.father.append(node)
            else:
                self.catergorise_num_father.append(node)
                self.catergorise_num_big.append(left_node)
                self.catergorise_num_big.append(right_node)
            if self.problem=='regression':
                left_varian = len(self.y[left_node.indexes])*(mean(self.y[left_node.indexes])-mean(self.y))**2
                #right_varian = self.RSS(self.y[right_node.indexes])
                right_varian = len(self.y[right_node.indexes])*(mean(self.y[right_node.indexes])-mean(self.y))**2
                #left_varian = self.RSS(self.y[left_node.indexes])
                ex_deviance = (right_varian + left_varian) #- len(self.y)*mean(self.y)**2 
            
            elif self.problem == "classifier":
                ex_deviance = self.deviance_cat(left_node)*len(left_node.indexes)/len(self.y) + self.deviance_cat(right_node)*len(right_node.indexes)/len(self.y)# )/2
                          
        else:
            ex_deviance_list= []
            if not self.catergorise_num1:
                for inode in self.bigtree:
                    if inode not in self.father:
                        if self.problem == 'regression':
                            ex_deviance_list.append(len(self.y[inode.indexes])*(mean(self.y[inode.indexes])-mean(self.y))**2) #dont know why this formula 
                            #ex_deviance_list.append(mean(self.y[node.indexes])**2)*len(self.y[node.indexes] ) 
                            #ex_deviance_list.append(self.RSS(self.y[inode.indexes]))
                        elif self.problem == 'classifier':
                            ex_deviance_list.append(self.deviance_cat(inode)*len(inode.indexes)/len(self.y))
            else:
                for inode in self.catergorise_num_big:
                    if inode not in self.catergorise_num_father:
                        ex_deviance_list.append(len(self.y[inode.indexes])*(mean(self.y[inode.indexes])-mean(self.y))**2) 

                ex_deviance = sum(ex_deviance_list)
                node_proportion_total = ex_deviance/ self.devian_y
                self.catergorise_num_np.append(node_proportion_total)
                
                if len(self.catergorise_num_np)>1:
                    delta = self.catergorise_num_np[-1] - self.catergorise_num_np[-2]
                    if delta < self.grow_rules['min_imp_gain'] :
                        return None

                # eta stopping rule 
                for inode in self.catergorise_num_big:
                    wss = 0 
                    if inode not in self.catergorise_num_father:
                        #this will analyse terminal nodes only (aka the classes)
                        mean_y = mean(self.y[inode.indexes]) 
                        for j in inode.indexes: 
                            wss += (self.y[j] - mean_y)**2
                        #print("hwijwi",inode.name, wss, self.categorise_num_tss)
                eta2 = 1-wss/ self.categorise_num_tss
                #print(eta2)
                if eta2 > 0.7: #arbitary number 0.7
                    #print("hi :)", self.features_names, len(self.catergorise_num_big)-len(self.catergorise_num_father))
                    return None #stop growing else continue

                self.catergorise_num_big.append(left_node)
                self.catergorise_num_big.append(right_node)
                self.catergorise_num_father.append(node)
            
            if self.problem == "classifier":
                ex_deviance = sum(ex_deviance_list) / len(ex_deviance_list)
            else:
                ex_deviance = sum(ex_deviance_list)
        
        if self.problem == "classifier":
            node_proportion_total = self.devian_y - ex_deviance
        else:
            node_proportion_total = ex_deviance/ self.devian_y   

        if not self.catergorise_num1:
            print("node_proportion_total ",node_proportion_total)
            self.node_prop_list.append(node_proportion_total)

            node.set_node_prop(node_proportion_total) #attaches the node propotion to the node. 

            if rout == "start":
                self.node_prop_dict[node] = node_proportion_total

            if self.problem == "regression":
                if len(self.node_prop_list)>1:
                    delta = self.node_prop_list[-1] - self.node_prop_list[-2]
                    print("Node_proportion_gain ",delta)
                    self.node_prop_dict[node] = delta
                    if delta < self.grow_rules['min_imp_gain'] :#all utente  :Controllo delle variazione nei nodi figli
                        print("This split isn't good now i cut it [reg delta]")
                        left_node.set_to_pop()
                        right_node.set_to_pop()
                        self.father_to_pop.append(node)
                        self.root.pop()
                        node.set_split2(None)
                        return None
            else:
                if len(self.node_prop_list)>1:
                    if self.impurity_fn == "entropy":
                        entropy_parent = self.impur(node)#stump[0].indexes + stump[1].indexes
                        delta = entropy_parent - ((len(left_node.indexes) / len(node.indexes)) * self.impur(left_node) + (len(right_node.indexes) / len(node.indexes)) * self.impur(right_node))
                    else:
                        #delta = +self.deviance_cat(node) - (self.deviance_cat(right_node) + self.deviance_cat(left_node)) #looks like change in entropy 
                        delta = self.node_prop_list[-1] - self.node_prop_list[-2]
                    print("Node_proportion_gain ",delta)
                    self.node_prop_dict[node] = delta
                    
                    if delta < self.grow_rules['min_imp_gain'] :#all utente  :Controllo delle variazione nei nodi figli #get rid of abs(delta)
                        print("This split isn't good now i cut it [cat delta]")
                        left_node.set_to_pop()
                        right_node.set_to_pop()
                        self.father_to_pop.append(node)
                        self.root.pop()
                        node.set_split2(None)
                        return None

            #if ex_deviance/ self.devian_y  >= propotion_total:
            if node_proportion_total >= propotion_total: 

                print("Not sure if used: ex_deviance/devian_y is greater than set value proportion total (0.9): return none")
                return None
        
            #else: #looks redundant
                #if node_proportion_total >= propotion_total: 
                #   return None
        
            self.nsplit += 1

        return self.growing_tree(left_node,"left",max_k = max_k, combination_split = combination_split, max_c = max_c),self.growing_tree(right_node,"right",max_k = max_k, combination_split = combination_split, max_c = max_c)

    def merge_leaves(self, all_node = None, leaves = None):
        '''merges leaves for classification trees that have the same class distinction in both, i.e. undoes the spilt'''
        
        if not all_node:
            all_node = self.get_all_node().copy()
        if not leaves:
            leaves = self.get_leaf().copy()
        
        new_dict = self.identify_subtrees(all_node, leaves)
                
        for i in new_dict:
            if len(new_dict[i][1]) == 2:
                #print(i.name)
                responses = []
                for j in new_dict[i][1]:
                    responses.append(j.get_value(self.y, self.problem))
                    
                if responses[0] == responses[1]:
                    all_node.remove(new_dict[i][1][0]) #removign nodes from all
                    all_node.remove(new_dict[i][1][1])
                    leaves.remove(new_dict[i][1][0])  #removing nodes from lead
                    leaves.remove(new_dict[i][1][1])
                    leaves.append(i)                  #adding parent to leaves

                    #new_dict[i][1][0].set_to_pop()
                    #new_dict[i][1][1].set_to_pop()

        while len(new_dict) > len(self.identify_subtrees(all_node, leaves)):
            new_dict = self.identify_subtrees(all_node, leaves)
            for i in new_dict:
                if len(new_dict[i][1]) == 2:
                    responses = []
                    for j in new_dict[i][1]:
                        responses.append(j.get_value(self.y, self.problem))
                    if responses[0] == responses[1]:
                        all_node.remove(new_dict[i][1][0]) 
                        all_node.remove(new_dict[i][1][1])
                        leaves.remove(new_dict[i][1][0])  
                        leaves.remove(new_dict[i][1][1])
                        leaves.append(i) 

        return all_node, leaves

    def get_key(self, my_dict, val):
        '''A function for dictionaries, returning key from value'''
        for key, value in my_dict.items():
            if val == value:
                return key
        return "key doesn't exist"

    def identify_subtrees(self, father, leaves):
        '''Will associate each node with it's children, grandchildren etc., thus creating subtrees for each node, as if the node was the root
           returns two lists for each parent node, the first list has the all nodes node elements, the second the leaf elements'''
        
        all_nodes_dict = {}
        all_nodes_list =[]
        relative_dict={}

        for node in father:                                 
            all_nodes_dict[node] = int(node.name[1:])      #Creating a dictionary for each node as a key with their node number as the value
            all_nodes_list.append(int(node.name[1:]))      #Creating a list of all node numbers 
               
        for node in father:                             #Iterating though all nodes that have children and have the ability to have a subtree. 
            level = int(node.node_level)                #Using the level for the while loop, ensuring a stopping element, that makes sense as you progress down the tree to the leaves 
            
            if (int(node.name[1:]) *2) in all_nodes_list:  #Using the property of node numbers being related to their parents, in this case assessing the left child, which as a node number twice that of the parent 
                if node.name in relative_dict:
                    relative_dict[node].append(node.get_name_as_number()*2) #adding multiple value to a dictionary key
                else:
                    relative_dict[node] = [node.get_name_as_number()  *2] #adding the first value to a dictionary key
            if (int(node.name[1:])*2+1) in all_nodes_list:     #Same as above but assessing for the right node, which is twice the parents node number +1
                if node in relative_dict:           
                    relative_dict[node].append(node.get_name_as_number()*2+1)
                else:
                    relative_dict[node] = [node.get_name_as_number()*2+1]      
            while level > -1 and node in relative_dict: #-1 was use for the while loop, as the root node exists at level 0
                level += -1                             
                for child in relative_dict[node]:       #Allows the continual adding of children to the subtree, based on the node numbers within the dictionary. 
                                
                    if child*2 in all_nodes_list and child*2 not in relative_dict[node]:
                        if node in relative_dict:
                            relative_dict[node].append(int(child)*2)
                        else:
                            relative_dict[node] = [int(node.name[1:])*2]                        
                    if child*2+1 in all_nodes_list and child*2+1 not in relative_dict[node]:
                        if node in relative_dict:
                            relative_dict[node].append(int(child)*2+1)                  
                        else:
                            relative_dict[node].append(int((node.name[1:]))*2+1)
       
        only_leaves_dictionary ={}
        for element in relative_dict:
            for child in relative_dict[element]:
                if self.get_key(all_nodes_dict, child) in leaves:
                    if element in only_leaves_dictionary:
                        only_leaves_dictionary[element].append(child)
                    else:
                        only_leaves_dictionary[element] =[child]
        
        new_dict = {}   
        for key in relative_dict: #only_leaves_dictionary:
            node = []
            for i in relative_dict[key]: # only_leaves_dictionary[key]:
                for j in father:# self.get_all_node():
                    if i == int(j.name[1:]): #get_name_as_number():
                        node.append(j)
            node2 =[]
            for i in only_leaves_dictionary[key]:
                for j in father:# self.get_all_node():
                    if i == int(j.name[1:]): #get_name_as_number():
                        node2.append(j)            
            node3 = []
            node3.append(node2)
            node3.append(node)                          #when pruning need to ensure both parent and children nodes in subtree are removed
            new_dict.update({key:node3})
        
        return(new_dict)        
    
    def print_alpha(self,alpha):
        '''
        chiamare questa funzione dopo aver effettuato il calcolo degli alpha.
        Stampa a schermo tutti gli alpha.
        Prints the alpha values. 
        '''
        for i in alpha:
            print(i)    
    
    def pop_list(self,lista,lista_to_pop):
        '''funzione di pura utilità
        A utility function to remove entries from lists'''
        for i in lista_to_pop:
            lista.pop(lista.index(i))
        return lista

    def alpha_calculator(self,new_dict):
        '''
        Questa funzione ritorna il l'alpha minimo calcolato su un albero di classificazione o regressione,
        il parametro problem : stabilisce il tipo di problema
        valori accettai sono (regression,classification)
        This function returns the minimum alpha value calculated for the classification or regression trees, used for pruning the tree 
        '''
        
        alpha_tmp = []
        deviance = []
        
        if self.problem == 'regression':
            for key in new_dict: #key  padre
                rt_children__ = []

                #if isinstance(feat, np.ndarray): #numpy input

                rt_father= sum((self.y[key.indexes] - mean(self.y[key.indexes]))**2)
                for figli in new_dict[key][0]:
                    rt_children__.append(sum((self.y[figli.indexes] - mean(self.y[figli.indexes]))**2))
                    deviance.append(sum((self.y[figli.indexes] - mean(self.y[figli.indexes]))**2)) #added a sum here
                rt_children = sum(rt_children__)
                deviance_tot = sum(deviance)
                denom = (len(new_dict[key][0])-1)
                alpha_par = (-rt_children + rt_father)/denom         
                alpha_tmp.append((alpha_par,key,deviance_tot))
        
        elif self.problem == 'classifier':   
            for key in new_dict: #key  padre
                c = Counter(self.y[key.indexes])
                p = c.most_common(1)
                c = len(self.y[key.indexes])-p[0][1]
                rt_father = c
                rt_children = 0
                for figli in new_dict[key][0]:
                    c = Counter(self.y[figli.indexes])
                    p = c.most_common(1)
                    c = len(self.y[figli.indexes])-p[0][1]
                    rt_children += c
                    
                denom = (len(new_dict[key][0])-1)
                
                if(denom <= 0):
                        denom = 0.000000001
                alpha_par = (-rt_children + rt_father)/denom
                alpha_tmp.append((alpha_par,key))
        else:
            print("error")
            exit(1)
        if len(alpha_tmp)<=1:
            alpha_tmp.append((0,None))
        return min(alpha_tmp,key=lambda l:l[0]) #alphamin
    
    def set_new_all_node(self,lista):
        '''
        Funzione di utilità richiamata dopo il cut
        per ridurre la dimensione dell'albero in termini della quantitò di nodi utilizzati
        resets the leaf list with those from from the pruned tree
        '''
        self.leaf = lista
    
    
    def set_new_leaf(self,lista):
        '''
        Funzione di utilità richiamata dopo il cut
        per ridurre la dimensione dell'albero in termini della quantitò di nodi utilizzati
        come nodi foglia.
        resets the all_node list with those from from the pruned tree

        '''
        self.all_node = lista
    

    def miss_classifications(self,list_node):
        '''Calculates the errors as mse or missclassifcation for aid when evaluating trees during pruning'''
        if self.problem == "classifier":
            
            errors = 0
            for node in list_node:
                errors += len(self.y[node.indexes])-Counter(self.y[node.indexes]).most_common(1)[0][1] #works
                #for val in self.y[node.indexes]:
                #    if Counter(self.y[node.indexes]).most_common(1)[0][0] != val:
                #        s +=1
           
        elif self.problem == "regression":
            errors = 0
            comparison = []
            for node in list_node:
                #s += (mean(self.y[i.indexes])**2)*len(self.y[i.indexes]) #will need changing 
                mean_y = mean(self.y[node.indexes])
                for val in self.y[node.indexes]:
                    errors+= (val - mean_y)**2
                    comparison.append([val, mean_y])
            #print("c1",comparison, "s", s, s/len(self.y))
            errors = errors/len(self.y)
        return errors
            
        
    def pruning(self, features_test, n_features_test, y_test, 
                png_name = "TREE4_tree_pruned.png", 
                dot_name = "tree_pruned.dot", 
                table = False, html = False, print_render = False, merge_leaves = False, 
                graph_results = False, print_tree = False, visual_pruning = False):
        '''
        call this function after the growing tree
        perform the pruning of the tree based on the alpha value
        Alfa = #########
        
        per ogni nodo prendi ogni finale prendi i suoi genitori verifica il livello  se è il massimo prendi i genitori

        performs the iterative pruning operation
        
        '''
        #TODO keep images assigned to the class: https://stackoverflow.com/questions/53438133/using-an-image-as-an-class-object-attribute-then-opening-that-image-in-a-tkinte
        #start = time.time()
        
        if table == True:
            if print_tree != True:
                print("To return the table, print_tree must be True: setting print_tree to True")
                print_tree = True

        all_node = self.get_all_node().copy()
        leaves = self.get_leaf().copy()

        alpha=[]  #(alpha,node) lista degli alpha minimi
        miss =[]
        leaves_for_prune = []
        leaves_mse = {}
        leaves_miss ={}
        result = []
        train_miss = {}

        #Creating alpha value for full tree
        alpha.append((0, None))
        leaves_for_prune.append(len(leaves))
        miss.append(self.miss_classifications(leaves))    #appends count of total obs that are not in the majority class of the leaf    
        if self.problem =="classifier":
            result.append((f"Alpha = {alpha[0][0]}",f"value soglia = {alpha[0][1]}",f"misclassification = {miss[0]}",f"leaves = {leaves_for_prune[0]}"))
        else:
            result.append((f"Alpha = {alpha[0][0]}",f"value soglia = {alpha[0][1]}",f"deviance = {miss[0]}",f"leaves = {leaves_for_prune[0]}"))

        #Running through original prediction for full tree 
        mse = 0
        miss_val = 0
        if self.problem =='regression':
            for i in range(len(y_test)):      #iterates through number of rows in n_feature_test 
                for node in all_node:
                    if node.name =="n1":           
                        new = []
                        new_n = []            
                        for name in self.features_names:
                            new.append(features_test[name][i])
                        for n_name in self.n_features_names:
                            new_n.append(n_features_test[n_name][i])        

                        d = dict(zip(self.features_names, new))
                        dn = dict(zip(self.n_features_names, new_n))
                        d.update(dn)
                        self.pred_x(node, d, all_node, leaves)

                        mse += (y_test[i] - self.prediction_reg[-1])**2
            leaves_mse[leaves_for_prune[-1]] = mse/len(y_test)
        
        #Classification
        else:
            for i in range(len(y_test)):       
                for node in all_node:
                    if node.name =="n1":           
                        new = []
                        new_n = []            
                        for name in self.features_names:
                            new.append(features_test[name][i])
                        for n_name in self.n_features_names:
                            new_n.append(n_features_test[n_name][i])

                        d = dict(zip(self.features_names, new))
                        dn = dict(zip(self.n_features_names, new_n))
                        d.update(dn)
                        self.pred_x(node, d, all_node, leaves)  
                        if y_test[i] != self.prediction_cat[-1]:
                            miss_val +=1
            leaves_miss[leaves_for_prune[-1]] = miss_val
               
        #print("after test evaluation", time.time()-start)
        pruned_trees =[]
        pruned_trees.append([len(leaves), all_node.copy(), leaves.copy()]) #full tree
       
        #Start Pruning Process, continuing until root node
        while len(all_node) >=3: #have changed this to 1 without just leaving root node, could just append to end of list if wanted 
            
            new_dict = self.identify_subtrees(all_node,leaves)
            cut = self.alpha_calculator(new_dict)
            alpha.append(cut)  #(alpha,node)
            
            if(cut[1])==None:
                break

            all_node = self.pop_list(all_node, lista_to_pop = new_dict[cut[1]][1]) #pop on all node
            leaves = self.pop_list(leaves, lista_to_pop = new_dict[cut[1]][0]) #pop on leaf
            leaves.append(cut[1])
            miss.append(self.miss_classifications(leaves))
            leaves_for_prune.append(len(leaves))
            pruned_trees.append([len(leaves), all_node.copy(), leaves.copy()])

            mse = 0
            miss_val = 0

            #print("during pruning iteration, before evaluation", time.time()-start )
            if self.problem =='regression':
                for i in range(len(y_test)):      #iterates through number of rows in n_feature_test 
                    for node in all_node:
                        if node.name =="n1":           
                            new = []
                            new_n = []            
                            for name in self.features_names:
                                new.append(features_test[name][i])
                            for n_name in self.n_features_names:
                                new_n.append(n_features_test[n_name][i])

                            d = dict(zip(self.features_names, new))
                            dn = dict(zip(self.n_features_names, new_n))
                            d.update(dn)
                            self.pred_x(node, d, all_node, leaves)

                            mse += (y_test[i] - self.prediction_reg[-1])**2
                leaves_mse[leaves_for_prune[-1]] = mse/len(y_test)
            
                #print("during pruning iteration, after reg evaluation", time.time()-start )

            else:
                #missclass1 = 0   #this is just set up for printing for the 4 class lbt problem 
                #class1 = 0
                #missclass2 = 0 
                #class2 = 0
                #missclass3 = 0 
                #class3 = 0
                #missclass4 = 0 
                #class4 = 0

                for i in range(len(y_test)):       
                    for node in all_node:
                        if node.name =="n1":           
                            new = []
                            new_n = []            
                            for name in self.features_names:
                                new.append(features_test[name][i])
                            for n_name in self.n_features_names:
                                new_n.append(n_features_test[n_name][i])

                            d = dict(zip(self.features_names, new))
                            dn = dict(zip(self.n_features_names, new_n))
                            d.update(dn)
                            
                            self.pred_x(node, d, all_node, leaves)                    

                            if y_test[i] != self.prediction_cat[-1]:
                                miss_val +=1

                            #if y_test[i] ==1:
                            #    class1 += 1
                            #    if self.prediction_cat[-1] != y_test[i]:
                            #        missclass1 += 1 
                            #if y_test[i] ==2:
                            #    class2 += 1
                            #    if self.prediction_cat[-1] != y_test[i]:
                            #        missclass2 += 1 
                            #if y_test[i] ==3:
                            #    class3 += 1
                            #    if self.prediction_cat[-1] != y_test[i]:
                            #        missclass3 += 1 
                            #if y_test[i] ==4:
                            #    class4 += 1
                            #    if self.prediction_cat[-1] != y_test[i]:
                            #        missclass4 += 1 
                #print("during pruning iteration, after cat evaluation", time.time()-start )

                leaves_miss[leaves_for_prune[-1]] = miss_val   
                #print("Accuracy", round((1-missclass1 / class1) * 100, 2), round((1-missclass2 / class2) * 100, 2), round((1-missclass3 / class3) * 100, 2), round((1-missclass4 / class4 )* 100, 2))
                pred_node_dict = {}
                count = 0
                for pred_node in self.pred_node[-len(y_test):]:
                    count += 1
                    if pred_node in pred_node_dict:
                        pred_node_dict[pred_node] +=1
                    else:
                        pred_node_dict[pred_node] = 1
            #print("after 1 pruning iteration", time.time()-start )

        if self.problem =='regression':
            #print("{leaves : mean square error} = ", leaves_mse)
            minimum = 100000
            key_min = 100000
            for key in leaves_mse:
                if leaves_mse[key] <= minimum:
                    if key < key_min:
                        minimum = leaves_mse[key]
                        key_min = key

            print(f"Best tree for test set has {key_min} leaves with a deviance of: {minimum} ")
            if graph_results:
                self.graph_results(leaves_for_prune,miss,"Training Set", list(leaves_mse.keys()),list(leaves_mse.values()),"Testing Set")
            
            if print_tree:
                for i in pruned_trees:
                    if i[0] == key_min:
                        tree_table = self.print_tree(i[1], i[2], png_name,dot_name, table = table, html = html, print_render= print_render, merge_leaves = merge_leaves, visual_pruning = visual_pruning )

        else:
            #print("{leaves : misclassification count} = ", leaves_miss)
            minimum = 10000
            key_min = 10000 
            for key in leaves_miss:
                if leaves_miss[key] <= minimum:
                    if key < key_min:
                        minimum = leaves_miss[key]
                        key_min = key

            print(f"Best tree for test set has {key_min} leaves with misclassification count {minimum} ") 
            misstrain = [i/ len(self.y) for i in miss] 
            misstest = [i/len(y_test) for i in list(leaves_miss.values())]         
            if graph_results:
                self.graph_results(leaves_for_prune,misstrain,"Training Set", list(leaves_miss.keys()),misstest,"Testing Set") #x1, y1, label1, x2, y2, label2 #list(leaves_miss.values())/len(y_test)

            #leaves for prune - amount of leaves at different cuts
            #miss is values that arent main *** wrong


            #print tree for minkey, and get resulting table
            if print_tree:
                for i in pruned_trees:
                    if i[0] == key_min:
                        tree_table = self.print_tree(i[1], i[2], png_name, dot_name, table = table, html = html, print_render= print_render, merge_leaves = merge_leaves, visual_pruning = visual_pruning)
        
        #make alpha lists
        if self.problem =="classifier":
            for i in range(len(alpha)):
                if alpha[i][1]!=None:
                    result.append((f"Alpha = {alpha[i][0]}",f"value soglia = {alpha[i][1].get_value_thresh()}",f"misclassification = {miss[i]}",f"leaves = {leaves_for_prune[i]}"))
        else:
            for i in range(len(alpha)):
                if alpha[i][1]!=None:
                    result.append((f"Alpha = {alpha[i][0]}",f"value soglia = {alpha[i][1].get_value_thresh()}",f"deviance = {miss[i]}",f"leaves = {leaves_for_prune[i]}"))
        

        return result, tree_table
    
    def cut_tree(self,total_leaves:int):
        '''For cutting tree to wanted size'''
        #Doesn't affect the right lists when popping, could be as easy as updating those lists
        #can also export the adjusted lists and use in conjunction with print_tree

        if total_leaves>len(self.get_leaf())-1:
            print("error on cut")
            exit(1)
        
        all_node = self.get_all_node()
        leaves = self.get_leaf()
        
        alpha=[]  #(alpha,node) lista degli alpha minimi
        
        while len(self.leaf) > total_leaves: #was != 
               
            new_dict = self.identify_subtrees(all_node,leaves)
            
            cut = self.alpha_calculator(new_dict)
            alpha.append(cut)  #(alpha,node)
            
            if cut[1] == None:
                break
            
            leaves = self.pop_list(leaves, lista_to_pop = new_dict[cut[1]][0]) #pop on leaf
            leaves.append(cut[1])
            self.leaf = leaves
            
            all_node = self.pop_list(all_node, lista_to_pop = new_dict[cut[1]][1]) #pop on all node
            self.all_node  = all_node
            
        return all_node, leaves

    def build_tree_recursively_pydot(self,nodenum, parent_node, parent_children, all_node,leaf_list, leaf_dict, graph, parent_node2):
        '''Creates a tree structire, placing the generated nodes from fit() into this required structure for printing'''
        
        for child in parent_children[nodenum]:          #iterating throught the values in the dictionary for the nodenum key
            for node2 in all_node:                      #Iterate through the all node dictionary
                if int(node2.name[1:]) == child:        #Matched the node to that in the dictionary, in order to apply the lines data below, and applyign the corresponding value 
                    if child not in leaf_list:
                        if self.impurity_fn =="gini":
                            child_node = pydot.Node(int(node2.name[1:]), label = f"{node2.split}\n{self.impurity_fn}: {round(self.impur(node2, display = True),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, node2.split])    #creates the new child node, if not a terminal node, to show the split information in "lines"
                            graph.add_node(child_node)
                            graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))                           
                            
                            child_node2 = Node([str(child),node2.name, node2.split, round(self.impur(node2, display = True),2)], parent=parent_node2, lines =[node2.name, node2.split, round(1-self.impur(node2)/len(node2.indexes),2)])                        
                        else:
                            child_node = pydot.Node(int(node2.name[1:]), label = f"{node2.split}\n{self.impurity_fn}: {round(self.impur(node2),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, node2.split])    #creates the new child node, if not a terminal node, to show the split information in "lines"
                            graph.add_node(child_node)
                            graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))

                            child_node2 = Node([str(child),node2.name, node2.split, round(self.impur(node2),2)], parent=parent_node2, lines =[node2.name, node2.split, round(self.impur(node2),2)])                        

                    else:                     
                        if self.problem == "classifier":        #For classifier problem
                            count_y = 0
                            response_dict ={}
                            for response in self.y[(self.get_key(leaf_dict,child)).indexes]:        #determing majority in terminal nodes
                                
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1

                            class_node = max(response_dict, key = response_dict.get)
                            if self.impurity_fn =="gini":
                                child_node = pydot.Node(int(node2.name[1:]), label = f"Class: {class_node}\n{self.impurity_fn}: {round(self.impur(node2, display = True),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, class_node]) #creates a new child with th lines set to the class of the node
                                graph.add_node(child_node)
                                graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))
                                
                                child_node2 = Node([str(child),node2.name, class_node, round(self.impur(node2, display = True),2)], parent=parent_node2, lines =[node2.name, class_node, round(1-self.impur(node2)/len(node2.indexes),2)])                            
                            else:
                                child_node = pydot.Node(int(node2.name[1:]), label = f"Class: {class_node}\n{self.impurity_fn}: {round(self.impur(node2),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, class_node]) #creates a new child with th lines set to the class of the node
                                graph.add_node(child_node)
                                graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))

                                child_node2 = Node([str(child),node2.name, class_node, round(self.impur(node2),2)], parent=parent_node2, lines =[node2.name, class_node, round(self.impur(node2),2)])                    

                        else:
                            mean_y = mean(self.y[(self.get_key(leaf_dict,child)).indexes])

                            child_node = pydot.Node(int(node2.name[1:]), label = f"Bin Value: {round(mean_y,2)}\n{self.impurity_fn}: {round(self.impur(node2),2)}\nSamples: {len(node2.indexes)}")#, parent=parent_node)#, lines =[node2.name, round(mean_y,2)]) #creates a new child node, when it is a terminal node, so instead present the mean of the y values in the node
                            graph.add_node(child_node)
                            graph.add_edge(pydot.Edge(parent_node, child_node, color="black"))
                            
                            child_node2 = Node([str(child), node2.name, round(mean_y,2)], parent=parent_node2, lines =[node2.name, round(mean_y,2)]) #creates a new child node, when it is a terminal node, so instead present the mean of the y values in the node

            if child in parent_children:            #Continues the growing only if the child has a key value in parent_children, and therefore has children
                self.build_tree_recursively_pydot(child, child_node, parent_children,all_node,leaf_list, leaf_dict, graph, child_node2)

    def build_tree_recursively_render(self,nodenum, parent_node, parent_children, all_node,leaf_list, leaf_dict):
        '''Creates a tree structire, placing the generated nodes from growing_tree() into this required structure for printing'''
        
        for child in parent_children[nodenum]:          #iterating throught the values in the dictionary for the nodenum key
            for node2 in all_node:                      #Iterate through the all node dictionary
                if int(node2.name[1:]) == child:        #Matched the node to that in the dictionary, in order to apply the lines data below, and applyign the corresponding value 
                    if child not in leaf_list:
                        child_node = Node(str(child), parent=parent_node, lines =[node2.split])    #creates the new child node, if not a terminal node, to show the split information in "lines"
                    else:                     
                        if self.problem == "classifier":        #For classifier problem
                            count_y = 0
                            for response in self.y[(self.get_key(leaf_dict,child)).indexes]:        #determing majority in terminal nodes
                                response_dict ={}
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1
                            class_node = max(response_dict, key = response_dict.get)
                            child_node = Node(str(child), parent=parent_node, lines =[class_node]) #creates a new child with th lines set to the class of the node
                        else:
                            mean_y = mean(self.y[(self.get_key(leaf_dict,child)).indexes])
                            child_node = Node(str(child), parent=parent_node, lines =[round(mean_y,2)]) #creates a new child node, when it is a terminal node, so instead present the mean of the y values in the node

            if child in parent_children:            #Continues the growing only if the child has a key value in parent_children, and therefore has children
                self.build_tree_recursively_render(child, child_node, parent_children,all_node,leaf_list, leaf_dict)
        
    def print_tree(self, all_node = None,leaf= None, filename="TREE4_tree.png", treefile = "tree.dot", table = False, html = False, print_render = False, visual_pruning = False, merge_leaves = False):
        '''Print a visual representation of the formed tree, showing splits at different branches and the mean of the leaves/ terminal nodes.'''
        start = time.time()
        if not all_node:
            all_node = self.get_all_node()
        if not leaf:
            leaf = self.get_leaf()

        if merge_leaves:
            all_node, leaf = self.merge_leaves(all_node, leaf)

        leaf_list =[]
        leaf_dict ={}
        for node in leaf:                           #creates a list of the node numbers and a dictionary connecting nodes with their node numbers
            leaf_list.append(int(node.name[1:]))
            leaf_dict[node] = int(node.name[1:])
        father_list =[]
        father_dict = {}
        for node in all_node:
            father_list.append(int(node.name[1:]))
            father_dict[node] = int(node.name[1:])

        
        parent_child =[]                            #list for having child with their parent, for use in dictionary below
        for node in all_node:
            if (int(node.name[1:]) *2) in father_list:
            
                parent_child.append([int(node.name[1:]), int(node.name[1:])*2])
            if (int(node.name[1:])*2+1) in father_list:
            
                parent_child.append([int(node.name[1:]), int(node.name[1:])*2+1])   

        parent_children = {}                        #dictionary for parents with children, only numbers
        for parent, child in parent_child: 
            if parent in parent_children:
                parent_children[parent].append(child)
            else:
                parent_children[parent] = [child]

        '''
        #pydot plot
        node_num = 1                            #The first node
        for node in all_node:
            if node.name =="n1":                #ensuring to start at "n1"
                
                graph = pydot.Dot("my_graph", graph_type="digraph", dir="forward", shape="ellipse", spines = "line")

                if self.impurity_fn =="gini":
                    tree = pydot.Node (int(node.name[1:]),  label =f"{node.split}\n{self.impurity_fn} : {round(self.impur(node, display = True),2)}\nSamples : {len(node.indexes)}" )#, lines =[node.name, node.split])         #creates root node
                    tree2 = Node([str(node_num), node.split, round(self.impur(node, display = True),2)], lines =[node.name, node.split])         #creates root node
                
                else:
                    tree = pydot.Node (int(node.name[1:]),  label =f"{node.split}\n{self.impurity_fn} : {round(self.impur(node),2)}\nSamples : {len(node.indexes)}" )#, lines =[node.name, node.split])         #creates root node
                    tree2 = Node([str(node_num), node.split, round(self.impur(node),2) ], lines =[node.name, node.split])         #creates root node

                graph.add_node(tree)
                self.build_tree_recursively_pydot(node_num, tree, parent_children,all_node,leaf_list, leaf_dict, graph, tree2) #starts applying parent and child names to respective instances

        #Dot exporter and dot to png
        try:                              
            DotExporter(tree2).to_dotfile(treefile)   #was tree
            graph.write_png(filename) 
        except: 
            DotExporter(tree2).to_dotfile(treefile)
        '''

        #Old print method
        if print_render:
            node_num = 1                            #The first node
            for node in all_node:
                if node.name =="n1":                #ensuring to start at "n1"
                    tree = Node(str(node_num), lines =[node.split])         #creates root node
                    self.build_tree_recursively_render(node_num, tree, parent_children,all_node,leaf_list, leaf_dict) #starts applying parent and child names to respective instances


            for pre, fill, node in RenderTree(tree):                #renders the tree for printing using the RengerTree function from anytree
                print("{}{}".format(pre, node.lines[0]))
                for line in node.lines[1:]:
                    print("{}{}".format(fill, line)) 

        #igraph Graph
        
        nr_vertices = max(father_list)                            # make too many to allow for missing nodes
        v_label = list(map(str, father_list) )                      # create node labels 
        G = Graph.Tree(nr_vertices, 2)                              # 2 stands for children number
        lay = G.layout_reingold_tilford(root=[0])
        position = {k: lay[k-1] for k in father_list}               # assigning nodes to positions , using reigngold layout
        
        
        
        #visual prunign 2nd attempt
        if visual_pruning:   #problem if the split with the highest purity gain is not the first, aka, lbt
          
            node_prop_gain = {}
            for i in self.node_prop_dict:
                node_prop_gain[int(i.name[1:])] = self.node_prop_dict[i]

            new_dict = self.identify_subtrees(all_node, leaf)# self.get_all_node(), self.get_leaf()) #careful with merge_leaves and visual pruning

            upward_tree = False
            if upward_tree:

                #attempt at upward tree 

                #resetting all the y positions to 0
                for i in position:
                    position[i] = [position[i][0],0]

                for i in position: #slightly longer list than new_dict
                    for j in new_dict:
                        if i == int(j.name[1:]):
                            for child in new_dict[j][1]:
                                position[int(child.name[1:])] = [position[int(child.name[1:])][0], position[int(child.name[1:])][1] + node_prop_gain[i]]           
            

            else:
                #attempt 2 at a downward tree 
                for i in position:
                    position[i] = [position[i][0],1]

                for i in position: #slightly longer list than new_dict
                    for j in new_dict:
                        if i == int(j.name[1:]):
                            for child in new_dict[j][1]:
                                position[int(child.name[1:])] = [position[int(child.name[1:])][0], position[int(child.name[1:])][1] - node_prop_gain[i]]   

            self.node_gain = node_prop_gain

        #updates position of nodes eitehr from visual pruning, or standard layout
        self.position = position
        
        #preparing layout for plotly tree
        Y = [lay[k][1] for k in range(len(father_list))] #will need actioning for list 
        M = max(Y)
        es = EdgeSeq(G)                                             # sequence of edges
        E = [e.tuple for e in G.es] # list of edges, connects nodes
        L = len(position)
        Xn = [position[k][0] for k in father_list]
        Yn = [2*M-position[k][1] for k in father_list]
        if visual_pruning:
            Yn = [position[k][1] for k in father_list]
        a = 0
        while a<20:                                                 # When the value is removed it skips to the next index value, jumping, a<10 is just overkill, increased to 20, for really narrow branches 
            for edge in E:   #this is meant to catch the mismateched E's 
                if edge[0] +1 not in position or edge[1]+1 not in position:
                    E.remove(edge) 
            a+=1
        Xe = []
        Ye = []
        for edge in E: 
            Xe+=[position[edge[0]+1][0],position[edge[1]+1][0], None]                   # edited for +1 poisiotn as the expected 0 root node it 1 in our dictionary, if index error, increase a 
            Ye+=[2*M-position[edge[0]+1][1],2*M-position[edge[1]+1][1], None]         
        #change labels here, edited to display more information than the node.name
        if visual_pruning:
            Ye = []
            for edge in E: 
                Ye+=[position[edge[0]+1][1],position[edge[1]+1][1], None]  

        #print("time before adding labels", time.time()- start)

        #applying labels to the nodes 
        for label in range(len(v_label)):
            for node in all_node:
                if v_label[label] == node.name[1:]:
                    if int(v_label[label]) in leaf_list:
                        if self.problem == "classifier":        #For classifier problem
                            response_dict ={}
                            for response in self.y[node.indexes]:        #determing majority in terminal nodes
                                
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1
                            
                            if self.method == "LATENT-BUDGET-TREE" or self.twoing: #multiclass methods
                                total_node_obs = sum(response_dict.values())
                                for key in response_dict:
                                    response_dict[key] = round(response_dict[key] / total_node_obs,2)

                                class_node = response_dict
                                myKeys = list(class_node.keys())
                                myKeys.sort()
                                class_node = {i: class_node[i] for i in myKeys}
                            else:
                                class_node = max(response_dict, key = response_dict.get)
                            
                            if self.impurity_fn == "gini":
                                v_label[label] = f"{node.name}<br>Class: {class_node}<br>{self.impurity_fn} : {round(self.impur(node, display = True),2)}<br>Samples : {len(node.indexes)}" 
                                
                            elif self.impurity_fn == "tau":
                                v_label[label] = f"{node.name}<br>Class: {class_node}<br>{self.impurity_fn} : None<br>Samples : {len(node.indexes)}<br>GPR: {node.global_predictability}<br>LPR: {node.local_predictability}" 
                            else:
                                v_label[label] = f"{node.name}<br>Class: {class_node}<br>{self.impurity_fn} : {round(self.impur(node),2)}<br>Samples : {len(node.indexes)}<br>GPR: {node.global_predictability}<br>LPR: {node.local_predictability}" 

                        else:
                            mean_y = mean(self.y[node.indexes])
                            if self.method == "CART":
                                v_label[label]=  f"{node.name}<br>{node.split}<br>Bin Value: {round(mean_y,2)}<br>{self.impurity_fn} : {round(self.impur(node),2)}<br>Samples : {len(node.indexes)}"
                            else:
                                v_label[label]=  f"{node.name}<br>{node.split}<br>Bin Value: {round(mean_y,2)}<br>{self.impurity_fn} : {round(self.impur(node),2)}<br>Samples : {len(node.indexes)}<br>GPR: {node.global_predictability}<br>LPR: {node.local_predictability}"
                    
                    #label for non leaves
                    else:
                        if self.problem == "classifier":
                            response_dict ={}
                            for response in self.y[node.indexes]:        #determing majority in terminal nodes
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1
                            
                            if self.method == "LATENT-BUDGET-TREE" or self.twoing: #multiclass methods
                                total_node_obs = sum(response_dict.values())
                                for key in response_dict:
                                    response_dict[key] = round(response_dict[key] / total_node_obs,2)

                                class_node = response_dict
                                myKeys = list(class_node.keys())
                                myKeys.sort()
                                class_node = {i: class_node[i] for i in myKeys}
                            else:
                                class_node = max(response_dict, key = response_dict.get)

                            if self.impurity_fn == "gini":
                                v_label[label] = f"{node.name}<br>{node.split}<br>Class:{class_node}<br>{self.impurity_fn} : {round(self.impur(node, display = True),2)}<br>Samples: {len(node.indexes)}"
                            elif self.impurity_fn == "tau":
                                v_label[label] =  f"{node.name}<br>{node.split}<br>Class:{class_node}<br>{self.impurity_fn} : {round(node.value_soglia_split[0][2],2)}<br>Samples: {len(node.indexes)}<br>GPR: {node.global_predictability}<br>LPR: {node.local_predictability}" 
                            else:
                                v_label[label] = f"{node.name}<br>{node.split}<br>Class:{class_node}<br>{self.impurity_fn} : {round(self.impur(node),2)}<br>Samples: {len(node.indexes)}<br>GPR: {node.global_predictability}<br>LPR: {node.local_predictability}"
                        else:
                            mean_y = mean(self.y[node.indexes])
                            if self.method == "CART":
                                v_label[label]=  f"{node.name}<br>{node.split}<br>Bin Value: {round(mean_y,2)}<br>{self.impurity_fn} : {round(self.impur(node),2)}<br>Samples : {len(node.indexes)}"
                            else:
                                v_label[label]=  f"{node.name}<br>{node.split}<br>Bin Value: {round(mean_y,2)}<br>{self.impurity_fn} : {round(self.impur(node),2)}<br>Samples : {len(node.indexes)}<br>GPR: {node.global_predictability}<br>LPR: {node.local_predictability}"
        #print("time after labels", time.time()-start)

        labels = v_label

        # Drawing using plotly library 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Xe,
                        y=Ye,
                        mode='lines',
                        line=dict(color='rgb(210,210,210)', width=1),
                        hoverinfo='none'
                        ))
        fig.add_trace(go.Scatter(x=Xn,
                        y=Yn,
                        mode='markers',
                        name='Nodes',
                        marker=dict(symbol='circle-dot',
                                        size=18,
                                        color='#6175c1',    #'#DB4551',
                                        line=dict(color='rgb(50,50,50)', width=1)
                                        ),
                        text=labels,
                        hoverinfo='text',
                        opacity=0.8, 
                        hoverlabel = dict(font_size = 20)

                        ))
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), 
            #showlegend = False, 
            #yaxis_title="Level", 
            plot_bgcolor='rgba(0, 0, 0, 0)',
            xaxis={'visible': False, 'showticklabels': False},    
            #    title=filename[:-4],    #chops off ".png"
            
            )
        

        if not visual_pruning:
            fig.update_layout(
                yaxis={'visible': False, 'showticklabels': True}
            )

        if visual_pruning:
            fig.update_layout(
                yaxis_title="Decrease in Variance (%)"
            )

        fig.show()
        if html:
            fig.write_html("TREE4_tree.html")
            webbrowser.open_new_tab("TREE4_tree.html")

        #print("after tree formed", time.time()- start)

        #startign to prepare for results table creation 
        for node in all_node:
            if self.problem == "regression":
                node.deviance = len(self.y[node.indexes])* self.RSS(self.y[node.indexes])
            else:
                node.deviance = self.deviance_cat2(node)

        if table == True and self.method == "LATENT-BUDGET-TREE": 
            if self.twoing:
                tree_table = pd.DataFrame(columns = ["Node", "Node Type", "Splitting Variable", "Twoing Classes C1", "Twoing Classes C2", "n", "Heterogeneity","Explained Heterogeneity","Class Probabilities", "Alpha","Beta" ])
            else:
                tree_table = pd.DataFrame(columns = ["Node", "Node Type", "Splitting Variable", "n", "Heterogeneity","Explained Heterogeneity", "Class Probabilities", "Alpha","Beta" ])
            n1node = self.get_key(father_dict, 1)
            n1index = all_node.index(n1node)

            for node in all_node[n1index:]:
                if int(node.name[1:]) not in leaf_dict.values():

                    if self.problem == "classifier":
                        response_dict ={}
                        for response in self.y[node.indexes]:        #determing majority in terminal nodes
                            if response in response_dict:
                                response_dict[response] +=1
                            else:
                                response_dict[response] =1
                        total_node_obs = sum(response_dict.values())
                        for key in response_dict:
                            response_dict[key] = round(response_dict[key] / total_node_obs,2)
                        class_node = response_dict
                        myKeys = list(class_node.keys())
                        myKeys.sort()
                        class_node = [{i: class_node[i] for i in myKeys}]
                    else:
                        class_node = sum(self.y[node.indexes])/len(self.y[node.indexes])

                    #if node.name == "n1":
                        #exp_dev = f"{round(node.node_prop,3)} of {round(self.devian_y,2)}"
                    #else:
                    exp_dev = f"{round(node.node_prop,3)}"
                    
                    if self.twoing:
                        if self.impurity_fn == "gini":
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split,"Twoing Classes C1":[self.twoing_c1[node]], "Twoing Classes C2": [self.twoing_c2[node]],"n":len(node.indexes), "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev, "Class Probabilities":class_node, "Alpha":[node.alpha], "Beta":[node.beta] })
                        elif self.impurity_fn == "tau":
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split,"Twoing Classes C1":[self.twoing_c1[node]], "Twoing Classes C2": [self.twoing_c2[node]],"n":len(node.indexes),  "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev, "Class Probabilities":class_node, "Alpha":[node.alpha], "Beta":[node.beta]})
                        else:
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split,"Twoing Classes C1":[self.twoing_c1[node]], "Twoing Classes C2": [self.twoing_c2[node]],"n":len(node.indexes),  "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev,"Class Probabilities":class_node,"Alpha":[node.alpha],"Beta":[node.beta]})

                    else:
                        if self.impurity_fn == "gini":
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split, "n":len(node.indexes), "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev, "Class Probabilities":class_node, "Alpha":[node.alpha], "Beta":[node.beta]})
                        elif self.impurity_fn == "tau":
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split,"n":len(node.indexes),  "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev,"Class Probabilities":class_node, "Alpha":[node.alpha], "Beta":[node.beta]})
                        else:
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split, "n":len(node.indexes),  "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev,"Class Probabilities":class_node,"Alpha":[node.alpha],"Beta":[node.beta]})
                    tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
                    node_num = int(node.name[1:])
                    if node_num *2 in leaf_dict.values():
                        
                        cnode = self.get_key(leaf_dict, node_num*2)
                        
                        if self.problem == "classifier":
                            response_dict ={}
                            for response in self.y[cnode.indexes]:        #determing majority in terminal cnodes
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1
                            
                            total_node_obs = sum(response_dict.values())
                            for key in response_dict:
                                response_dict[key] = round(response_dict[key] / total_node_obs,2)
                            class_node = response_dict
                            myKeys = list(class_node.keys())
                            myKeys.sort()
                            class_node = [{i: class_node[i] for i in myKeys}]
                        else:
                            class_node = sum(self.y[cnode.indexes])/len(self.y[cnode.indexes])
                        
                        if self.twoing:
                            if self.impurity_fn == "gini":
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta]})
                            elif self.impurity_fn == "tau": #tau is only logged if there is a split 
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2": None,"n":len(cnode.indexes), "Class Probabilities":class_node, "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Alpha":[cnode.alpha], "Beta":[cnode.beta]})
                            else:
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node,"Alpha":[cnode.alpha],"Beta":[cnode.beta]})

                        else:
                            if self.impurity_fn == "gini":
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child","Splitting Variable":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node, "Alpha":[cnode.alpha],"Beta":[cnode.beta]} )
                            elif self.impurity_fn == "tau":  
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"n":len(cnode.indexes), "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta]})
                            else:
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child","Splitting Variable":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node,"Alpha":[cnode.alpha], "Beta":[cnode.cbeta]})
                        tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
                    if node_num *2+1 in leaf_dict.values():
                        cnode = self.get_key(leaf_dict, node_num*2+1)
                        if self.problem == "classifier":
                            response_dict ={}
                            for response in self.y[cnode.indexes]:        #determing majority in terminal cnodes
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1
                            
                            total_node_obs = sum(response_dict.values())
                            for key in response_dict:
                                response_dict[key] = round(response_dict[key] / total_node_obs,2)
                            class_node = response_dict
                            myKeys = list(class_node.keys())
                            myKeys.sort()
                            class_node = [{i: class_node[i] for i in myKeys}]
                        else:
                            class_node = sum(self.y[cnode.indexes])/len(self.y[cnode.indexes])


                        if self.twoing:
                            if self.impurity_fn == "gini":
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta]})
                            elif self.impurity_fn == "tau": 
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2": None,"n":len(cnode.indexes), "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta]})
                            else:
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node,"Alpha":[cnode.alpha],"Beta":[cnode.beta]})
                        else:
                            if self.impurity_fn == "gini":
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child","Splitting Variable":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node, "Alpha":[cnode.alpha],"Beta":[cnode.beta]} )
                            elif self.impurity_fn == "tau":  
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"n":len(cnode.indexes), "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta]})                        
                            else:
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child","Splitting Variable":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class Probabilities":class_node, "Alpha":[cnode.alpha],"Beta":[cnode.beta], "LS Error":[cnode.error]} )
                        tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
                
            return tree_table
    

        if table == True:
            if self.twoing:
                tree_table = pd.DataFrame(columns = ["Node", "Node Type", "Splitting Variable", "Twoing Classes C1", "Twoing Classes C2", "n", "Heterogeneity", "Explained Heterogeneity","Class/Value"])
            else:
                tree_table = pd.DataFrame(columns = ["Node", "Node Type", "Splitting Variable", "n", "Heterogeneity", "Explained Heterogeneity", "Class/Value"])
            n1node = self.get_key(father_dict, 1)
            n1index = all_node.index(n1node)

            for node in all_node[n1index:]:
                if int(node.name[1:]) not in leaf_dict.values():
                    if self.problem == "regression":
                        class_node = f"{round(mean(self.y[node.indexes]),2)}"
                    else:
                        response_dict ={}
                        for response in self.y[node.indexes]:        #determing majority in terminal nodes
                            if response in response_dict:
                                response_dict[response] +=1
                            else:
                                response_dict[response] =1
                        
                        if self.twoing:
                            total_node_obs = sum(response_dict.values())
                            for key in response_dict:
                                response_dict[key] = round(response_dict[key] / total_node_obs,2)
                            class_node = response_dict
                            myKeys = list(class_node.keys())
                            myKeys.sort()
                            class_node = [{i: class_node[i] for i in myKeys}]
                        else:
                            class_node = max(response_dict, key = response_dict.get)
                    
                    #if node.name == "n1":
                    #    exp_dev = f"{round(node.node_prop,3)} of {round(self.devian_y,2)}"
                    #else:
                    exp_dev = f"{round(node.node_prop,3)}"
                    
                    if self.twoing:
                        if self.impurity_fn == "gini":
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split,"Twoing Classes C1":[self.twoing_c1[node]], "Twoing Classes C2": [self.twoing_c2[node]],"n":len(node.indexes), "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev,"Class/Value":[class_node]})
                        elif self.impurity_fn == "tau": 
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split,"Twoing Classes C1":[self.twoing_c1[node]], "Twoing Classes C2": [self.twoing_c2[node]],"n":len(node.indexes),  "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev,"Class/Value":[class_node]})
                        else:
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split,"Twoing Classes C1":[self.twoing_c1[node]], "Twoing Classes C2": [self.twoing_c2[node]],"n":len(node.indexes),  "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev,"Class/Value":[class_node]})

                    else:
                        if self.impurity_fn == "gini":
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split, "n":len(node.indexes), "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev, "Class/Value":[class_node]}) #class node in brackets or need to pass an index 
                        elif self.impurity_fn == "tau": 
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split,"n":len(node.indexes), "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev, "Class/Value":[class_node]})
                        else:
                            new_df = pd.DataFrame({"Node":node.name, "Node Type":"Parent", "Splitting Variable":node.split, "n":len(node.indexes),  "Heterogeneity":f"{round(node.deviance,2)}", "Explained Heterogeneity": exp_dev,"Class/Value":[class_node]})
                    
                    tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
                    
                    node_num = int(node.name[1:])
                    
                    if node_num *2 in leaf_dict.values():
                        cnode = self.get_key(leaf_dict, node_num*2)
                        if self.problem == "regression":
                            class_node = f"{round(mean(self.y[cnode.indexes]),2)}"
                        else:
                            response_dict ={}
                            for response in self.y[cnode.indexes]:        #determing majority in terminal cnodes
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1
                            if self.twoing:
                                total_node_obs = sum(response_dict.values())
                                for key in response_dict:
                                    response_dict[key] = round(response_dict[key] / total_node_obs,2)
                                class_node = response_dict
                                myKeys = list(class_node.keys())
                                myKeys.sort()
                                class_node = [{i: class_node[i] for i in myKeys}]
                            else:
                                class_node = max(response_dict, key = response_dict.get)

                        if self.twoing:
                            if self.impurity_fn == "gini":
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2": None,"n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None, "Class/Value":[class_node]})
                            elif self.impurity_fn == "tau": 
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2":None,"n":len(cnode.indexes), "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None, "Class/Value":[class_node]})
                            else:
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2": None,"n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class/Value":[class_node]})
                        else:
                            if self.impurity_fn == "gini":
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child","Splitting Variable":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class/Value":[class_node]} )
                            elif self.impurity_fn == "tau": 
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"n":len(cnode.indexes), "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class/Value":[class_node]})
                            else:
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child","Splitting Variable":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}", "Explained Heterogeneity": None,"Class/Value":[class_node]})
                        
                        tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
                    
                    if node_num *2+1 in leaf_dict.values():
                        cnode = self.get_key(leaf_dict, node_num*2+1)
                        if self.problem == "regression":
                            class_node = f"{round(mean(self.y[cnode.indexes]),2)}"
                        else:
                            response_dict ={}
                            for response in self.y[cnode.indexes]:        #determing majority in terminal cnodes
                                if response in response_dict:
                                    response_dict[response] +=1
                                else:
                                    response_dict[response] =1
                            if self.twoing:
                                total_node_obs = sum(response_dict.values())
                                for key in response_dict:
                                    response_dict[key] = round(response_dict[key] / total_node_obs,2)
                                class_node = response_dict
                                myKeys = list(class_node.keys())
                                myKeys.sort()
                                class_node = [{i: class_node[i] for i in myKeys}]
                            else:
                                class_node = max(response_dict, key = response_dict.get)
                        if self.twoing:
                            if self.impurity_fn == "gini":
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2": None,"n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}","Explained Heterogeneity": None,"Class/Value":[class_node]})
                            elif self.impurity_fn == "tau": 
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2":None,"n":len(cnode.indexes), "Heterogeneity":f"{round(cnode.deviance,2)}","Explained Heterogeneity": None,"Class/Value":[class_node]})

                            else:
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"Twoing Classes C1":None, "Twoing Classes C2": None,"n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}","Explained Heterogeneity": None,"Class/Value":[class_node]})
                        else:
                            if self.impurity_fn == "gini":
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child","Splitting Variable":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}","Explained Heterogeneity": None,"Class/Value":[class_node]} )
                            elif self.impurity_fn == "tau": 
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child", "Splitting Variable":None,"n":len(cnode.indexes), "Heterogeneity":f"{round(cnode.deviance,2)}","Explained Heterogeneity": None, "Class/Value":[class_node]})
                            else:
                                new_df = pd.DataFrame({"Node":cnode.name, "Node Type":"Child","Splitting Variable":None, "n":len(cnode.indexes),  "Heterogeneity":f"{round(cnode.deviance,2)}","Explained Heterogeneity": None,"Class/Value":[class_node]} )
                        tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
            
            return tree_table
  

    def pred_x(self,node, x, all_node, leaves): #-> tree :
        '''Provides a prediction for the y value (based on the mean of the terminal node), for a new set of unsupervised values'''
        
        node_list =[]
        node_dict ={}
        for node1 in all_node:                      #Creating dictionaries and lists to move between nodes and node numbers
            node_list.append(int(node1.name[1:]))
            node_dict[node1] = int(node1.name[1:])
        
        if node in leaves:                            #Provides the final output for the predicted node
            if self.problem =="classifier":         #checks if the problem is classification
                response_dict ={}
                for response in self.y[node.indexes]:        #determing majority in terminal nodes
                    if response in response_dict:
                        response_dict[response] +=1
                    else:
                        response_dict[response] =1
                class_node = max(response_dict, key = response_dict.get)
                self.prediction_cat.append(class_node)

            else:
                self.prediction_reg.append(mean(self.y[node.indexes]))

            self.pred_node.append(node.name)

            return node
        
        else:
            if isinstance(x, bytes):
                x = x.decode("UTF-8") #numpy arrays

            if self.combination_split: #not 100% functional, issues with errors in x values, and appears every split leads to a true?
                split_string = node.split.split(" in")[0]
                combinations = split_string.split("__")
                split = str(combinations[0])+"__"+str(combinations[1])
                y = {}
                y[split] = x[combinations[0]] + x[combinations[1]] 
                if eval(node.split, y):                 #Evaluates the split for the unsupervised x, whether it is true or not, will deterine if the split goes rigtht or left
                    new_node = self.get_key(node_dict, int(node.name[1:])*2+1)
                    self.pred_x(new_node, x, all_node, leaves) # go to the right child
                else:
                    new_node = self.get_key(node_dict, int(node.name[1:])*2)
                    self.pred_x(new_node, x, all_node, leaves) # go to the left child


            #want to add a section for using surrogates for missing values in prediction, will only do so for non combination split for now. 
            else:
                #print(node.name, node.split)
                split_var = node.split.split(" ")[0] #this is for non terminal nodes so should be ok
                if self.checkNaN(x[split_var]): #checking if nan  
                    for i in node.surrogate_splits:
                        sur_var = i[0].split(" ")[0]
                        if self.checkNaN(x[sur_var]):
                            continue

                        try:
                            if eval(i[0], x):                 #Evaluates the split for the unsupervised x, whether it is true or not, will deterine if the split goes rigtht or left
                                #print("eval", i[0])
                                new_node = self.get_key(node_dict, int(node.name[1:])*2+1)
                                self.pred_x(new_node, x, all_node, leaves) # go to the right child
                                break
                            else:
                                #print("eval2", i[0])
                                new_node = self.get_key(node_dict, int(node.name[1:])*2)
                                self.pred_x(new_node, x, all_node, leaves) # go to the left child
                                break
                        except:
                            continue


                elif eval(node.split, x):                 #Evaluates the split for the unsupervised x, whether it is true or not, will deterine if the split goes rigtht or left
                    new_node = self.get_key(node_dict, int(node.name[1:])*2+1)
                    self.pred_x(new_node, x, all_node, leaves) # go to the right child
                else:
                    new_node = self.get_key(node_dict, int(node.name[1:])*2)
                    self.pred_x(new_node, x, all_node, leaves) # go to the left child
    
    def checkNaN(self, str):
        '''checks for nans'''
        try:
            return math.isnan(float(str))
        except:
            return False
        
    def misclass(self, y):
        '''performs misclass calculation'''
        
        comparison = []
        if self.problem =="classifier":         #checks if the problem is classification        
            for i in range(len(y)):
                comparison.append([y[i], self.prediction_cat[i]])

            count = 0
            for i in comparison:
                if i[0] != i[1]:
                    count +=1
            print("Misclassification", str(round((count/len(self.prediction_cat))/100,6))+ "%")  
        
        else:
            for i in range(len(y)):
                comparison.append([y[i], self.prediction_reg[i]])

            mse = 0
            for i in comparison:
                mse += (i[0] - i[1])**2
            mse = mse/ len(self.prediction_reg)
            print("Deviance ", round(mse,2))              


    def prints(self):
        '''A checking function'''
        for i in self.get_leaf():
            print(len(self.y[i.indexes]),Counter(self.y[i.indexes]))


    def graph_results(self, x1, y1,  dataset1, x2, y2, dataset2):
        '''For plotting the results graph from pruning, with tree size vs error metric'''
        plt.plot(x1, y1, label = dataset1)
        plt.plot(x2, y2, label = dataset2)

        if self.problem =="regression":
            y_label = 'MSE'
        else:
            y_label = 'Misclassification %'

        plt.xlabel('Leaves')
        plt.ylabel(y_label)
        plt.title(f"{y_label} vs Leaves for Training and Test Set for {self.impurity_fn}")    
        plt.legend()
        plt.axis([max(x1+x2)*1.05, min(x1+x2)*.95, min(y1+y2)*0.95, max(y1+y2)*1.05])
        plt.show()
        return

#End TREE4

##################################################
 # K-folds


class k_folds():
    '''A class for completing k-folds methods on TREE4 objects, still in development'''
    def __init__(self, 
                 y, 
                 features,
                 features_names,
                 n_features, 
                 n_features_names,
                 impurity_fn,
                 k = 10, 
                 user_impur=None, 
                 problem = "regression",  
                 method = "CART",
                 twoing = False,
                 min_cases_parent = 10, 
                 min_cases_child = 5, 
                 min_imp_gain=0.01, 
                 max_level = 10 ):
        
        self.y = y
        self.features = features #needs to be an object that can be have its elements accessed with features[var] nomenculature
        self.features_names = features_names
        self.n_features = n_features
        self.n_features_names = n_features_names
        self.problem = problem
        self.impurity_fn = impurity_fn
        self.method = method
        self.user_impur = user_impur
        self.max_level = max_level
        self.twoing = twoing  
        self.min_cases_parent = min_cases_parent
        self.min_cases_child = min_cases_child
        self.min_imp_gain = min_imp_gain                      
        self.k = k 
        
        self.dict_to_dataframe()

        indices = np.arange(0, len(self.df)) #hopefully works
        fold_size = len(self.df) // k
        #within indices generate a number 1:10
        self.folds = []
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            self.folds.append((train_indices, test_indices))

        #folds nested list with outter shape equal to number of folds, inner is length 2, first element is
            #arrat of train, second is test
        self.models = []
        self.overall_errors  = []
        for n, i in enumerate(self.folds):
            
            print("\n\nFolds: ", n +1)

            df = self.df.loc[list(i[0]), :]
            df.reset_index(drop = True, inplace = True)

            y = df["y"]
            features =  df.loc[:, self.features_names]
            n_features = df.loc[:, self.n_features_names]

            ind = np.arange(0, len(self.df) - fold_size)


            my_tree = NodeClass('n1', ind) 
            model = TREE4(y, features, features_names, n_features, n_features_names, impurity_fn = self.impurity_fn, problem = self.problem, method = self.method, max_level = self.max_level) 
            model.growing_tree(my_tree)

            self.models.append(model)


            df_test = self.df.loc[list(i[1]), :]
            df_test.reset_index(drop = True, inplace = True)

            y_test = df_test["y"]
            features_test =  df_test.loc[:, self.features_names]
            n_features_test = df_test.loc[:, self.n_features_names]

            self.prediction_fn(model, y_test, features_test, features_names, n_features_test, n_features_names)
            
            y_list = y_test.to_list() #dataframe
            self.overall_errors.append([sum(self.error_checker(model, y_list))])

        print("\n\n")
        self.errors = []
        for i in range(len(self.models)):
            print("errors: ", self.overall_errors[i][0])
            self.errors.append(self.overall_errors[i][0])

        print("\n\nmean errors:", sum(self.errors) / len(self.errors))



    def dict_to_dataframe(self):
        '''Returns a dataframe with all numerical and categorical variables initialised in 
        TREE4, and the feature variable, with column heading "y"'''
        df = pd.DataFrame(self.features, columns = self.features_names)
        df2 = pd.DataFrame(self.n_features, columns = self.n_features_names)
        df = pd.concat([df, df2], axis = 1)
        df["y"] = self.y
        self.df = df

    def  prediction_fn(self, model, y, X_num_1, num_var_1, X_cat_1, cat_var_1):
        '''Internal prediction function, attaching the model to TREE4.pred_x'''
        for i in range(len(y)): 
            for node in model.get_all_node():  
                if node.name =="n1":
                    new = []
                    new_n = []            

                    if num_var_1:               #was checking if empty , now getting ith observation for pred
                        for name in num_var_1:
                            new.append(X_num_1[name][i])

                    if cat_var_1:
                        for n_name in cat_var_1:
                            new_n.append(X_cat_1[n_name][i])        

                    d = dict(zip(num_var_1, new))
                    dn = dict(zip(cat_var_1, new_n))
                    d.update(dn)
                    model.pred_x(node, d, model.get_all_node(), model.get_leaf()) #no return as the values are stores in the TREE4 class 

    def error_checker(self, model, y_list):
        '''Checks the errors of the model'''
        
        errors = []
        if model.prediction_cat or model.prediction_reg:   #an error checking line 
            for j in range(len(y_list)):
                if self.problem == "regression":                     #appears not fuctional, copied from cat
                        errors.append(((y_list[j] -model.prediction_reg[j])**2 )/len(y_list))
                else:
                    if model.prediction_cat[j] != y_list[j]:
                        errors.append(True)
                    else:
                        errors.append(False)
            
            if self.problem == "regression":
                print("training mse", round(sum(errors),2))
            else:
                print("training missclassifications", sum(errors))
            
        else:
            print("THERE MAY BE AN ISSUE")
            errors = [True]*len(y_list)

        return errors
    

#^kfolds
###########################################
#ADABOOST

# Importing libraries 

#not sure if it needs to be in a class 
    
  
class Adaboost:

    def __init__(self, df, feature_var, num_var, cat_var, _problem, impurity_fn,  method = "CART", weak_learners = 11 , max_level = 0):
        self.df = df
        self.feature_var = feature_var
        self.num_var = num_var
        self.cat_var = cat_var
        self._problem = _problem
        self.impurity_fn = impurity_fn
        self.weak_learners = weak_learners
        self.method = method
        self.max_level = max_level #max level 0 creates the stump
    

        impurity_fns = ["gini", "entropy", "between_variance", "pearson", "tau"]

        if impurity_fn not in impurity_fns:
            print("Name Error: impurity_fn must be in ['gini', 'entropy', 'between_variance', 'pearson', 'tau']")
            #return None
        
        methods = ['CART', 'TWO-STAGE', 'FAST', 'LATENT-BUDGET-TREE']

        if method not in methods:
            print("Name Error: method must be in ['CART', 'TWO-STAGE', 'FAST', 'LATENT-BUDGET-TREE']")
            #return None
        


    def add_weights(self):#, df, first = True):
        '''Adding initial weights : w = 1/n'''
        
        w = [1/ self.df.shape[0] for i in range(self.df.shape[0]) ]
        self.df["weights"] = w                                    #using a dataframe or dictionary setup
            
        self.first = False
        

    
    #could add the alpha and overall errors into a container to be viewed on the outside
    def update_weights(self, alpha, overall_errors):
        '''This is for updating weights, based on whether the observation was correctly predicted'''

        #For a regression problem, the alpha value contains some information about the degree of the change, an innovation. 

        for i in range(len(self.df["weights"])):
            if overall_errors[i] == True:
                self.df["weights"][i] = self.df["weights"][i] * math.exp(alpha)
            elif overall_errors[i] == False:
                self.df["weights"][i] = self.df["weights"][i] * math.exp(-alpha) #correctly classified == False
        
        #normalise
        for i in range(len(self.df["weights"])):
            self.df["weights"][i] = self.df["weights"][i] / self.df["weights"].sum()
        
        cum_sum = [0]
        for i in range(len(self.df["weights"])):
            cum_sum.append(self.df["weights"][i]+cum_sum[-1])
        
        cum_sum.pop(0)
        cum_sum[-1] = 1
        
        self.df["cum_sum"] = cum_sum
        
        


    def new_df(self):
        '''Creates the new dataframe fto be used, based on random sampling, utilising the newly applied weights'''
        random.seed(1)

        random_index =[]
        for i in range(len(self.df["weights"])):
            random_index.append(random.random())
        
        new_indices = []
        previous_val = 0
        for i in range(len(random_index)):
            for j in range(len(self.df["weights"])):
                if random_index[i] < self.df["cum_sum"][j]:
                    new_indices.append(j)
                    break 
        
        new_df = pd.DataFrame(np.zeros(self.df.shape), columns = self.df.columns)
        
        count = 0
        for i in  new_indices:
            new_df.iloc[count] = self.df.iloc[i]
            count +=1
        
        #del df
        self.df = new_df




    def dict_sum(self, df_series):
        '''Performs a sum of the values in a dictionary'''
        dict_val = {}
        for n in df_series.to_list(): #counting instances of the class
            if n in dict_val:
                dict_val[n] +=1
            else:
                dict_val[n] =1
        return dict_val



    def adaboost(self):
        '''adaboost algorithm'''

        self.first = True                                      #checks whether it is the first iteration (weights = 1/n)
        iterations = 0
        best_weak =[]                                          #adds the best weak learnings to a list 
        final_predictions = pd.DataFrame(self.df[self.feature_var]) #self.feature_var is the first missing variable

        while iterations < self.weak_learners:
            iterations +=1
            print("\nIteration",iterations)

            if not self.first:
                self.update_weights(alpha, overall_errors[best_index]) 
                self.new_df()
                self.add_weights() 

            else:
                self.add_weights()

            #training set
            y = self.df[self.feature_var]       #applies feature var (each feature variable is the y to be predicted)
            y_list = y.to_list()
            X = self.df.drop(labels = [self.feature_var,"weights", "cum_sum"], axis = 1, errors = "ignore")
            X_num = self.df[self.num_var]       #selecting multiple items        
            X_cat = self.df[self.cat_var]    
            
            weak_learner = []
            overall_errors =[]

            num_var_1 = self.num_var.copy()
            cat_var_1 = self.cat_var.copy()
            if self.feature_var in self.num_var:
                num_var_1.remove(self.feature_var)
            elif self.feature_var in self.cat_var:
                cat_var_1.remove(self.feature_var)


            my_tree = NodeClass('n1', np.arange(len(y)))
            model = TREE4(y, X_num, num_var_1, X_cat, cat_var_1, impurity_fn = self.impurity_fn, problem = self._problem, method = self.method, max_level = self.max_level) 
            model.growing_tree(my_tree)
            self.prediction_fn(model, y, X_num, num_var_1, X_cat, cat_var_1)
            overall_errors.append(self.error_checker(model, y_list))
            
            if self._problem == "regression":                    
                weak_learner.append([model.prediction_reg[:len(y)], model.get_all_node(), model.get_leaf(), model])  
            else:
                weak_learner.append([model.prediction_cat[:len(y)], model.get_all_node(), model.get_leaf(), model])

            #not sure how this works            
            error_metric = []
            for error in overall_errors:
                error_metric.append(sum(error))
    
            #if not pure:
            #best = min(error_metric)
            best_index = error_metric.index(min(error_metric))
            colname = "pred"+str(iterations)
            final_predictions[colname] = weak_learner[best_index][0][:len(y)]

            alpha = self.alpha_calculator(overall_errors, best_index)
            if math.isinf(alpha):
                print("no errors")
                break

            best_weak.append([weak_learner[best_index], error_metric[best_index], alpha, overall_errors[best_index]])
    
            #clearing memory items
            #del y
            #del X_cat
            #del X_cat_1
            #del X_num
            #del X_num_1
            #del X
            #gc.collect()

        final_model =[]
        combined_response = []
        models = []
        alphas = []
        for i in best_weak:
            final_model.append([i[-2],i[1], i[0]])
            combined_response.append(i[0][0])
            models.append(i[0][-1])
            alphas.append(i[-2])

        self.final_model = final_model
        self.models = models
        self.alphas = alphas

        final_predictions = self.vote(combined_response, final_predictions,  trains = True)
        final_e = self.final_error(y_list, final_predictions)

        if self._problem == "classifier":
            print("Final Training Missclassification", sum(final_e), "\n")
            self.error = sum(final_e)
        else:
            print("Final Training MSE", round(sum(final_e),2), "\n")
            self.error = round(sum(final_e),2)
        

        return {"final_model":final_model, "models":models, "alphas" : alphas}

    def test_prediction(self, y_test, num_var, cat_var, X_test_num, X_test_cat):
        '''Prediction function'''

        test_predictions = pd.DataFrame(y_test)
        
        for model in self.models:
            weak_learner_test = []
            for j in range(len(y_test)): 
                for node in model.get_all_node():  
                    if node.name =="n1":

                        new = []
                        new_n = []            
                        for name in num_var:
                            new.append(X_test_num[name])#used to have [j] index notation, but these are signle predictions

                        for n_name in cat_var:
                            new_n.append(str(X_test_cat[n_name])) #TODO added str?        

                        d = dict(zip(num_var, new))
                        dn = dict(zip(cat_var, new_n))
                        d.update(dn)
        
                        model.pred_x(node, d, model.get_all_node(), model.get_leaf()) #appending to a list in TREE4 
            
            if self._problem == "classifier":
                weak_learner_test.append(model.prediction_cat[-len(y_test):]) 
            else:
                weak_learner_test.append(model.prediction_reg[-len(y_test):]) 
            colname = "pred"+str(self.models.index(model))
            test_predictions[colname] = weak_learner_test[0] #is a nested list

        test_predictions = self.vote(y_test, test_predictions) #cant multiple by alpha here , need to assign it to the node, the value

        print("Prediction", test_predictions["final_pred"][0] )

        return test_predictions["final_pred"] 



    def get_key(self, my_dict, val):
        '''Returns the key from a dictionary value'''
        for key, value in my_dict.items():
            if val == value:
                return key

        return "key doesn't exist"



    def vote(self,y_test, test_predictions, trains = False):
        '''Voting mechanism for prediction selection'''
        
        final_pred_test = []    

        #do the votes need to be weighted by the alpha value as per schapire 1999, added alpha values to come into the fn

        if not trains:
            if self._problem == "classifier":
                for i in range(len(y_test)):
                    votes = []
                    for j in range(1,test_predictions.shape[1]):
                        votes.append(test_predictions.iloc[i,j])
                    #final_pred_test.append(max(set(votes), key = votes.count))     #takes the highest vote
                    scaled = {}
                    for i in range(len(votes)):
                        if votes[i] in scaled:
                            scaled[votes[i]] += self.alphas[i]
                        else:
                            scaled[votes[i]] = self.alphas[i]
                    
                    final_pred_test.append(self.get_key(scaled,max(scaled.values()))) #take highest alpha contribution
            else:
                for i in range(len(y_test)):
                    votes = []
                    for j in range(1,test_predictions.shape[1]):
                        votes.append(test_predictions.iloc[i,j])
                    #final_pred_test.append(mean(votes))       #pretty sure mean vote is right, could be mode
                    final_pred_test.append(sum([votes[i] * self.alphas[i] for i in range(len(votes))])/ sum(self.alphas))
        else:
            if self._problem == "classifier":
                for i in range(len(y_test[0])):
                    votes = []
                    for j in range(len(y_test)):
                        votes.append(y_test[j][i])
                    #final_pred_test.append(max(set(votes), key = votes.count))     #takes the highest vote
                    scaled = {}
                    for i in range(len(votes)):
                        if votes[i] in scaled:
                            scaled[votes[i]] += self.alphas[i]
                        else:
                            scaled[votes[i]] = self.alphas[i]
                    
                    final_pred_test.append(self.get_key(scaled,max(scaled.values()))) #take highest alpha contribution
            else:
                for i in range(len(y_test[0])):
                    votes = []
                    for j in range(len(y_test)):
                        votes.append(y_test[j][i])
                    #final_pred_test.append(mean(votes))       #pretty sure mean vote is right, could be mode
                    final_pred_test.append(sum([votes[i] * self.alphas[i] for i in range(len(votes))])/ sum(self.alphas))


        
        test_predictions["final_pred"] = final_pred_test

        return test_predictions


    def  prediction_fn(self, model, y, X_num_1, num_var_1, X_cat_1, cat_var_1):
        '''Internal prediction function, attaching the model to TREE4.pred_x'''
        for i in range(len(y)): 
            for node in model.get_all_node():  
                if node.name =="n1":
                    new = []
                    new_n = []            

                    if num_var_1:               #was checking if empty , now getting ith observation for pred
                        for name in num_var_1:
                            new.append(X_num_1[name][i])

                    if cat_var_1:
                        for n_name in cat_var_1:
                            new_n.append(X_cat_1[n_name][i])        

                    d = dict(zip(num_var_1, new))
                    dn = dict(zip(cat_var_1, new_n))
                    d.update(dn)
                    model.pred_x(node, d, model.get_all_node(), model.get_leaf()) #no return as the values are stores in the TREE4 class 


    def error_checker(self, model, y_list):
        '''Checks the errors of the model'''
        
        errors = []
        if model.prediction_cat or model.prediction_reg:   #an error checking line 
            for j in range(len(y_list)):
                if self._problem == "regression":                     #appears not fuctional, copied from cat
                        errors.append(((y_list[j] -model.prediction_reg[j])**2 )/len(y_list))

                else:
                    if model.prediction_cat[j] != y_list[j]:
                        errors.append(True)
                    else:
                        errors.append(False)
            
            if self._problem == "regression":
                print("training mse", round(sum(errors),2))
            else:
                print("training missclassifications", sum(errors))
            
        else:
            print("THERE MAY BE AN ISSUE")
            errors = [True]*len(y_list)

        return errors
        

    def alpha_calculator(self, overall_errors, best_index):
        '''alpha calculator for the adaboost function'''

        TE = 0 #total error 
        for i in range(len(self.df["weights"])):
            if self._problem == "classifier":
                TE += self.df["weights"][i] * overall_errors[best_index][i] 
            else:
                TE += self.df["weights"][i] * overall_errors[best_index][i] / (max( overall_errors[best_index]) - min( overall_errors[best_index])) #may need a different variation for regression, needs to be probability

        alpha = 0.5 * np.log ( ((1-TE) / TE) + 1e-7)
        return alpha



    def final_error(self, y_list, final_predictions):
        '''calculate final error of the model'''
        
        final_miss =[]
        if self._problem == "classifier":
            for i in range(len(y_list)):
                if y_list[i] != final_predictions["final_pred"][i]:
                    final_miss.append(True)
                else:
                    final_miss.append(False)
        else:
            for i in range(len(y_list)):
                final_miss.append(((y_list[i] - final_predictions["final_pred"][i])**2)/len(y_list))
        return final_miss





#end ADABOOST
##################################################

#start BINPI


class BINPI:

    def __init__(self,df, num_var, bin_var, class_var, weak_learners = 7):
        self.df = df
        self.num_var = num_var
        self.bin_var = bin_var
        self.class_var = class_var
        self.weak_learners = weak_learners
        self.cat_var = bin_var + class_var
        
        #def id_matrix_creator(self):

        id_matrix = self.df.notna()
        id_matrix.replace(True, "a", inplace = True)
        id_matrix.replace(False, "b", inplace = True)
        id_matrix.replace("a", 0, inplace = True)
        id_matrix.replace("b", 1, inplace = True)
        self.id_matrix = id_matrix
        #return id_matrix



    def row_vector(self):
        '''Returns a row vector sorted by missingness for use in lexicographical_matrix'''
        row_vect = []

        for variable_name in self.id_matrix.columns:
            row_vect.append((self.id_matrix[variable_name].sum(axis=0), variable_name ))

        row_vect.sort()
        row_name_vect = []

        for i in row_vect:
            row_name_vect.append(i[1])

        return row_name_vect, row_vect


    def condition(self, element):
        '''Sorts list by amount of missing, and then by alphabetical order of variable name'''

        return element[0], element[2]



    def column_vector(self):
        '''Returns a column vector sorted by missinness for use in lexicographical_matrix'''

        column_vect = []
        for row_number in range(self.id_matrix.shape[0]):
            column_vect.append([self.id_matrix.iloc[row_number].sum(axis=0), row_number])

        column_vect_dict = {}

        for value in column_vect:
            if value[0] in column_vect_dict:
                column_vect_dict[value[0]] +=1
            else:
                column_vect_dict[value[0]] = 1

        for values in column_vect:
            if values[0] == 0:
                values.append("NM") #for no missing
            else:
                position = 0
                list_var = []
                for variable_name in self.id_matrix.columns:
                    position +=1
                    if self.id_matrix.iloc[values[1], position-1] == 1:
                        list_var.append(variable_name)
                values.append(list_var)
                
        column_vect.sort(key = self.condition)
        column_number_vect = []

        for i in column_vect:
            column_number_vect.append(i[1])

        return column_number_vect, column_vect

    '''
    def rearrangement(df, row_name_vector, column_number_vector):

        df2=df.reindex(columns= row_name_vector)
        df2 = df2.reindex(column_number_vector)

        return df2

    '''
    def lexicographical_matrix(self):
        '''Returns a matrix sorted by missingness'''
        row_name_vector, row_vect = self.row_vector()
        column_number_vector, column_vect = self.column_vector()
        df2 = self.df.reindex(columns= row_name_vector)
        df2 = df2.reindex(column_number_vector)

        self.column_vect = column_vect 
        self.df2 = df2

        return df2

    def first_nan(self, last_nan = 0):
        '''Finds the first nan'''
        #if you wanted to find the maximal size when it wasnt so obvious, do it during this step, and pass column_vect 
        skip_point = (False, 0, 0, 0)
        row_no = last_nan                    #skip full iteration, and start with last nan only
        
        for pair in self.column_vect[last_nan:]:
            if pair[0] > 0: 
                if pair[0] >1:
                    skip_point = (True,  pair[0], pair[1], pair[2])
                
                self.column_vect[row_no] = (pair[0]-1, pair[1], pair[2]) #adjust the list removing the to be imputated value
                last_nan = row_no
                break
            else:
                first = None
            row_no +=1

        return row_no, last_nan, skip_point



    def matches_dict(self, column_vect):
        '''Finds matches'''
        dict_match = {}
        for i in column_vect:
            if " ".join(i[2]) in dict_match:
                dict_match[" ".join(i[2])] +=1
            else:
                dict_match[" ".join(i[2])] = 1

        return dict_match 



    def checkNaN(self, str):
        '''Checks for nans'''
        try:
            return math.isnan(float(str))
        except:
            return False
        

    def feature_variable(self, row_no):
        '''Checks for the feature variable'''
        row_no
        pos = 0
        feature_var = "a"
        for value in self.df2.iloc[row_no]:
            pos += 1

            if self.checkNaN(value):            #this only looks for the first instance 
                feature_var = self.df2.columns[pos-1]
                break
        return feature_var, pos


    

    def imputation_process(self, feature_var, row_no, pos, old_model="",  previous_var= "", old_adaboost = ""):
        '''Checks for the 3 types of imputation processes '''
        

        if self.sklearn:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.ensemble import AdaBoostRegressor


        complete_df = self.df2.iloc[0:row_no].copy()  #subset only the complete dataset
        complete_df.reset_index(drop = True, inplace = True)  

        y = complete_df[feature_var]
        X = complete_df.drop(feature_var, axis = 1)

        #As a temporary fix for multiple missing values, will use mean imputation for a secondary, tertiary etc missing value temporarily 
        prediction_feat = self.df2.iloc[row_no].copy()
        #print("prediction_feat2\n",prediction_feat)
        prediction_feat.drop(feature_var, inplace = True)
        
        for series_name in X.columns:                                  
            if self.checkNaN(prediction_feat[series_name]):
                if series_name in self.cat_var:
                    prediction_feat[series_name] = Counter(X[series_name][X[series_name].notna()]).most_common(1)[0][0] 
                    #print("prediction_feat", series_name ,Counter(X[series_name][X[series_name].notna()]).most_common(1)[0][0])
                else:
                    prediction_feat[series_name] = round(mean(X[series_name][X[series_name].notna()]),0)


        y_test = [1] #one element list, used for len and also for helping voting mechanism

        #removing feature_var from appropiate list, before indexing
        if feature_var in self.num_var:
            num_var_full = self.num_var.copy()
            num_var_1 = self.num_var.copy()
            cat_var_1 = self.cat_var.copy()
            num_var_1.remove(feature_var)

        elif feature_var in self.cat_var:
            num_var_full = self.num_var.copy()
            num_var_1 = self.num_var.copy()
            cat_var_1 = self.cat_var.copy()
            cat_var_1.remove(feature_var) 
        else:
            print("Variable Error", feature_var, self.num_var, self.cat_var)

        X_test_num = prediction_feat[num_var_1]  
        X_test_cat = prediction_feat[cat_var_1] 
        
        imp_time_start = time.time()

        if not self.sklearn:
            if feature_var in num_var_full:
                if feature_var != previous_var:
                    #don't think it matters if i pass num_var_full or num_var as there is filtering later
                    adaboost = Adaboost(df = complete_df, feature_var = feature_var, num_var= self.num_var, cat_var = self.cat_var, _problem = "regression", impurity_fn = "pearson",  method = "FAST", weak_learners = self.weak_learners , max_level = 0)
                    model = adaboost.adaboost()  
                    yhat = adaboost.test_prediction(y_test, num_var_1, cat_var_1, X_test_num, X_test_cat) 
                else:
                    yhat = old_adaboost.test_prediction(y_test, num_var_1, cat_var_1, X_test_num, X_test_cat)
                    model = old_model
                    adaboost = old_adaboost
            
            elif feature_var in self.bin_var:
                if feature_var != previous_var:
                    adaboost = Adaboost(df = complete_df, feature_var = feature_var, num_var= self.num_var, cat_var = self.cat_var, _problem = "classifier", impurity_fn = "tau",  method = "FAST", weak_learners = self.weak_learners , max_level = 0)
                    model = adaboost.adaboost() 
                    yhat = adaboost.test_prediction(y_test, num_var_1, cat_var_1, X_test_num, X_test_cat)
                else:
                    yhat = old_adaboost.test_prediction(y_test, num_var_1, cat_var_1, X_test_num, X_test_cat)
                    model = old_model
                    adaboost = old_adaboost

            elif feature_var in self.class_var:
                if feature_var != previous_var:
                    
                    adaboost = Adaboost(df = complete_df, feature_var = feature_var, num_var= self.num_var, cat_var = self.cat_var, _problem = "classifier", impurity_fn = "tau",  method = "FAST", weak_learners = self.weak_learners , max_level = 3)
                    model = adaboost.adaboost()  
                    yhat = adaboost.test_prediction(y_test, num_var_1, cat_var_1, X_test_num, X_test_cat)
                else:
                    yhat = old_adaboost.test_prediction(y_test, num_var_1, cat_var_1, X_test_num, X_test_cat)
                    model = old_model
                    adaboost = old_adaboost
            else: 
                print("Error, found variable missing from variable lists")

        else:
            if feature_var in self.num_var:
                if feature_var != previous_var:
                    adaboost = AdaBoostRegressor(random_state = 42, n_estimators = self.weak_learners)
                    model = adaboost.fit(X.values,y.values)
                    yhat = adaboost.predict([prediction_feat])
                else:
                    yhat = old_adaboost.predict([prediction_feat])
                    model, adaboost = old_model, old_adaboost
            
            elif feature_var in self.bin_var:
                if feature_var != previous_var:
                    adaboost = AdaBoostClassifier(random_state = 42, n_estimators = self.weak_learners)
                    model = adaboost.fit(X.values,y.values)
                    yhat = adaboost.predict([prediction_feat])
                else:
                    yhat = old_adaboost.predict([prediction_feat])
                    model, adaboost = old_model, old_adaboost

            elif feature_var in self.class_var:
                if feature_var != previous_var:
                    adaboost = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 3),  random_state = 42, n_estimators = self.weak_learners)
                    model = adaboost.fit(X.values,y.values)
                    yhat = adaboost.predict([prediction_feat]) 
                else:
                    yhat = old_adaboost.predict([prediction_feat])
                    model, adaboost = old_model, old_adaboost
            else: 
                print("Error, found variable missing from variable lists")


        #Applying the value to the dataset
        self.df2.iloc[row_no, pos-1] = yhat[0]
        previous_var = feature_var #used for reusing the model 

        print("imp time", time.time() - imp_time_start)
        
        return model, previous_var, adaboost

    def binpi_imputation(self, sklearn = False):
        '''Actual imputation process'''

        self.sklearn = sklearn

        #Future adaption - for a dataset with no complete area, need to impute the least missing column with a simple method, mean mode, andrea frazzoni
        dict_match = self.matches_dict(self.column_vect)

        last_nan = 0
        iteration = 0 
        while self.df2.isna().any().any() > 0: 
    
            start = time.time()
            iteration +=1
    
            row_no, last_nan, skip_point = self.first_nan(last_nan)       #finds first nan
            feature_var, pos = self.feature_variable(row_no)

            if skip_point[0]: #checks if can reuse model 

                #feature_var, pos = feature_variable(df2, row_no)

                for i in range(dict_match[" ".join(skip_point[3])]):     

                    if iteration >1:
                        model_1, previous_var_1, adaboost_1 = self.imputation_process(feature_var, row_no, pos,  old_model,  previous_var, adaboost)
                    else:
                        model_1, previous_var_1, adaboost_1 = self.imputation_process( feature_var, row_no, pos)

                    old_model,  previous_var, adaboost =  model_1, previous_var_1, adaboost_1

                    print("time", time.time() - start)
                    iteration +=1

                    if i >0:

                        self.column_vect[row_no] = (skip_point[1]-1, skip_point[2], skip_point[3]) # for multi missing points, to stop it from going back in
            
                    row_no+=1
                continue

            #feature_var, pos = feature_variable(df2, row_no)

            print("\nFeature Variable: ", feature_var, "\nMissing Values: ", self.df2.isna().sum().sum())

            if iteration >1:
                model_1, previous_var_1, adaboost_1 = self.imputation_process( feature_var, row_no, pos,  old_model,  previous_var, adaboost)
            else:
                model_1, previous_var_1, adaboost_1 = self.imputation_process( feature_var, row_no, pos)

            old_model,  previous_var, adaboost =  model_1, previous_var_1, adaboost_1
            print("time", time.time() - start)

        return self.df2
    

#^BINPI
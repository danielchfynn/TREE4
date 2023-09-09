#Future ideas 
#cross validation 
#check if between variance is ok, swap for deviance
#speed up multiprocessing or joblib 

import itertools #base library
import math #base library
import numpy as np # use numpy arraysfrom
from  statistics import mean,variance,mode #base library
from anytree import Node, RenderTree, NodeMixin
from collections import Counter #base library
import matplotlib.pyplot as plt
#from anytree.exporter import DotExporter
#from anytree.dotexport import RenderTreeGraph
#from anytree.exporter import DictExporter
import pydot
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
#from pygments import highlight
import webbrowser #base library 


import random #base library 
import pandas as pd
import gc #base library 
import time #base library 

#rpy2 objects for lba
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.rinterface as rinterface
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri

#rpy2.robjects.packages.quiet_require('utils')
#import rpy2.rinterface as rinterface
#rinterface.initr((b'rpy2', b'--no-save', b'--no-restore', b'--quiet'))

#utils = importr('utils')  
#utils.chooseCRANmirror(ind=1) 
#utils.install_packages("lba")
#lba = importr("lba")
#base = importr('base')

robjects.r("library(utils, quietly = TRUE)")
robjects.r("library(base, quietly = TRUE)")
#robjects.r("suppressWarnings(install.packages('lba', quiet = TRUE))")
robjects.r("suppressWarnings(suppressMessages(library(lba, quietly = TRUE)))")

pd.options.mode.chained_assignment = None #settingwithcopywarning
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 3000)

# define base class for nodes
class MyBaseClass(object):  # Just a basic base class
    value = None            # it only brings the node value


class MyNodeClass(MyBaseClass, NodeMixin):  # Add Node feature
    
    children = []
    value_soglia_split = []
    beta = []
    alpha = []
    error = []

    def __init__(self, name, indexes, split=None, parent=None,node_level= 0,to_pop = False):
        super(MyNodeClass, self).__init__()
        self.name = name                   # id n_node number
        self.indexes = indexes             # array of indexes of cases
        #self.impurity = impurity          # vue in the node of the chosen impurity function
        self.split = split                 # string of the split (if any in the node, None => leaf)
        self.parent = parent               # parent node (if None => root node)
        self.node_level = node_level       # Tiene traccia del livello dei nodi all'interno dell albero in ordine crescente : il root node avrà livello 0
        self.to_pop = to_pop
        

    def get_value(self, y, problem):
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
        ritorna il figlio
        se esiste 
        altrimenti none
        
        '''
        return self.children
    

    def get_value_thresh(self):
        return self.value_soglia_split[0][0:2] + [self.value_soglia_split[0][3]]
        

    def set_to_pop(self):
        '''
        Durante il growing tiene traccia dei nodi da potare.
        '''
        self.to_pop = True 


    def get_name(self):
        return self.name
    

    def get_level(self):
        return self.node_level
    

    def set_features(self,features):
        self.features = features
    

    def get_parent(self):
        '''
        return the parent node 
        if the the parent node is None , is the root.
        '''
        return self.parent
    

    def set_children(self,lista:list):#lista di nodi    
        for i in lista:
            self.children.append(i)
    
    def set_split2(self, split):
        self.split = split

    def set_split(self,value_soglia):
        self.value_soglia_split = value_soglia
    
    def set_beta(self, beta):
        self.beta = beta
    
    def set_alpha(self,alpha):
        self.alpha = alpha

    def set_error(self, error):
        self.error = error

    # define binary split mechanics (for numerical variables)
    def bin_split(self, feat, feat_nominal, var_name, soglia):
        #_self_ is the node object, feat and feature_names (these could be better implemented via a *dict*)
        # var_name the string name and soglia the sogliashold
        if var_name in feat:         #is_numeric(var) :      # split for numerical variables
            var = self.features[var_name]    # obtains the var column by identifiying the feature name 
            self.split = var_name + ">" + str(soglia) # compose the split string (just for numerical features)
            parent = self.name
            select = var[(self.indexes)] > soglia              # split cases belonging to the parent node
        elif  var_name in feat_nominal:         #is_numeric(var) :      # split for nominal variables
            var = feat_nominal[var_name]    # obtains the var column by identifiying the nominal feature name 

            #TODO may need to write more to allow for classes with a single char
            if type(soglia) is tuple:
                self.split = var_name + " in " + str(soglia) # compose the split string (just for numerical features)
            else:
                self.split = var_name + " in " + "'" +str(soglia)+"'" 

            parent = self.name
            select = np.array([i in soglia for i in var[(self.indexes)]]) # split cases belonging to the parent node

        else :
            print("Var name is not among the supplied features!")
            return
        
        left_i = self.indexes[~select]                      # to the left child criterion FALSE
        right_i = self.indexes[select]                      # to the right child criterion TRUE
        child_l = "n" + str(int(parent.replace("n",""))*2)
        child_r = "n" + str(int(parent.replace("n",""))*2 + 1)         
        return MyNodeClass(child_l, left_i, None, parent = self,node_level=self.node_level+1), MyNodeClass(child_r, right_i, None, parent = self,node_level=self.node_level+1)   # instantiate left & right children
            
    # add a method to fast render the tree in ASCII
    def print_tree(self):
        for pre, _, node in RenderTree(self):
            treestr = u"%s%s" % (pre, node.name)
            print(treestr.ljust(8), node.split, node.indexes)


class CART:
    '''
    bigtree =  []
    nsplit = 0
    father = []
    root = []
    tree = []
    father_to_pop = []
    node_prop_list = []
    grow_rules = {}
    leaf = []
    all_node = []
    prediction_cat = []
    prediction_reg = []
    '''

    def __init__(self,y,features,features_names,n_features \
                    ,n_features_names, impurity_fn, user_impur=None, problem = "regression"  \
                    ,method = "CART"
                    ,twoing = False
                    ,min_cases_parent = 10 \
                    ,min_cases_child = 5\
                    ,min_imp_gain=0.01
                    ,max_level = 10):

        self.y = y
        self.features = features
        self.features_names = features_names
        self.n_features = n_features
        self.n_features_names = n_features_names
        self.problem = problem
        self.impurity_fn = impurity_fn
        self.method = method
        self.user_impur = user_impur
        self.max_level = max_level
        self.twoing = twoing

        if problem =="regression":
            self.devian_y = len(self.y)*variance(self.y) # impurity function will equal between variance 
        elif problem == "classifier":
            #n_class = len(np.unique(self.y))
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
    
        self.bigtree =  []
        self.nsplit = 0
        self.father = []
        self.root = []
        self.tree = []
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

    def user_impur_fn(self, func, node):
        return func(self, node)
        

    def impur(self,node, display = False):
        if self.problem =='regression':

            if self.impurity_fn =="between_variance":
                return (mean(self.y[node.indexes])**2)*len(self.y[node.indexes]) 
            
            elif self.impurity_fn == "pearson":
                df = self.dict_to_dataframe()
                df = df.iloc[node.indexes]
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
                        prom += prob_i*i[1]#/len(self.y[node.indexes]) #original weighted, only looking at purity # got rid of issues with cart gini
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
        return self.root


    def __get_RSS(self,node):
        '''
        return the RSS of a node
        this funcion is for only internal uses (private_funcion)
        '''
        mean_y = mean(self.y[node.indexes])
        return (1/len(node.indexes)*sum((self.y[node.indexes] - mean_y)**2))


    def get_all_node(self):
        foglie = [nodi for nodi in self.get_leaf()]
        self.all_node = foglie + self.get_father()
        return foglie + self.get_father()


    def dict_to_dataframe(self):
        '''Returns a dataframe with all numerical and categorical variables initialised in 
        CART, and the feature variable, with column heading "y"'''
        df = pd.DataFrame(self.features, columns = self.features_names)
        df2 = pd.DataFrame(self.n_features, columns = self.n_features_names)
        df = pd.concat([df, df2], axis = 1)
        df["y"] = self.y
        return df
    

    def gini(self, node): #will need to include the node
        df = self.dict_to_dataframe()
        df = df.iloc[node.indexes]
        gini = 0
        for j in list(set(df["y"])):
            gini += (len(df.loc[df["y"] == j])/len(df))**2
        return gini


    def tau_ordering(self, node):
        df = self.dict_to_dataframe()
        df = df.iloc[node.indexes]
        gini = self.gini(node)

        tau_list = []
        for var in self.features_names+ self.n_features_names:
            firstterm = 0 
            for i in list(set(df[var])):
                df2 = df.loc[df[var]==i]
                for j in list(set(df2["y"])):
                    firstterm += (len(df2.loc[df2["y"] == j])/len(df2))**2 * len(df2)/len(df) 
            tau_list.append(((firstterm - gini) / (1-gini), var))
        tau_list.sort(reverse = True)
        return tau_list
    
    
    def tss(self, node):
        df = self.dict_to_dataframe()
        df = df.iloc[node.indexes]
        tss = 0
        mean_y = mean(df["y"])
        for j in range(len(df["y"])):
            tss += (df["y"].iloc[j] - mean_y)**2
        return tss


    def pearson_ordering(self, node):
        df = self.dict_to_dataframe()
        df = df.iloc[node.indexes]
        tss = self.tss(node)

        pearson_list = []
        for var in self.features_names+ self.n_features_names:
            wss = 0 
            for i in list(set(df[var])):
                df2 = df.loc[df[var]==i]
                if len(df2["y"]) > 1: #there is only a within, when theres more than 1, otherwise its 0 
                    mean_y = mean(df2["y"])
                    for j in range(len(df2["y"])):
                        wss += (df2["y"].iloc[j] - mean_y)**2
            pearson_list.append((1- wss/ tss, var))

        pearson_list.sort(reverse = True)
        return pearson_list


    def __node_search_split(self,node:MyNodeClass, max_k, combination_split, max_c):

        '''
        The function return the best split thath the node may compute.
        Il calcolo è effettuato effettuando ogni possibile split e 
        calcolando la massima between variance 
        tra i nodi figli creati.
       
       Attenzione: questo è un metodo privato non chiamabile a di fuori della classe.
        '''
        
        impurities_1=[]
        between_variance=[]
        splits=[]
        variables=[]
        distinct_values=np.array([])
        t=0
        k = False
        
        node.set_features(self.features)
        if Counter(self.y[node.indexes]).most_common(1)[0][1] == len(self.y[node.indexes]):

            print("This split isn't good now i cut it [counter] - node class purity")
            node.get_parent().set_to_pop()
            node.get_parent().set_to_pop()
            self.father_to_pop.append(node)
            node.set_split2(None)
            return None

        if len(node.indexes) >= self.grow_rules['min_cases_parent']:
            
            #will implement as two-stage, finidng the best split of the highest tau
            #stage 1
            if self.method == "LATENT-BUDGET-TREE": #classification only method     #could pass the impurity fn as the method to use in lba [ls or mle] as not used
                if self.problem == "classifier":
                    ordered_list = self.tau_ordering(node)  
                else:
                    print("Latent Budget Tree only works with Classifier response variable")
                    return None
                #print("ordered_list",len(ordered_list), ordered_list)
                df = self.dict_to_dataframe().iloc[node.indexes].copy() #creates subset of overall df dependent on indexes in node and a copy so the combination split isnt kept
                
                betas = []
                alphas = []
                errors = []
                k = -1
                
                while k < len(ordered_list)-1:
                    k +=1   
                    n = 0                    #can start a while loop here with k to use try, except to continue loop
                    #print(k, ordered_list[k][1])
                    while n <= max_c: 
                        n+=1
                        #print("combination_split11", k, n, len(ordered_list))
                        if combination_split:
                            if k  < len(ordered_list)-n: #combines the ajoined high tau values, as soon as it hits the bottom once it will stop, max_c should be a third of predictors max
                                comb_split = str(ordered_list[k][1])+"__"+str(ordered_list[k+n][1])
                                df[comb_split] = df[ordered_list[k][1]] + df[ordered_list[k+n][1]] 
                                cont = pd.crosstab(index = df[comb_split], columns= df["y"], normalize = 'index')
                            else:
                                if len(splits):
                                    print("Unable to go through max_k, only went through: ", len(splits), "time/s")
                                    best_index = between_variance.index(max(between_variance))
                                    node.set_beta(betas[best_index])
                                    node.set_alpha(alphas[best_index])
                                    node.set_error(errors[best_index])

                                    var1, var2 = variables[best_index].split("__")
                                    self.n_features[variables[best_index]] = self.n_features[var1] + self.n_features[var2]
                                    return variables[best_index], tuple(splits[best_index]), between_variance[best_index] 
                                else:
                                    print("No splits found")
                                    return None
                        else:
                            #creates crosstable
                            cont = pd.crosstab(index = df[ordered_list[k][1]], columns= df["y"], normalize = 'index')

                        #converts into an r dataframe
                        with (robjects.default_converter + pandas2ri.converter).context():
                            cont_r = robjects.conversion.get_conversion().py2rpy(cont)


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

                            robjects.r("error <- out$val_func")
                            error = robjects.r('error')
                            error = np.asarray(error)
                            #error = -round(error.item(), 16)
                            #out = lba.lba(base.as_matrix(cont_r), K = 2 , what = 'outer', method = 'ls') #base.trace.lba = 0 doesnt work
                        except:
                            print("Error in LBA function")
                            time.sleep(4) #issue with printing order bewtten python n r 
                            continue
                        
                        split = []
                        for i in range(alpha.shape[0]):
                            if alpha[i][0] >= 0.5:            #threshold point set to 0.5, what if the alphas are less than 0.5 for both groups, i think it gets caught later by teh delta fn
                                split.append(cont.index[i])

                        if split and len(split) != alpha.shape[0]: #len(set(df[ordered_list[k][1]])):  #looks that at least 1 alpha > 0.5, and not all values

                            #might do some split evaluation here 
                            if not combination_split:
                                stump = node.bin_split(self.features, self.n_features, str(ordered_list[k][1]),split)
                            else:
                                #print("combination_split", df.columns, comb_split)
                                stump = node.bin_split(self.features, df.copy() , comb_split,split) #may not work later when evaluating in the table
                               
                            if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child'] \
                                and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:

                                impur0 = self.impur(stump[0])
                                impur1 = self.impur(stump[1])

                                splits.append(split) #had list around it , had -1index
                                #between_variance.append(error)
                                if combination_split:
                                    variables.append(comb_split)
                                else:
                                    variables.append(ordered_list[k][1])
                                betas.append(np.around(beta,2).tolist()) #before were still arrays
                                alphas.append(np.around(alpha,2).tolist())
                                errors.append(np.around(error,2).tolist())

                                if self.impurity_fn =="entropy":
                                    entropy_parent = self.impur(node)
                                    inf_gain = entropy_parent - ((len(stump[0].indexes) / len(node.indexes)) * impur0 + (len(stump[1].indexes) / len(node.indexes)) * impur1)
                                    between_variance.append([inf_gain, error])                                
                                else:
                                    between_variance.append([(impur0) + (impur1), error]) #TODO evaluate if error is good enough to use as metric 
                            else:
                                continue
                        else:
                            #when no split is found with alpha greater than 0.5, or len(split) is = len(set(var))
                            continue

                        #print(splits, between_variance, variables, alpha, cont )
                        #print(cont, alpha, split)


                        #max_k = 2 #allows for selecting the first max_k complete splits aka no error from lba


                        if len(splits) >= max_k: #max k can be a user controlled variable, passed to the CART class 
                            best_index = between_variance.index(max(between_variance))
                            #print(splits, between_variance, variables, best_index )
                            node.set_beta(betas[best_index])
                            node.set_alpha(alphas[best_index]) #np.around(alphas[best_index],4).tolist())
                            node.set_error(errors[best_index])

                            if combination_split:
                                var1, var2 = variables[best_index].split("__")
                                self.n_features[variables[best_index]] = self.n_features[var1] + self.n_features[var2]
                                
                            return variables[best_index], tuple(splits[best_index]), between_variance[best_index]    #"latent_budget_tree doesnt return an error" 
                        else:
                            continue

            elif self.method == "FAST" or self.method == "TWO-STAGE":
                #TODO include entropy, and shannon
                if self.problem == "classifier":
                    ordered_list = self.tau_ordering(node)  
                else:
                    ordered_list = self.pearson_ordering(node) 
                #print("ordered_list",ordered_list)
                k = 0 #iterator 
                while k < len(ordered_list):       #stopping rule 
                    between_variance=[]
                    splits=[]
                    variables=[]
                    if ordered_list[k][1] in self.n_features_names:
                        cat_var = [ordered_list[k][1]]
                        num_var = []
                    else:
                        num_var = [ordered_list[k][1]]
                        cat_var = []
                    for var in cat_var:  
                        #df = pd.DataFrame(self.n_features[str(var)])
                        #print("combinations", var, list(set(self.n_features[str(var)])), "nan" in list(set(self.n_features[str(var)])))
                        #if not df[str(var)].isnull().values.any(): 
                        combinazioni = []
                        distinct_values= []
                        distinct_values.append(list(set(self.n_features[str(var)][node.indexes])))
                        distinct_values = list(itertools.chain(*distinct_values)) #flattens, removed nesting
                        for i in range(1,len(distinct_values)):
                            combinazioni.append(list(itertools.combinations(distinct_values, i)))
                        combinazioni=combinazioni[1:]
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
                                    between_variance.append(tau)
                                elif self.problem == "regression" and self.impurity_fn == "pearson": 
                                    impurities_1.append(impur0)
                                    impurities_1.append(impur1)
                                    between_variance.append(1- sum(impurities_1[t:]) / self.tss(node)) #exploratory slides 43
                                else:
                                    print("Error, Two-Stage and FAST algorithm require impurity_fn as tau for classifier, \
                                          and pearson for regression")
                                    return None
                                splits.append(i)
                                variables.append(str(var))
                                t+=2
                            else:
                                continue
                        #else:
                        #    print("NaN found in observation")
                        #    continue            

                        
                    for var in num_var:
                        #df = pd.DataFrame(self.features[str(var)])
                        #if not df[str(var)].isnull().values.any():  

                        for i in range(len(self.features[str(var)])): #TODO should be set
                                stump = node.bin_split(self.features, self.n_features, str(var),self.features[str(var)][i])
                                if self.y[stump[0].indexes].size >= self.grow_rules['min_cases_child'] \
                                    and self.y[stump[1].indexes].size >= self.grow_rules['min_cases_child']:
                                    impur0 = self.impur(stump[0])
                                    impur1 = self.impur(stump[1])
                                    if self.problem == 'classifier' and self.impurity_fn == "tau":    
                                        gini_parent = self.impur(node)
                                        tau = (impur0 * len(stump[0].indexes) / len(node.indexes) + impur1 * len(stump[1].indexes)/ len(node.indexes) - gini_parent) / (1- gini_parent)
                                        between_variance.append(tau)
                                    elif self.problem == "regression" and self.impurity_fn == "pearson": 
                                        impurities_1.append(impur0)
                                        impurities_1.append(impur1)
                                        between_variance.append(1- sum(impurities_1[t:])/ self.tss(node))
                                    else:
                                        print("Error, Two-Stage and FAST algorithm require impurity_fn as tau for classifier, \
                                          and pearson for regression")
                                        return None
                                    splits.append(self.features[str(var)][i])
                                    variables.append(str(var))
                                    t+=2
                                else: 
                                    continue
                        #else:
                        #    print("NaN found in observation")
                        #    continue 
                    try:                  
                        if k == 0:
                            s_star_k = max(between_variance)  
                            s_star_k_between = between_variance[between_variance.index(max(between_variance))] 
                            s_star_k_split = splits[between_variance.index(max(between_variance))]
                            s_star_k_variables = variables[between_variance.index(max(between_variance))]
                            if self.method == "TWO-STAGE" and max_k == 1:                              
                                return s_star_k_variables, s_star_k_split, s_star_k_between
                    except:
                        k += 1
                        s_star_k = 0
                        continue
                    try:
                        if k != 0 and max(between_variance) > s_star_k:
                            s_star_k = max(between_variance) 
                            s_star_k_between = between_variance[between_variance.index(max(between_variance))]
                            s_star_k_split = splits[between_variance.index(max(between_variance))]
                            s_star_k_variables = variables[between_variance.index(max(between_variance))]
                    except: 
                        k +=1 #failing minimum child size condition
                        continue
                    
                    
                    if self.method == "TWO-STAGE":
                        if max_k == 1:         ##if initial iteration fails to get a result  #len(s_star_k_between) == 1 had previous, but to get to this point cant have error
                            return s_star_k_variables, s_star_k_split, s_star_k_between
                        elif k >= max_k-1:
                            return s_star_k_variables, s_star_k_split, s_star_k_between
                        else:
                            k +=1
                    if self.method == "FAST":
                        if s_star_k < ordered_list[k+1][0] :  #termination for FAST algoirthm
                            k += 1               #only for operation of iteration 
                        else:
                            return s_star_k_variables, s_star_k_split, s_star_k_between
                    
                
                
                try:
                    return s_star_k_variables, s_star_k_split, s_star_k_between #if all fails after all variables 
                except:
                    return None
        

            #had issues with having a boolean predictor 
            elif self.method == "CART":
                for var in self.n_features_names:
                    #print(var)
                    combinazioni = []
                    distinct_values= [] #was np before
                    distinct_values.append(list(set(self.n_features[str(var)])))
                    distinct_values = list(itertools.chain(*distinct_values)) #flattens, removed nesting
                    for i in range(1,len(distinct_values)):
                        combinazioni.append(list(itertools.combinations(distinct_values, i)))
                    combinazioni=combinazioni[1:]
                    combinazioni = list(itertools.chain(*combinazioni))
                    combinazioni = combinazioni +  distinct_values
                    #print(combinazioni)
                    for i in combinazioni: 
                        #print(i)
                        stump = node.bin_split(self.features, self.n_features, str(var),i)
                        #print(len(self.y[stump[0].indexes]))
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
                            t+=2
                            #print(splits[-1], variables[-1], between_variance[-1])
                    else:
                        continue
                        

                #print("self",self.features_names)
                for var in self.features_names:    
                    for i in range(len(self.features[str(var)])):
                            stump = node.bin_split(self.features, self.n_features, str(var),self.features[str(var)][i])
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
                                
                                splits.append(self.features[str(var)][i])
                                variables.append(str(var))
                                t+=2
                            else: 
                                continue
            else:
                print("Method given is not included")
        try:
            #print("max",max(between_variance))
            if self.method == "LATENT-BUDGET-TREE":
                return variables[between_variance.index(max(between_variance))],tuple(splits[between_variance.index(max(between_variance))]),between_variance[between_variance.index(max(between_variance))]
            else:
                return variables[between_variance.index(max(between_variance))],splits[between_variance.index(max(between_variance))],between_variance[between_variance.index(max(between_variance))]
        except:
            #this is mostly an error where the length is less than min size 
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
    

    def control(self):
        for i in self.get_leaf():
            for j in self.get_leaf():
                if i.get_parent() == j.get_parent():
                    if mode(self.y[i.indexes]) == mode(self.y[j.indexes]):
                        #i.set_to_pop()
                        #set_to_pop()
                        self.father_to_pop.append(i.get_parent)
        
    ''' 
    def __ex_devian(self,varian,nodo):
        if self.problem =='regression':
            return varian - len(self.y)*mean(self.y)**2
        elif self.problem == 'classifier':
            
            prop = Counter(self.y[nodo.indexes])
            prop = list(prop.items())
            som = []
            for i in prop:
                som.append((i[1]/len(self.y[nodo.indexes]))**2)
            
            return varian - sum(som)              
    '''     
        
    
    def deviance_cat(self,node):
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
    

    def prop_nodo(self,node):
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
        
        value_soglia_variance = []
        mini_tree = [] 

        self.combination_split = combination_split

        level = node.get_level()
        #print("level",level)
        if level > self.max_level:
            return None 

        #TODO if want to add adaboost, this is where it would go

        #twoing
        if self.twoing:
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
                    if isinstance(i, int): 
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
                            value,soglia,varian = self.__node_search_split(node, max_k, combination_split, max_c)  
                        except TypeError:
                            #print("TypeError [Twoing, pure node after new class assignment]")                    
                            if len(node.indexes) >= self.grow_rules['min_cases_parent']:
                                continue
                            else:
                                self.y = yold
                                return None
                                
                        twoing_value.append(value)
                        twoing_soglia.append(soglia)
                        twoing_varian.append(varian)

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
                    #print(c1[best_index], len(twoing_value), len(self.y[node.indexes]), len(set(self.y[node.indexes])))

                else:
                    self.y = yold
                    return None
                
            
            elif self.problem == "regression":
                
                yold = self.y
                y = pd.DataFrame(self.y[node.indexes]) #hopefully no issues if it is passed as a dataframe
                y.rename(columns = {y.columns[0] : "y"}, inplace= True)
                y["twoing"] = 0

                distinct_values= [set(y["y"])]
                distinct_values = list(itertools.chain(*distinct_values)) #flattens, removed nesting

                '''    
                if twoing_between: #A way to check both codes
                    between_variance =  []
                    for i in distinct_values[0:-1]: #gets rid of last value to avoid errors
                        y_c1 = y[y["y"]<= i] #making an assumption that the groups will be separated in order
                        y_c2 = y[y["y"] > i]
                        if len(y_c1["y"])>0 and len(y_c2["y"])>0:
                            between_variance.append((mean(y_c1["y"])**2)*len(y_c1["y"]) + (mean(y_c2["y"])**2)*len(y_c2["y"]) )
                        else:
                            between_variance.append(0)
                    
                    split_val = distinct_values[between_variance.index(max(between_variance))] #TODO give the c1/c2 class split that maximises between variance of the feature??
                    y["twoing"].loc[y["y"]<= split_val] = "c1"
                    y["twoing"].loc[y["y"]> split_val] = "c2"
                    self.y = y["twoing"]

                    self.problem = "classifier" #changing to classifier after changing classes 
                    if self.method == "CART":
                        self.impurity_fn = "gini"
                    else:
                        self.impurity_fn = "tau"

                    try:
                        value,soglia,varian = self.__node_search_split(node, max_k, combination_split)  
                    except TypeError:
                        print("TypeError [Twoing, pure node after new class assignment]")                    
                        self.y = yold
                        self.problem = "regression"
                        if self.method == "CART":
                            self.impurity_fn = "between_variance"
                        else:
                            self.impurity_fn = "pearson"
                        return None
                
                    self.twoing_c1[node] = "y <= "+str(split_val) 
                    self.twoing_c2[node] = "y > "+str(split_val) 
                    y= pd.DataFrame(y["twoing"]) 
                    y.rename(columns = {y.columns[0] : node.name}, inplace= True) #that way can find it based on the name of the ndoe of the split 
                    self.twoing_y = pd.concat([self.twoing_y, y])
                    
                    self.y = yold
                    self.problem = "regression"     
                    if self.method == "CART":
                        self.impurity_fn = "between_variance"
                    else:
                        self.impurity_fn = "pearson"

                else:
                '''

                twoing_value = []
                twoing_soglia = []
                twoing_varian = [] #either using this to determine best or deviance 
                #if len(c1) > 2: #2 classes will cause node purity checker to proc.  
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
                        value,soglia,varian = self.__node_search_split(node, max_k, combination_split, max_c)  
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
                    #print(c1[best_index], len(twoing_value), len(self.y[node.indexes]), len(set(self.y[node.indexes])))
                    
                    self.problem = "regression"
                    if self.method == "CART":
                        self.impurity_fn = "between_variance"
                    else:
                        self.impurity_fn = "pearson"
                
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
        else:
            try:
                value,soglia,varian = self.__node_search_split(node, max_k, combination_split, max_c)  

            except TypeError:
                print("TypeError")
                return None
            
            #if self.method == "LATENT-BUDGET-TREE":
                #varian = -varian #change put in place to worth with infracture, but want the correct ls value from lba to be printed 

        value_soglia_variance.append([value,soglia,varian,level])
        self.root.append((value_soglia_variance,rout))

        #chunk of appending
        left_node,right_node = node.bin_split(self.features, self.n_features, str(value),soglia)
        node.set_children((left_node,right_node))
        node.set_split(value_soglia_variance)
        node.varian = varian
        mini_tree.append((node,left_node,right_node))
        self.tree.append(mini_tree) 
        self.bigtree.append(node)
        if rout != 'start': #checks the position of the current node, either start, left, right from the previous node
            self.father.append(node) # may be redundant with the same appending happenign below
        self.bigtree.append(node)#append nodo padre
        self.bigtree.append(left_node)#append nodo figlio sinistro
        self.bigtree.append(right_node)#append nodo figlio desto
        
        print("Split Found: ",node.name, value_soglia_variance,rout)

        ###### Calcolo della deviance nel nodo  
        if rout == 'start':
            self.father.append(node)
            if self.problem=='regression':
                #if self.method == "FAST" or self.method == "TWO-STAGE" or self.method == "LATENT-BUDGET-TREE":
                right_varian = len(self.y[left_node.indexes])*(mean(self.y[left_node.indexes])-mean(self.y))**2
                left_varian =len(self.y[right_node.indexes])*(mean(self.y[right_node.indexes])-mean(self.y))**2
                ex_deviance = (right_varian + left_varian) #- len(self.y)*mean(self.y)**2
                
                #else:
                #    ex_deviance = varian - len(self.y)*mean(self.y)**2   
            
            elif self.problem == "classifier":
                ex_deviance = self.deviance_cat(left_node)*len(left_node.indexes)/len(self.y) + self.deviance_cat(right_node)*len(right_node.indexes)/len(self.y)# )/2
                          
        else:
            ex_deviance_list= []
            for inode in self.bigtree:
                if inode not in self.father:
                    if self.problem == 'regression':
                        ex_deviance_list.append(len(self.y[inode.indexes])*(mean(self.y[inode.indexes])-mean(self.y))**2)
                    elif self.problem == 'classifier':
                        ex_deviance_list.append(self.deviance_cat(inode)*len(inode.indexes)/len(self.y))

            if self.problem == "classifier":
                ex_deviance = sum(ex_deviance_list) / len(ex_deviance_list)
            else:
                ex_deviance = sum(ex_deviance_list)
        
        if self.problem == "classifier":
            node_proportion_total = self.devian_y - ex_deviance

        else:
            node_proportion_total = ex_deviance/ self.devian_y   


        #print(ex_deviance, self.devian_y)
        print("node_proportion_total ",node_proportion_total)
        self.node_prop_list.append(node_proportion_total)
        

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
                
        #if self.problem=="regression":S
        if node_proportion_total >= propotion_total: 

            return None
        
        #else: #looks redundant
            #if node_proportion_total >= propotion_total: 
             #   return None
        
        self.nsplit += 1
        return self.growing_tree(left_node,"left",max_k = max_k, combination_split = combination_split, max_c = max_c),self.growing_tree(right_node,"right",max_k = max_k, combination_split = combination_split, max_c = max_c)

    
    def get_key(self, my_dict, val):
        for key, value in my_dict.items():
            if val == value:
                return key
    
        return "key doesn't exist"


    def identify_subtrees(self, father, leaves):
        '''Will associate each node with it's children, grandchildren etc., thus creating subtrees for each node, as if the node was the root'''
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
        '''
        for i in alpha:
            print(i)    
    

    def pop_list(self,lista,lista_to_pop):
        #funzione di pura utilità.
        for i in lista_to_pop:
            lista.pop(lista.index(i))
        return lista


    def alpha_calculator(self,new_dict):
        '''
        Questa funzione ritorna il l'alpha minimo calcolato su un albero di classificazione o regressione,
        il parametro problem : stabilisce il tipo di problema
        valori accettai sono (regression,classification)
        '''
        
        alpha_tmp = []
        deviance = []
        if self.problem == 'regression':
            for key in new_dict: #key  padre
                rt_children__ = []
                rt_father= sum((self.y[key.indexes] - mean(self.y[key.indexes]))**2)
                for figli in new_dict[key][0]:
                    rt_children__.append(sum((self.y[figli.indexes] - mean(self.y[figli.indexes]))**2))
                    deviance.append((self.y[figli.indexes] - mean(self.y[figli.indexes]))**2)
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
        '''
        self.leaf = lista
    
    
    def set_new_leaf(self,lista):
        '''
        Funzione di utilità richiamata dopo il cut
        per ridurre la dimensione dell'albero in termini della quantitò di nodi utilizzati
        come nodi foglia.
        '''
        self.all_node = lista
    

    def miss_classifications(self,list_node):
        
        if self.problem == "classifier":
            
            s = 0
            for node in list_node:
                s += len(self.y[node.indexes])-Counter(self.y[node.indexes]).most_common(1)[0][1] #works
                #for val in self.y[node.indexes]:
                #    if Counter(self.y[node.indexes]).most_common(1)[0][0] != val:
                #        s +=1
           
        elif self.problem == "regression":
            s = 0
            comparison = []
            for node in list_node:
                #s += (mean(self.y[i.indexes])**2)*len(self.y[i.indexes]) #will need changing 
                mean_y = mean(self.y[node.indexes])
                for val in self.y[node.indexes]:
                    s+= (val - mean_y)**2
                    comparison.append([val, mean_y])
            #print("c1",comparison, "s", s, s/len(self.y))
            s = s/len(self.y)
        return s
            
        
    def pruning(self, features_test, n_features_test, y_test):
        '''
        call this function after the growing tree
        perform the pruning of the tree based on the alpha value
        Alfa = #########
        
        per ogni nodo prendi ogni finale prendi i suoi genitori verifica il livello  se è il massimo prendi i genitori
        
        '''
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
               
        
        pruned_trees =[]
        pruned_trees.append([len(leaves), all_node.copy(), leaves.copy()]) #full tree
       
        #Start Pruning Process, continuing until root node
        while len(all_node) >=3:
            
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

            else:
                missclass1 = 0 
                class1 = 0
                missclass2 = 0 
                class2 = 0
                missclass3 = 0 
                class3 = 0
                missclass4 = 0 
                class4 = 0

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

                            if y_test[i] ==1:
                                class1 += 1
                                if self.prediction_cat[-1] != y_test[i]:
                                    missclass1 += 1 
                            if y_test[i] ==2:
                                class2 += 1
                                if self.prediction_cat[-1] != y_test[i]:
                                    missclass2 += 1 
                            if y_test[i] ==3:
                                class3 += 1
                                if self.prediction_cat[-1] != y_test[i]:
                                    missclass3 += 1 
                            if y_test[i] ==4:
                                class4 += 1
                                if self.prediction_cat[-1] != y_test[i]:
                                    missclass4 += 1 

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
                #print("pred_node_dict", pred_node_dict, count, len(y_test))

        if self.problem =='regression':
            print("{leaves : mean square error} = ", leaves_mse)
            minimum = 100000
            key_min = 100000
            for key in leaves_mse:
                if leaves_mse[key] <= minimum:
                    if key < key_min:
                        minimum = leaves_mse[key]
                        key_min = key

            print(f"Best tree for test set has {key_min} leaves with a deviance of: {minimum} ")
            self.graph_results(leaves_for_prune,miss,"Training Set", list(leaves_mse.keys()),list(leaves_mse.values()),"Testing Set")
            
            for i in pruned_trees:
                if i[0] == key_min:
                    tree_table = self.print_tree(i[1], i[2], "CART_tree_pruned.png","tree_pruned.dot", table = True, html = True)

        else:
            print("{leaves : misclassification count} = ", leaves_miss)
            minimum = 10000
            key_min = 10000 
            for key in leaves_miss:
                if leaves_miss[key] <= minimum:
                    if key < key_min:
                        minimum = leaves_miss[key]
                        key_min = key

            print(f"Best tree for test set has {key_min} leaves with misclassification count {minimum} ")           
            self.graph_results(leaves_for_prune,miss,"Training Set", list(leaves_miss.keys()),list(leaves_miss.values()),"Testing Set") #x1, y1, label1, x2, y2, label2

            #leaves for prune - amount of leaves at different cuts
            #miss is values that arent main *** wrong


            #print tree for minkey, and get resulting table
            for i in pruned_trees:
                if i[0] == key_min:
                    tree_table = self.print_tree(i[1], i[2], "CART_tree_pruned.png", "tree_pruned.dot", table = True, html = True)
        

        #make alpha lists
        if self.problem =="classifier":
            for i in range(len(alpha)):
                if alpha[i][1]!=None:
                    result.append((f"Alpha = {alpha[i][0]}",f"value soglia = {alpha[i][1].get_value_thresh()}",f"misclassification = {miss[i]}",f"leaves = {leaves_for_prune[i]}"))
        else:
            for i in range(len(alpha)):
                if alpha[i][1]!=None:
                    result.append((f"Alpha = {alpha[i][0]}",f"value soglia = {alpha[i][1].get_value_thresh()}",f"deviance = {miss[i]}",f"leaves = {leaves_for_prune[i]}"))
        
        '''
        if self.problem =="classifier":
            deviance = 0
            for node in leaves:
                c = Counter(self.y[node.indexes]) #Creates a dictionary {"yes":number, "no"}
                c = list(c.items())
                for i in c:

                    #deviance += 2 * i[1] * math.log10(i[1]/) 
                    p = i[1]/len(self.y[node.indexes])
                    deviance += p * math.log2(p)
            #print(f"WANT TO CHECK Deviance for classification problem {-deviance/len(self.y)} {-deviance} {len(self.y)}")
        ''' 

        return result, tree_table
    
    
    def cut_tree(self,how_many_leaves:int):
        if how_many_leaves>len(self.get_leaf())-1:
            print("error on cut")
            exit(1)
        
        all_node = self.get_all_node()
        leaves = self.get_leaf()
        
        alpha=[]  #(alpha,node) lista degli alpha minimi
        
        while len(self.leaf) != how_many_leaves:
               
            new_dict = self.identify_subtrees(all_node,leaves)
            
            cut = self.alpha_calculator(new_dict)
            alpha.append(cut)  #(alpha,node)
            
            if cut[1] == None:
                break
            
            all_node = self.pop_list(all_node, lista_to_pop = new_dict[cut[1]][1]) #pop on all node
            self.all_node  = all_node
            
            leaves = self.pop_list(leaves, lista_to_pop = new_dict[cut[1]][0]) #pop on leaf
            leaves.append(cut[1])
            self.leaf = leaves


    def build_tree_recursively(self,nodenum, parent_node, parent_children, all_node,leaf_list, leaf_dict, graph, parent_node2):
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
                self.build_tree_recursively(child, child_node, parent_children,all_node,leaf_list, leaf_dict, graph, child_node2)


    def print_tree(self, all_node = None,leaf= None, filename="CART_tree.png", treefile = "tree.dot", table = False, html = False):
        '''Print a visual representation of the formed tree, showing splits at different branches and the mean of the leaves/ terminal nodes.'''

        if not all_node:
            all_node = self.get_all_node()
        if not leaf:
            leaf = self.get_leaf()
              
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
                self.build_tree_recursively(node_num, tree, parent_children,all_node,leaf_list, leaf_dict, graph, tree2) #starts applying parent and child names to respective instances


        #Old print method
        
        for pre, fill, node in RenderTree(tree):                #renders the tree for printing using the RengerTree function from anytree
            print("{}{}".format(pre, node.lines[0]))
            for line in node.lines[1:]:
                print("{}{}".format(fill, line)) 
        
        
        #Dot exporter and dot to png

        try:                              
            DotExporter(tree2).to_dotfile(treefile)   #was tree
            graph.write_png(filename) 
        except: 
            DotExporter(tree2).to_dotfile(treefile)
        '''

        #igraph Graph
        
        nr_vertices = max(father_list)                            # make too many to allow for missing nodes
        v_label = list(map(str, father_list) )                      # create node labels 
        G = Graph.Tree(nr_vertices, 2)                              # 2 stands for children number
        lay = G.layout_reingold_tilford(root=[0])
        position = {k: lay[k-1] for k in father_list}               # assigning nodes to positions , using reigngold layout
        
        
        #new_dict = self.identify_subtrees(self.get_all_node(), self.get_leaf())
        #for i in sorted(position.keys()):
        #    if i ==2:
        #        position[i] = [position[i][0]-self.max_level*2, position[i][1]]
        #        unique_nodes = []
        #        for father in new_dict:
        #            if i==int(father.name[1:]):
        #                for nestedlist in new_dict[father]:
        #                    for childs in nestedlist:
        #                        if childs.name not in unique_nodes:
        #                            unique_nodes.append(childs.name)
        #                            position[int(childs.name[1:])] = [position[int(childs.name[1:])][0]- self.max_level *2, position[int(childs.name[1:])][1]]
#
        #    elif i ==3:
        #        position[i] = [position[i][0]+self.max_level*2, position[i][1]]
        #        unique_nodes = []
        #        for father in new_dict:
        #            if i==int(father.name[1:]):
        #                for nestedlist in new_dict[father]:
        #                    for childs in nestedlist:
        #                        if childs.name not in unique_nodes:
        #                            unique_nodes.append(childs.name)
        #                            position[int(childs.name[1:])] = [position[int(childs.name[1:])][0]+ self.max_level*2 , position[int(childs.name[1:])][1]]
        '''
        Attempts at reorganising the output from the reingold_tilford algorithm for equilength that could then be adjusted 
        max_val = 0 
        for i in sorted(position.keys()):
            if i *2 in position: #aka if it splits
                dist = ((position[i*2][0] - position[i][0])**2+ (position[i*2][1]-position[i][1])**2 )**(1/2)
                if dist > max_val:
                    max_val = dist
            if i *2+1 in position: #
                dist = ((position[i*2+1][0] - position[i][0])**2+ (position[i*2+1][1]-position[i][1])**2 )**(1/2)
                if dist > max_val:
                    max_val = dist

        #print("max_val",max_val)

        new_dict = self.identify_subtrees(self.get_all_node(), self.get_leaf())

        #levels = 0 
        #i = 1 
        #while levels < 10:

            #father_num = sorted(position.keys())[i] #takes the 
            
        for i in sorted(position.keys()):
            print(i, position[i])
 
            if i*2 in position: 
                print("    ",i*2,position[i*2])
                dist = ((position[i*2][0] - position[i][0])**2+ (position[i*2][1]-position[i][1])**2   )**(1/2)
                #print("    dist", dist)

                if dist < max_val: #making them all the same length
                    
                    gradient = (position[i*2][1] - position[i][1])/ (position[i*2][0] - position[i][0])
                    
                    #we want to favour the y coordinate moving more than the x 

                    #method below works but lines cross 
                    print(max_val/dist, math.ceil(max_val/dist))
                    adjuster = math.ceil(max_val/dist) 
                    #if adjuster > 5:
                     #   adjuster = 5
                    new_pos1 = (position[i*2][1] - position[i][1]) * adjuster + position[i][1]
                    b = (max_val**2 - ((position[i*2][1] - position[i][1]) * adjuster)**2 )**0.5
                    if np.sign(position[i][0]) == -1:
                        new_pos0 = np.sign(position[i][0])*b + position[i][0] 
                    else:
                        new_pos0 = -1*np.sign(position[i][0])*b + position[i][0] 

                    print("        ",new_pos0, new_pos1)

                    shift0 = abs(new_pos0) -abs(position[i*2][0])
                    shift1 = abs(new_pos1) - abs(position[i*2][1])
                    
                    #print("        ",shift0,  shift1)
                    
                    position[i*2] = [new_pos0, new_pos1]

                    #position[i*2] = [position[i*2][0]+ np.sign(position[i*2][0])*shift0, position[i*2][1]+shift1]
                    #print("        ",position[i*2])
                    #need to shift all position in the subtree
                    unique_nodes = []
                    for fathers in new_dict:
                        if i*2 == int(fathers.name[1:]):
                            for nestedlist in new_dict[fathers]:
                                for childs in nestedlist:
                                    if childs.name not in unique_nodes:
                                        unique_nodes.append(childs.name)
                                        #if int(childs.name[1:]) %2 == 0:
                                        #print("            ",childs.name, position[int(childs.name[1:])])
                                        position[int(childs.name[1:])] = [position[int(childs.name[1:])][0]+ np.sign(position[int(childs.name[1:])][0])*shift0, position[int(childs.name[1:])][1]+shift1]
                                        #print("            NEWPOS", position[int(childs.name[1:])])
                                        #else:
                                        #   print("else",position[int(childs.name[1:])][0], position[int(childs.name[1:])][1])
                                        #   position[int(childs.name[1:])] = [position[int(childs.name[1:])][0]+shift0, position[int(childs.name[1:])][1]+shift1]
        

            if i*2+1 in position: #trying to make positive go out further to right, while negative more left 
                print("    ",i*2+1,position[i*2+1])
                dist = ((position[i*2+1][0] - position[i][0])**2+ (position[i*2+1][1]-position[i][1])**2 )**(1/2)
                #print("    dist", dist)
            
            
                #elif i*2 in position:
                #    dist = ((position[i*2+1][0] - position[i][0])**2+ (position[i*2+1][1]-position[i][1])**2 )**(1/2)
                if dist < max_val: #making them all the same length
                    print(max_val/dist, math.ceil(max_val/dist))
                    gradient = (position[i*2+1][1] - position[i][1])/ (position[i*2+1][0] - position[i][0])
                    #new_pos0 = (position[i*2+1][0] - position[i][0]) * max_val/dist + position[i][0]
                    #new_pos1 = (position[i*2+1][1] - position[i][1]) * max_val/dist + position[i][1]
                    adjuster = math.ceil(max_val/dist) 
                    #if adjuster > 5:
                    #    adjuster = 5                    
                    
                    new_pos1 = (position[i*2+1][1] - position[i][1]) * adjuster+ position[i][1]
                    b = (max_val**2 - ((position[i*2+1][1] - position[i][1]) *adjuster)**2 )**0.5
                    if np.sign(position[i][0]) == -1:
                        new_pos0 = -1*np.sign(position[i][0])*b + position[i][0]                     
                    else:
                        new_pos0 = np.sign(position[i][0])*b + position[i][0]                     

                    print("        ",new_pos0, new_pos1)
                      
                    shift0 = abs(new_pos0) -abs(position[i*2+1][0])
                    shift1 = abs(new_pos1) - abs(position[i*2+1][1])
                    
                    #print("        ",shift0,  shift1)
                    position[i*2+1] = [new_pos0, new_pos1]
                    #position[i*2+1] = [position[i*2+1][0]+ np.sign(position[i*2+1][0])*shift0, position[i*2+1][1]+shift1]
                    #print("        ",position[i*2+1])
                    #need to shift all position in the subtree
                    unique_nodes = []
                    for fathers in new_dict:
                        if i*2+1 == int(fathers.name[1:]):
                            for nestedlist in new_dict[fathers]:
                                for childs in nestedlist:
                                    if childs.name not in unique_nodes:
                                        unique_nodes.append(childs.name)
                                        #if int(childs.name[1:]) %2 == 0:
                                        #print("            ",childs.name, position[int(childs.name[1:])])
                                        position[int(childs.name[1:])] = [position[int(childs.name[1:])][0]+ np.sign(position[int(childs.name[1:])][0])*shift0, position[int(childs.name[1:])][1]+shift1]
                                        #print("            NEWPOS", position[int(childs.name[1:])])
        print("\n\n")                                
        for i in position:
            print(i)
            if 2*i in position:
                print(((position[i*2][0] - position[i][0])**2+ (position[i*2][1]-position[i][1])**2 )**(1/2))
            if 2*i+1 in position:
                print(((position[i*2+1][0] - position[i][0])**2+ (position[i*2+1][1]-position[i][1])**2 )**(1/2))
        '''

        Y = [lay[k][1] for k in range(len(father_list))] #will need actioning for list 
        M = max(Y)
        es = EdgeSeq(G)                                             # sequence of edges
        E = [e.tuple for e in G.es] # list of edges, connects nodes
        L = len(position)
        Xn = [position[k][0] for k in father_list]
        Yn = [2*M-position[k][1] for k in father_list]

        a = 0
        while a<20:                                                 # When the value is removed it skips to the next index value, jumping, a<10 is just overkill, increased to 20, for really narrow branches 

            for edge in E:   #this is meant to catch the mismateched E's 
                if edge[0] +1 not in position or edge[1]+1 not in position:
                    E.remove(edge) 
            a+=1

        Xe = []
        Ye = []
        
        #think this is where i can interact the positions, using length dependent on the prop gain, can normalise too 
        self.position = position
    
        for edge in E: 
            Xe+=[position[edge[0]+1][0],position[edge[1]+1][0], None]                   # edited for +1 poisiotn as the expected 0 root node it 1 in our dictionary, if index error, increase a 
            Ye+=[2*M-position[edge[0]+1][1],2*M-position[edge[1]+1][1], None]         
        #change labels here, edited to display more information than the node.name
        
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
                                v_label[label] = f"{node.name}, Class: {class_node}, {self.impurity_fn} : {round(self.impur(node, display = True),2)}, Samples : {len(node.indexes)}" 
                            elif self.impurity_fn == "tau":
                                v_label[label] = f"{node.name}, Class: {class_node}, {self.impurity_fn} : None, Samples : {len(node.indexes)}" 
                            else:
                                v_label[label] = f"{node.name}, Class: {class_node}, {self.impurity_fn} : {round(self.impur(node),2)}, Samples : {len(node.indexes)}" 

                        else:
                            mean_y = mean(self.y[node.indexes])
                            v_label[label]=  f"{node.name}, {node.split}, Bin Value: {round(mean_y,2)}, {self.impurity_fn} : {round(self.impur(node),2)}, Samples : {len(node.indexes)}"
                    
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
                                v_label[label] = f"{node.name}, {node.split}, Class:{class_node}, {self.impurity_fn} : {round(self.impur(node, display = True),2)}, Samples: {len(node.indexes)}"
                            elif self.impurity_fn == "tau":
                                v_label[label] = f"{node.name}, {node.split}, Class:{class_node}, {self.impurity_fn} : {round(node.value_soglia_split[0][2],2)}, Samples: {len(node.indexes)}" 
                            else:
                                v_label[label] = f"{node.name}, {node.split}, Class:{class_node}, {self.impurity_fn} : {round(self.impur(node),2)}, Samples: {len(node.indexes)}"
                        else:
                            mean_y = mean(self.y[node.indexes])
                            v_label[label]=  f"{node.name}, {node.split}, Bin Value: {round(mean_y,2)}, {self.impurity_fn} : {round(self.impur(node),2)}, Samples : {len(node.indexes)}"



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
                        opacity=0.8
                        ))
        fig.update_layout(
            title=filename[:-4],    #chops off ".png"
            )
        fig.show()
        if html:
            fig.write_html("cart_tree.html")
            webbrowser.open_new_tab("cart_tree.html")

        



        if table == True and self.method == "LATENT-BUDGET-TREE": 
            if self.twoing:
                tree_table = pd.DataFrame(columns = ["Node", "Node_type", "Variable_Split", "Twoing_Classes_C1", "Twoing_Classes_C2", "n", "Impurity_Value", "Class_Probabilities", "Alpha","Beta", "LS Error" ])

            else:
                tree_table = pd.DataFrame(columns = ["Node", "Node_type", "Variable_Split", "n", "Impurity_Value", "Class_Probabilities", "Alpha","Beta","LS Error" ])
            n1node = self.get_key(father_dict, 1)
            n1index = all_node.index(n1node)

            for node in all_node[n1index:]:
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

                if self.twoing:
                    if self.impurity_fn == "gini":
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split,"Twoing_Classes_C1":[self.twoing_c1[node]], "Twoing_Classes_C2": [self.twoing_c2[node]],"n":len(node.indexes), "Impurity_Value":round(self.impur(node, display = True),2), "Class_Probabilities":class_node, "Alpha":[node.alpha], "Beta":[node.beta],"LS Error":[node.error] })
                    elif self.impurity_fn == "tau":
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split,"Twoing_Classes_C1":[self.twoing_c1[node]], "Twoing_Classes_C2": [self.twoing_c2[node]],"n":len(node.indexes), "Impurity_Value":round(node.value_soglia_split[0][2],2), "Class_Probabilities":class_node, "Alpha":[node.alpha], "Beta":[node.beta],"LS Error":[node.error]})
                    elif self.impurity == "tau":
                                v_label[label] = f"Node: {node.name}, Class: {class_node}, {self.impurity_fn} : {round(node.value_soglia_split[0][2],2)}, Samples : {len(node.indexes)}" 
                    else:
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split,"Twoing_Classes_C1":[self.twoing_c1[node]], "Twoing_Classes_C2": [self.twoing_c2[node]],"n":len(node.indexes), "Impurity_Value":round(self.impur(node),2), "Class_Probabilities":class_node,"Alpha":[node.alpha],"Beta":[node.beta],"LS Error":[node.error]})

                else:
                    if self.impurity_fn == "gini":
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split, "n":len(node.indexes), "Impurity_Value":round(self.impur(node, display = True),2), "Class_Probabilities":class_node, "Alpha":[node.alpha], "Beta":[node.beta],"LS Error":[node.error]})
                    elif self.impurity_fn == "tau":
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split,"n":len(node.indexes), "Impurity_Value":round(node.value_soglia_split[0][2],2), "Class_Probabilities":class_node, "Alpha":[node.alpha], "Beta":[node.beta],"LS Error":[node.error]})
                    else:
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split, "n":len(node.indexes), "Impurity_Value":round(self.impur(node),2), "Class_Probabilities":class_node,"Alpha":[node.alpha],"Beta":[node.beta],"LS Error":[node.error]})
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
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2":None, "n":len(cnode.indexes), "Impurity_Value":round(self.impur(cnode, display = True),2), "Class_Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta], "LS Error":[cnode.error]})
                        elif self.impurity_fn == "tau": #tau is only logged if there is a split 
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2": None,"n":len(cnode.indexes), "Impurity_Value":None, "Class_Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta],"LS Error":[cnode.error]})
                        else:
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2":None, "n":len(cnode.indexes), "Impurity_Value":round(self.impur(cnode),2), "Class_Probabilities":class_node,"Alpha":[cnode.alpha],"Beta":[cnode.beta],"LS Error":[cnode.error]})

                    else:
                        if self.impurity_fn == "gini":
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child","Variable_Split":None, "n":len(cnode.indexes),  "Impurity_Value":round(self.impur(cnode, display = True),2),"Class_Probabilities":class_node, "Alpha":[cnode.alpha],"Beta":[cnode.beta], "LS Error":[cnode.error]} )
                        elif self.impurity_fn == "tau":  
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"n":len(cnode.indexes), "Impurity_Value":None, "Class_Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta], "LS Error":[cnode.error]})
                        else:
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child","Variable_Split":None, "n":len(cnode.indexes),  "Impurity_Value":round(self.impur(cnode),2),"Class_Probabilities":class_node,"Alpha":[cnode.alpha], "Beta":[cnode.cbeta], "LS Error":[cnode.error]})
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
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2":None, "n":len(cnode.indexes), "Impurity_Value":round(self.impur(cnode, display = True),2), "Class_Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta], "LS Error":[cnode.error]})
                        elif self.impurity_fn == "tau": 
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2": None,"n":len(cnode.indexes), "Impurity_Value":None, "Class_Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta], "LS Error":[cnode.error]})
                        else:
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2":None, "n":len(cnode.indexes), "Impurity_Value":round(self.impur(cnode),2), "Class_Probabilities":class_node,"Alpha":[cnode.alpha],"Beta":[cnode.beta], "LS Error":[cnode.error]})
                    else:
                        if self.impurity_fn == "gini":
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child","Variable_Split":None, "n":len(cnode.indexes),  "Impurity_Value":round(self.impur(cnode, display = True),2),"Class_Probabilities":class_node, "Alpha":[cnode.alpha],"Beta":[cnode.beta], "LS Error":[cnode.error]} )
                        elif self.impurity_fn == "tau":  
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"n":len(cnode.indexes), "Impurity_Value":None, "Class_Probabilities":class_node, "Alpha":[cnode.alpha], "Beta":[cnode.beta], "LS Error":[cnode.error]})                        
                        else:
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child","Variable_Split":None, "n":len(cnode.indexes),  "Impurity_Value":round(self.impur(cnode),2),"Class_Probabilities":class_node, "Alpha":[cnode.alpha],"Beta":[cnode.beta], "LS Error":[cnode.error]} )
                    tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
            
            return tree_table
    

        if table == True:
            if self.twoing:
                tree_table = pd.DataFrame(columns = ["Node", "Node_type", "Variable_Split", "Twoing_Classes_C1", "Twoing_Classes_C2", "n", "Impurity_Value", "Class/Value"])
            else:
                tree_table = pd.DataFrame(columns = ["Node", "Node_type", "Variable_Split", "n", "Impurity_Value", "Class/Value"])
            n1node = self.get_key(father_dict, 1)
            n1index = all_node.index(n1node)

            for node in all_node[n1index:]:

                if self.problem == "regression":
                    class_node = round(mean(self.y[node.indexes]),3)
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

                if self.twoing:
                    if self.impurity_fn == "gini":
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split,"Twoing_Classes_C1":[self.twoing_c1[node]], "Twoing_Classes_C2": [self.twoing_c2[node]],"n":len(node.indexes), "Impurity_Value":round(self.impur(node, display = True),2), "Class/Value":[class_node]})
                    elif self.impurity_fn == "tau": 
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split,"Twoing_Classes_C1":[self.twoing_c1[node]], "Twoing_Classes_C2": [self.twoing_c2[node]],"n":len(node.indexes), "Impurity_Value":round(node.value_soglia_split[0][2],2), "Class/Value":[class_node]})
                    else:
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split,"Twoing_Classes_C1":[self.twoing_c1[node]], "Twoing_Classes_C2": [self.twoing_c2[node]],"n":len(node.indexes), "Impurity_Value":round(self.impur(node),2), "Class/Value":[class_node]})

                else:
                    if self.impurity_fn == "gini":
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split, "n":len(node.indexes), "Impurity_Value":round(self.impur(node, display = True),2), "Class/Value":[class_node]}) #class node in brackets or need to pass an index 
                    elif self.impurity_fn == "tau": 
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split,"n":len(node.indexes), "Impurity_Value":round(node.value_soglia_split[0][2],2), "Class/Value":[class_node]})
                    else:
                        new_df = pd.DataFrame({"Node":node.name, "Node_type":"Parent", "Variable_Split":node.split, "n":len(node.indexes), "Impurity_Value":round(self.impur(node),2), "Class/Value":[class_node]})
                
                tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
                
                node_num = int(node.name[1:])
                
                if node_num *2 in leaf_dict.values():
                    cnode = self.get_key(leaf_dict, node_num*2)
                    if self.problem == "regression":
                        class_node = round(mean(self.y[cnode.indexes]),3)
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
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2": None,"n":len(cnode.indexes), "Impurity_Value":round(self.impur(cnode, display = True),2), "Class/Value":[class_node]})
                        elif self.impurity_fn == "tau": 
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2":None,"n":len(cnode.indexes), "Impurity_Value":None, "Class/Value":[class_node]})
                        else:
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2": None,"n":len(cnode.indexes), "Impurity_Value":round(self.impur(cnode),2), "Class/Value":[class_node]})
                    else:
                        if self.impurity_fn == "gini":
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child","Variable_Split":None, "n":len(cnode.indexes),  "Impurity_Value":round(self.impur(cnode, display = True),2),"Class/Value":[class_node]} )
                        elif self.impurity_fn == "tau": 
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"n":len(cnode.indexes), "Impurity_Value":None, "Class/Value":[class_node]})
                        else:
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child","Variable_Split":None, "n":len(cnode.indexes),  "Impurity_Value":round(self.impur(cnode),2),"Class/Value":[class_node]})
                    
                    tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
                
                if node_num *2+1 in leaf_dict.values():
                    cnode = self.get_key(leaf_dict, node_num*2+1)
                    if self.problem == "regression":
                        class_node = round(mean(self.y[cnode.indexes]),3)
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
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2": None,"n":len(cnode.indexes), "Impurity_Value":round(self.impur(cnode, display = True),2), "Class/Value":[class_node]})
                        elif self.impurity_fn == "tau": 
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2":None,"n":len(cnode.indexes), "Impurity_Value":None, "Class/Value":[class_node]})

                        else:
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"Twoing_Classes_C1":None, "Twoing_Classes_C2": None,"n":len(cnode.indexes), "Impurity_Value":round(self.impur(cnode),2), "Class/Value":[class_node]})
                    else:
                        if self.impurity_fn == "gini":
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child","Variable_Split":None, "n":len(cnode.indexes),  "Impurity_Value":round(self.impur(cnode, display = True),2),"Class/Value":[class_node]} )
                        elif self.impurity_fn == "tau": 
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child", "Variable_Split":None,"n":len(cnode.indexes), "Impurity_Value":None, "Class/Value":[class_node]})
                        else:
                            new_df = pd.DataFrame({"Node":cnode.name, "Node_type":"Child","Variable_Split":None, "n":len(cnode.indexes),  "Impurity_Value":round(self.impur(cnode),2),"Class/Value":[class_node]} )
                    tree_table = pd.concat([tree_table, new_df], ignore_index=True, sort=False)
            
            return tree_table
  

    def pred_x(self,node, x, all_node, leaves): #-> tree :
        '''Provides a prediction for the y value (based on the mean of the terminal node), for a new set of unsupervised values'''
                
        #all_node = self.get_all_node()
        #leaves = self.get_leaf()
        
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
                #print("pred_x", class_node, response_dict)

            else:
                self.prediction_reg.append(mean(self.y[node.indexes]))
            #print("pred_node",node.name)

            self.pred_node.append(node.name)

            return node
        
        else:
            if self.combination_split: #not 100% functional, issues with errors in x values, and appears every split leads to a true?
                string2 = node.split.split(" in")[0]
                combinations = string2.split("__")
                split = str(combinations[0])+"__"+str(combinations[1])
                #y = x.copy()
                y = {}
                y[split] = x[combinations[0]] + x[combinations[1]] 
                #print("pruningcomb2",y, node.split)
                #time.sleep(2)
                #print(eval(node.split, y))#y has all this shit in it
                if eval(node.split, y):                 #Evaluates the split for the unsupervised x, whether it is true or not, will deterine if the split goes rigtht or left
                    new_node = self.get_key(node_dict, int(node.name[1:])*2+1)
                    self.pred_x(new_node, x, all_node, leaves) # go to the right child
                else:
                    new_node = self.get_key(node_dict, int(node.name[1:])*2)
                    self.pred_x(new_node, x, all_node, leaves) # go to the left child


            elif eval(node.split, x):                 #Evaluates the split for the unsupervised x, whether it is true or not, will deterine if the split goes rigtht or left
                new_node = self.get_key(node_dict, int(node.name[1:])*2+1)
                self.pred_x(new_node, x, all_node, leaves) # go to the right child
            else:
                new_node = self.get_key(node_dict, int(node.name[1:])*2)
                self.pred_x(new_node, x, all_node, leaves) # go to the left child
    

    def misclass(self, y):
        
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
            print("Deviance", round(mse,2))              


    def prints(self):
        for i in self.get_leaf():
            print(len(self.y[i.indexes]),Counter(self.y[i.indexes]))


    def graph_results(self, x1, y1,  dataset1, x2, y2, dataset2):
        plt.plot(x1, y1, label = dataset1)
        plt.plot(x2, y2, label = dataset2)

        if self.problem =="regression":

            y_label = 'MSE'
        else:
            y_label = 'Misclassification'

        plt.xlabel('Leaves')
        plt.ylabel(y_label)
        plt.title(f"{y_label} vs Leaves for Training and Test Set for {self.impurity_fn}")
        
        plt.legend()
        
        plt.axis([max(x1+x2)*1.05, min(x1+x2)*.95, min(y1+y2)*0.95, max(y1+y2)*1.05])


        plt.show()
        return
    
###########################################
#ADABOOST

# Importing libraries 

'''
class BINPI:

    def __init__(self, df, feature_var, num_var, cat_var, _problem, test, iteration_no = 51):#variables
        self.df = df,
        self.feature_var = feature_var, 
        self.num_var = num_var, 
        self.cat_var = cat_var, 
        self.problem = _problem, 
        self.test = test, 
        self.iteration_no = iteration_no
'''

#Adding initial weights : w = 1/n

def add_weights(df, first = True):
    
    w = [1/ df.shape[0] for i in range(df.shape[0]) ]
    w

    df["weights"] = w
        
    first = False
    
    return df, first

# This is for updating weights, based on whether the observation was correctly predicted

def update_weights(df, alpha, overall_errors):
    
    for i in range(len(df["weights"])):
        if overall_errors[i] == True:
            df["weights"][i] = df["weights"][i] * math.exp(alpha)
        elif overall_errors[i] == False:
            df["weights"][i] = df["weights"][i] * math.exp(-alpha) #correctly classified == False
    
    #normalise
    for i in range(len(df["weights"])):
        df["weights"][i] = df["weights"][i] / df["weights"].sum()
    
    cum_sum = [0]
    for i in range(len(df["weights"])):
        cum_sum.append(df["weights"][i]+cum_sum[-1])
    
    cum_sum.pop(0)
    cum_sum[-1] = 1
    
    df["cum_sum"] = cum_sum
    
    return df
    


#Creates the new dataframe fto be used, based on random sampling, utilising the newly applied weights

def new_df(df):
    random_list =[]
    for i in range(len(df["weights"])):
        random_list.append(random.random())
    
    new_indices = []
    previous_val = 0
    for i in range(len(random_list)):
        for j in range(len(df["weights"])):
            if random_list[i] < df["cum_sum"][j]:
                new_indices.append(j)
                break 
    
   
    new_df = pd.DataFrame(np.zeros(df.shape), columns = df.columns)
    
    count = 0
    for i in  new_indices:
        new_df.iloc[count] = df.iloc[i]
        count +=1
    
    del df
    
    return new_df

def dict_sum(df_series):
    dict_val = {}
    for n in df_series.to_list(): #counting instances of the class
        if n in dict_val:
            dict_val[n] +=1
        else:
            dict_val[n] =1
    return dict_val


#Adaboost algorithm 
def adaboost(df, feature_var, num_var, cat_var, _problem, impur_fn,  method = "CART", weak_learners = 11 , max_level = 0):

    first = True                    #checks whether it is the first iteration (weights = 1/n)
    iterations = 0
    best_weak =[]                   #adds the best weak learnings to a list 
    final_predictions = pd.DataFrame(df[feature_var]) #feature_var is the first missing variable

    while iterations < weak_learners:
        iterations +=1
        print("\nIteration",iterations)
        #print("nan in complete df: ",df.isna().sum().sum())

        if not first:
            df = update_weights(df, alpha, overall_errors[best_index]) 
            df = new_df(df)
            df, first = add_weights(df, first) 

        else:
            df, first = add_weights(df, first)

        #training set
        y = df[feature_var]       #applies feature var (each feature variable is the y to be predicted)
        y_list = y.to_list()
        X = df.drop(labels = [feature_var,"weights", "cum_sum"], axis = 1, errors = "ignore")
        X_num = df[num_var]       #selecting multiple items        
        X_cat = df[cat_var]    
        
        weak_learner = []
        overall_errors =[]
        '''
        for i in X.columns: #must have assumed weak learner only used 1 variable at a time
            print("feature: ",i)
            
            #maybe only makes sense for depth one not depth 3
            if i in num_var:                  #To ensure only one variable is measured
                X_cat_1 = []
                cat_var_1 =[]                 #removing other variable type
                num_var_1 = [i] 
                X_num_1 = df[i]
                X_num_1 = X_num_1.to_frame()        #selecting only one variable from its type 
                dict_val = dict_sum(X_num_1[i])

            elif i in cat_var:
                X_num_1 = []
                num_var_1 = []
                cat_var_1 = [i] 
                X_cat_1 = df[i]
                X_cat_1 = X_cat_1.to_frame()
                dict_val = dict_sum(X_cat_1[i])

            pure = False                            
            for val in dict_val.values():
                if val == len(y_list):
                    pure = True
                    break
            if pure:                                    #issues in running CART with an inseperable dataset - aka already pure 
                print("parent node is pure, requires at least some stratification to be processed by CART")
                break
            my_tree = MyNodeClass('n1', np.arange(len(y)))
            model = CART(y, X_num_1, num_var_1, X_cat_1, cat_var_1, impurity_fn = impur_fn, problem = _problem, method = method, max_level = max_level) 
            model.growing_tree(my_tree)
            prediction_fn(model, y, X_num_1, num_var_1, X_cat_1, cat_var_1)
            overall_errors.append(error_checker(model, y_list, _problem))
        '''

        num_var_1 = num_var.copy()
        cat_var_1 = cat_var.copy()
        if feature_var in num_var:
            num_var_1.remove(feature_var)
        elif feature_var in cat_var:
            cat_var_1.remove(feature_var)


        my_tree = MyNodeClass('n1', np.arange(len(y)))
        model = CART(y, X_num, num_var_1, X_cat, cat_var_1, impurity_fn = impur_fn, problem = _problem, method = method, max_level = max_level) 
        model.growing_tree(my_tree)
        prediction_fn(model, y, X_num, num_var_1, X_cat, cat_var_1)
        overall_errors.append(error_checker(model, y_list, _problem))
        
        
        
        if _problem == "regression":                    
            weak_learner.append([model.prediction_reg[:len(y)], model.get_all_node(), model.get_leaf(), model])  
        else:
            weak_learner.append([model.prediction_cat[:len(y)], model.get_all_node(), model.get_leaf(), model])

        
        error_metric = []
        for error in overall_errors:
            error_metric.append(sum(error))
   
        #if not pure:
        #best = min(error_metric)
        best_index = error_metric.index(min(error_metric))
        colname = "pred"+str(iterations)
        final_predictions[colname] = weak_learner[best_index][0][:len(y)]

        alpha = alpha_calculator(df, overall_errors, best_index, _problem)
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
    for i in best_weak:
        final_model.append([i[-2],i[1], i[0]])
        combined_response.append(i[0][0])
        models.append(i[0][-1])
   
    final_predictions = vote(combined_response, final_predictions,_problem, trains = True)
    final_e = final_error(y_list, final_predictions, _problem)

    if _problem == "classifier":
        print("Final Training Missclassification", sum(final_e), "\n")
    else:
        print("Final Training MSE", round(sum(final_e),2), "\n")

    return {"final_model":final_model, "models":models}

# Prediction funcion

def test_prediction(y_test, models, num_var, cat_var, X_test_num, X_test_cat, _problem):

    test_predictions = pd.DataFrame(y_test)
    
    for model in models:
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
                    

                    #print("sample", dn)
      
                    model.pred_x(node, d, model.get_all_node(), model.get_leaf()) #appending to a list in cart 
        
        if _problem == "classifier":
            weak_learner_test.append(model.prediction_cat[-len(y_test):]) 
        else:
            weak_learner_test.append(model.prediction_reg[-len(y_test):]) 
        colname = "pred"+str(models.index(model))
        test_predictions[colname] = weak_learner_test[0] #is a nested list

    test_predictions = vote(y_test, test_predictions, _problem)

    print("Prediction", test_predictions["final_pred"][0] )

    return test_predictions["final_pred"] 



def vote(y_test, test_predictions, _problem,  trains = False):
    final_pred_test = []    

    if not trains:
        if _problem == "classifier":
            for i in range(len(y_test)):
                votes = []
                for j in range(1,test_predictions.shape[1]):
                    votes.append(test_predictions.iloc[i,j])
                final_pred_test.append(max(set(votes), key = votes.count))     #takes the highest vote
        else:
            for i in range(len(y_test)):
                votes = []
                for j in range(1,test_predictions.shape[1]):
                    votes.append(test_predictions.iloc[i,j])
                final_pred_test.append(mean(votes))       #pretty sure mean vote is right, could be mode
    
    else:
        for i in range(len(y_test[0])):
            votes = []
            for j in range(len(y_test)):
                votes.append(y_test[j][i])
            final_pred_test.append(max(set(votes), key = votes.count))     #takes the highest vote
    
    
    test_predictions["final_pred"] = final_pred_test

    return test_predictions


def  prediction_fn(model, y, X_num_1, num_var_1, X_cat_1, cat_var_1):
    
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
                model.pred_x(node, d, model.get_all_node(), model.get_leaf()) #no return as the values are stores in the cart class 


def error_checker(model, y_list, _problem):
    
    errors = []
    if model.prediction_cat or model.prediction_reg:   #an error checking line 
        for j in range(len(y_list)):
            if _problem == "regression":                     #appears not fuctional, copied from cat
                    errors.append(((y_list[j] -model.prediction_reg[j])**2 )/len(y_list))

            else:
                if model.prediction_cat[j] != y_list[j]:
                    errors.append(True)
                else:
                    errors.append(False)
        
        if _problem == "regression":
            print("training mse", round(sum(errors),2))
        else:
            print("training missclassifications", sum(errors))
        
    else:
        print("THERE MAY BE AN ISSUE")
        errors = [True]*len(y_list)

    return errors
     

def alpha_calculator(df, overall_errors, best_index, _problem):
    TE = 0 #total error 
    for i in range(len(df["weights"])):
        if _problem == "classifier":
            TE += df["weights"][i] * overall_errors[best_index][i] 
        else:
            TE += df["weights"][i] * overall_errors[best_index][i] / (max( overall_errors[best_index]) - min( overall_errors[best_index])) #may need a different variation for regression, needs to be probability

    alpha = 0.5 * np.log ( ((1-TE) / TE) + 1e-7)
    return alpha



def final_error(y_list, final_predictions, _problem):
    final_miss =[]
    if _problem == "classifier":
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


def id_matrix_creator(df):

    id_matrix = df.notna()

    id_matrix.replace(True, "a", inplace = True)
    id_matrix.replace(False, "b", inplace = True)

    id_matrix.replace("a", 0, inplace = True)
    id_matrix.replace("b", 1, inplace = True)

    id_matrix.head()

    return id_matrix

def row_vector(id_matrix):

    row_vect = []

    for variable_name in id_matrix.columns:
        row_vect.append((id_matrix[variable_name].sum(axis=0), variable_name ))

    row_vect.sort()
    #print(row_vect)

    row_name_vect = []

    for i in row_vect:
        row_name_vect.append(i[1])

    return row_name_vect, row_vect


def condition(element):
    '''sort list by amount of missing, and then by alphabetical order of variable name'''

    return element[0], element[2]



def column_vector(id_matrix):

    column_vect = []
    for row_number in range(id_matrix.shape[0]):
        column_vect.append([id_matrix.iloc[row_number].sum(axis=0), row_number])

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
            for variable_name in id_matrix.columns:
                position +=1
                if id_matrix.iloc[values[1], position-1] == 1:
                    list_var.append(variable_name)
            values.append(list_var)
            
    column_vect.sort(key = condition)
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
def lexographical_matrix(df, id_matrix):

    row_name_vector, row_vect = row_vector(id_matrix)
    column_number_vector, column_vect = column_vector(id_matrix)
    df2 = df.reindex(columns= row_name_vector)
    df2 = df2.reindex(column_number_vector)

    return df2, column_vect

def first_nan(df2, column_vect, last_nan = 0):

    #if you wanted to find the maximal size when it wasnt so obvious, do it during this step, and pass column_vect 
    skip_point = (False, 0, 0, 0)
    row_no = last_nan                    #skip full iteration, and start with last nan only
    
    for pair in column_vect[last_nan:]:
        if pair[0] > 0: 
            if pair[0] >1:
                skip_point = (True,  pair[0], pair[1], pair[2])
            
            column_vect[row_no] = (pair[0]-1, pair[1], pair[2]) #adjust the list removing the to be imputated value
            last_nan = row_no
            break
        else:
            first = None
        row_no +=1

    return row_no, column_vect, last_nan, skip_point

def matches_dict(column_vect):
  
    dict_match = {}
    for i in column_vect:
        if " ".join(i[2]) in dict_match:
            dict_match[" ".join(i[2])] +=1
        else:
            dict_match[" ".join(i[2])] = 1

    return dict_match 

def checkNaN(str):
    try:
        return math.isnan(float(str))
    except:
        return False
    

def feature_variable(df2, row_no):

    row_no
    pos = 0
    feature_var = "a"
    for value in df2.iloc[row_no]:
        pos += 1

        if checkNaN(value):            #this only looks for the first instance 
            feature_var = df2.columns[pos-1]
            break
    return feature_var, pos


#checks for the 3 types of imputation processes 

def imputation_process(df2, feature_var, row_no, pos, num_var, bin_var, class_var, weak_learners, old_model="",  previous_var= ""):

    complete_df = df2.iloc[0:row_no].copy()  #subset only the complete dataset
    complete_df.reset_index(drop = True, inplace = True)  
    X = complete_df.drop(feature_var, axis = 1)
    cat_var = bin_var + class_var
    #As a temporary fix for multiple missing values, will use mean imputation for a secondary, tertiary etc missing value temporarily 
    prediction_feat = df2.iloc[row_no].copy()
    #print("prediction_feat2\n",prediction_feat)
    prediction_feat.drop(feature_var, inplace = True)
    
    for series_name in X.columns:                                  
        if checkNaN(prediction_feat[series_name]):
            if series_name in cat_var:
                prediction_feat[series_name] = Counter(X[series_name][X[series_name].notna()]).most_common(1)[0][0] 
                #print("prediction_feat", series_name ,Counter(X[series_name][X[series_name].notna()]).most_common(1)[0][0])
            else:
                prediction_feat[series_name] = round(mean(X[series_name][X[series_name].notna()]),0)


    y_test = [1] #one element list, as test is for prediction 

    #removing feature_var from appropiate list, before indexing
    if feature_var in num_var:
        num_var_full = num_var.copy()
        num_var_1 = num_var.copy()
        cat_var_1 = cat_var.copy()
        num_var_1.remove(feature_var)

    elif feature_var in cat_var:
        num_var_full = num_var.copy()
        num_var_1 = num_var.copy()
        cat_var_1 = cat_var.copy()
        cat_var_1.remove(feature_var) 
    else:
        print("Variable Error", feature_var, num_var, cat_var)

    X_test_num = prediction_feat[num_var_1]  
    X_test_cat = prediction_feat[cat_var_1] 
    
    imp_time_start = time.time()

    if feature_var in num_var_full:
        if feature_var != previous_var:
            #don't think it matters if i pass num_var_full or num_var as there is filtering later 
            model = adaboost(df = complete_df, feature_var = feature_var, num_var = num_var, cat_var = cat_var,_problem =  "regression", weak_learners = weak_learners, impur_fn = "pearson", method = "FAST")  
            yhat = test_prediction(y_test, model["models"], num_var_1, cat_var_1, X_test_num, X_test_cat, _problem =  "regression") 
        else:
            yhat = test_prediction(y_test, old_model["models"], num_var_1, cat_var_1, X_test_num, X_test_cat,"regression")
            model = old_model
    
    elif feature_var in bin_var:
        if feature_var != previous_var:
            model = adaboost(df = complete_df, feature_var = feature_var, num_var = num_var, cat_var = cat_var,_problem =  "classifier", weak_learners = weak_learners, impur_fn = "tau", method = "FAST") ##does it need test set passed 
            yhat = test_prediction(y_test, model["models"], num_var_1, cat_var_1, X_test_num, X_test_cat,_problem =  "classifier")
        else:
            yhat = test_prediction(y_test, old_model["models"], num_var_1, cat_var_1, X_test_num, X_test_cat, _problem =  "classifier")
            model = old_model

    elif feature_var in class_var:
        if feature_var != previous_var:
            model = adaboost(df = complete_df, feature_var = feature_var, num_var = num_var, cat_var = cat_var,_problem =  "classifier", weak_learners = weak_learners, impur_fn = "tau",method = "FAST", max_level = 3)  
            yhat = test_prediction(y_test, model["models"], num_var_1, cat_var_1, X_test_num, X_test_cat, _problem =  "classifier")
        else:
            yhat = test_prediction(y_test, old_model["models"], num_var_1, cat_var_1, X_test_num, X_test_cat, _problem =  "classifier")
            model = old_model
    else: 
        print("Error, found variable missing from variable lists")

    #Applying the value to the dataset
    df2.iloc[row_no, pos-1] = yhat[0]

    previous_var = feature_var #used for reusing the model 

    print("imp time", time.time() - imp_time_start)
    return model, previous_var

def binpi_imputation(df2,column_vect, num_var, bin_var, class_var, weak_learners):
   

    #Future adaption - for a dataset with no complete area, need to impute the least missing column with a simple method, mean mode, andrea frazzoni
    dict_match = matches_dict(column_vect)

    last_nan = 0
    iteration = 0 
    while df2.isna().any().any() > 0: 
   
        start = time.time()
        iteration +=1
 
        row_no, column_vect, last_nan, skip_point = first_nan(df2, column_vect, last_nan)       #finds first nan
        feature_var, pos = feature_variable(df2, row_no)

        if skip_point[0]: #checks if can reuse model 

            #feature_var, pos = feature_variable(df2, row_no)

            for i in range(dict_match[" ".join(skip_point[3])]):     

                if iteration >1:
                    model_1, previous_var_1 = imputation_process(df2, feature_var, row_no, pos,  num_var, bin_var, class_var, weak_learners, old_model,  previous_var)
                else:
                    model_1, previous_var_1 = imputation_process(df2, feature_var, row_no, pos,  num_var, bin_var, class_var, weak_learners)

                old_model,  previous_var =  model_1, previous_var_1

                print("time", time.time() - start)
                iteration +=1

                if i >0:

                    column_vect[row_no] = (skip_point[1]-1, skip_point[2], skip_point[3]) # for multi missing points, to stop it from going back in
         
                row_no+=1
            continue


        #feature_var, pos = feature_variable(df2, row_no)

        print("\nFeature Variable: ", feature_var, "\nMissing Values: ", df2.isna().sum().sum())

        if iteration >1:
            model_1, previous_var_1 = imputation_process(df2, feature_var, row_no, pos,  num_var, bin_var, class_var, weak_learners, old_model,  previous_var)
        else:
            model_1, previous_var_1 = imputation_process(df2, feature_var, row_no, pos,  num_var, bin_var, class_var, weak_learners)

        old_model,  previous_var =  model_1, previous_var_1
        print("time", time.time() - start)

    return df2
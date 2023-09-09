#please see lba article heijden 1989 for more info


alpha = matrix( data = c(0.76, 0.24, 
                         0.78, 0.22, 
                         0.60, 0.40, 
                         0.50, 0.50, 
                         0.29, 0.71, 
                         0.11, 0.89), 
                         nrow = 6, 
                         ncol = 2, 
                         byrow = TRUE  )



beta = matrix( data = c(0.29, 0.39,0.21,0.11, 
                        0.07, 0.34, 0.23, 0.36),
               byrow = TRUE, 
               nrow = 2)




alpha[,1] %*% t(beta[1,])
alpha[,2] %*% t(beta[2,])
alpha[,1] %*% t(beta[1,]) + alpha[,2] %*% t(beta[2,])


##############################

#library : "lba"
library(lba)

#lba(base.as_matrix(cont_table), K = 2 , what = 'outer', method = 'ls')


data('PerfMark') 
ex2 <- lba(as.matrix(PerfMark),
           K = 2,
           what='outer',
           trace.lba = FALSE)
ex2


A <- ex2$A #this has the A parameters for splitting (is a matrix that can be returned )

A

B <- ex2$B
t(B)
ex2$val_func
#performs 100 iterations for convergence 



# Tau index 
#exactly the same as our calcs
library(GoodmanKruskal)

datatraining = read.csv("data_training.csv")

vars = c('travel_mode', 'purpose', 'fueltype', 'faretype', 'bus_scale','travel_month', 'day_of_week', 'female', 'driving_license','car_ownership', 'pt_interchanges', 'age_group', 'distance_group', 'starttime_group', 'cost_group', 'costdrivingfuel_group',      'trafficperc_group', 'drivingcharge_group')


for (i in 1:length(vars)){
  print(vars[i])
  print(GKtau(datatraining[,vars[i]], datatraining[,"travel_mode"], dgts = 3, includeNA = "ifany")
)
}


##############################


# Plotting Classification Trees with the plot.rpart and rattle pckages

library(rpart)				        # Popular decision tree algorithm
library(rattle)					# Fancy tree plot
library(rpart.plot)				# Enhanced tree plots
library(RColorBrewer)				# Color selection for fancy tree plot
library(party)					# Alternative decision tree algorithm
library(partykit)				# Convert rpart object to BinaryTree
library(caret)					# Just a data source for this script
# but probably one of the best R packages ever. 
data(segmentationData)				# Get some data
data <- segmentationData[,-c(1,2)]

# Make big tree
form <- as.formula(Class ~ .)
tree.1 <- rpart(form,data=data,control=rpart.control(minsplit=20,cp=0))
# 
plot(tree.1)					# Will make a mess of the plot
text(tree.1)
# 
prp(tree.1)					# Will plot the tree
prp(tree.1,varlen=3)				# Shorten variable names

# Interatively prune the tree
new.tree.1 <- prp(tree.1,snip=TRUE)$obj # interactively trim the tree
prp(new.tree.1) # display the new tree
#
#-------------------------------------------------------------------
tree.2 <- rpart(form,data)			# A more reasonable tree
prp(tree.2)                                     # A fast plot													
fancyRpartPlot(tree.2)				# A fancy plot from rattle
#
#-------------------------------------------------------------------
# Plot a tree built with RevoScaleR
# Construct a model formula
sdNames <- names(segmentationData)
X <- as.vector(sdNames[-c(1,2,3)])
form <- as.formula(paste("Class","~", paste(X,collapse="+")))
# Run the model
rx.tree <- rxDTree(form, data = segmentationData,maxNumBins = 100,
                   minBucket = 10,maxDepth = 5,cp = 0.01, xVal = 0)
# Plot the tree						
prp(rxAddInheritance(rx.tree))
fancyRpartPlot(rxAddInheritance(rx.tree))


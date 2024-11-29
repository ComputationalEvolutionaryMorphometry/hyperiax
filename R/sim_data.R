# install.packages(c("corHMM", "phytools")
library("corHMM")
library("phytools")

#-------------- Sim Data

# simulate tree with 50 taxa
tree<-pbtree(n=50, scale=100, b=1, d=0)
plot(tree)


# make rate matrix Q
Q <- matrix(
  c(
    -0.03, 0.03,
    0.1, -0.1
  ), 2,2, byrow = T)
Q


# simulate character on tree
hist <- sim.history(tree, Q, nsim=1)
plot(hist)


#-------------- Save Data
# Uncomment to use

# # tree
# write.tree(tree, file='data/tree.tre')
# # character
# taxa <- cbind(hist$tip.label, hist$states)
# write.table(taxa, file = "data/character.txt", sep = "\t", row.names = F, col.names = F, quote = FALSE)


#-------------- read in python as:
# import pandas as pd
# 
# # Read the tab-delimited file
# data = pd.read_csv("R/data/character.txt", sep="\t", header=None)
# 
# # Display the data
# print(data)

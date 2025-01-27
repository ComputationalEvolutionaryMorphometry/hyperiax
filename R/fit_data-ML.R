# install.packages(c("corHMM", "phytools")
library("corHMM")
library("phytools")

# Read data

tree <- read.tree(file='data-R/tree.tre')
plot(tree)

taxa <- read.table("data-R/character.txt", sep = "\t", header = FALSE, stringsAsFactors = FALSE)
taxa

#------ Run ML

# 2-rate symmetrical model
Q.Asym <- matrix(
  c(
    NA, 1,
    2, NA
  ), 2,2, byrow = T)
Q.Asym


# Inference
#taxa <- cbind(hist$tip.label, hist$states)
Recon_Q.Asim <- rayDISC(tree, taxa, rate.mat=Q.Asym, node.states="marginal", 
                        model="ARD", root.p="maddfitz",verbose=T)


# plot ANS
plotRECON(tree, Recon_Q.Asim$states, piecolors=c('black', 'red'), title="1-rate Model", pie.cex=0.5)


# infered rate matrix
print(Recon_Q.Asim)



#--------------------- Result
# Fit
#     -lnL     AIC     AICc ntax
# -31.7373 67.4746 67.72991   50
# 
# Rates
#           1        2
# 1        NA 0.110423
# 2 0.1905973       NA
# 
# Arrived at a reliable solution 


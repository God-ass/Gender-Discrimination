library(rpart)
library(rpart.plot)
library(nnet)
library(randomForest)


# Investigate the substantive issue in the real world case: Gender Discrimination in the workplace. 
# Did female doctors get lower pay and rank due to gender?


# Explore dataser with clustering methods and suitable visualization.
setwd("~/Year3/ML/L_project")
data1 <- read.csv("Lawsuit.csv")

# Scatterplot Matrix with smooth curves
pairs(~ . , panel=panel.smooth, data = data1, main = "Scatterplot Matrix of Lawsuit Data")
# Some Findings in from the plots: 
# Sal94 and Sal95 have very high correlation.
# Promotion info not in data.
# Some Dept has consistently high salary.
# Fewer Females than Males and top Salaries belong to Males.
# Top salaries with Clinical Emphasis.
# Top salaries with Board Certified.

#However, there is no information on Promotion. 
# thus, focus on Sal94. Sal95 is highly correlated.


# Discard ID and Sal95 in analysis
data2 <- data1[,2:9]


# PCA
pc <- prcomp(data2, scale.=T) 
summary(pc)
# First two principal components capture 70% of variance. 
pc$rotation
# Gender is relatively not important in PC1 but relatively important in PC2.
# PCA concludes that Gender is important differentiator.


data2.scaled <- scale(data2)


# K Means Clustering
set.seed(2020)
k2 <- kmeans(data2.scaled, centers=2)  # set k = 2 to see natural clusters of 2.
summary(k2)
k2results <- data.frame(data2$Gender, data2$Sal94, k2$cluster)
cluster1 <- subset(k2results, k2$cluster==1)
cluster2 <- subset(k2results, k2$cluster==2)
cluster1$data2.Gender <- factor(cluster1$data2.Gender)
cluster2$data2.Gender <- factor(cluster2$data2.Gender)

summary(cluster1$data2.Sal94)
summary(cluster2$data2.Sal94)
# Cluster 1 has higher salary than cluster 2.


round(prop.table(table(cluster1$data2.Gender)),2)
round(prop.table(table(cluster2$data2.Gender)),2)
# 67% in Cluster 1 are Males, 50% in Cluster 2 are Males.


# Goodness of Fit Test
# Is Cluster 1 statistically same as Cluster 2 in terms of Gender?
M <- as.matrix(table(cluster1$data2.Gender))
p.null <- as.vector(prop.table(table(cluster2$data2.Gender)))
chisq.test(M, p=p.null)
# Cluster 1 Gender Proportions are different from Cluster 2 Gender Proportions
# K-means clustering concludes Gender is significant differentiator.


# Hierarchical Clustering
hc.average =hclust(dist(data2.scaled), method ="average")

plot(hc.average , main ="Average Linkage", xlab="", sub ="", cex =.9)
sum(cutree(hc.average , 2)==2)  # 2
# Average linkage fails to provide sufficient sample size for one cluster.

hc.complete =hclust(dist(data2.scaled), method ="complete")
plot(hc.complete , main ="Complete Linkage", xlab="", sub ="", cex =.9)
sum(cutree(hc.complete, 2)==2)  # 159 cases of second cluster
hc.cluster1 <- subset(k2results, cutree(hc.complete, 2)==1)
hc.cluster2 <- subset(k2results, cutree(hc.complete, 2)==2)

hc.cluster1$data2.Gender <- factor(hc.cluster1$data2.Gender)
hc.cluster2$data2.Gender <- factor(hc.cluster2$data2.Gender)

summary(hc.cluster1$data2.Sal94)
summary(hc.cluster2$data2.Sal94)
# Cluster 2 has higher salary than cluster 1.


round(prop.table(table(hc.cluster1$data2.Gender)),2)
round(prop.table(table(hc.cluster2$data2.Gender)),2)
# 62% in Cluster 2 are Males, 55% in Cluster 1 are Males.

# Goodness of Fit Test
# Is hc.cluster 2 statistically same as hc.cluster 1 in terms of Gender?
M <- as.matrix(table(hc.cluster2$data2.Gender))
p.null <- as.vector(prop.table(table(hc.cluster1$data2.Gender)))
chisq.test(M, p=p.null)
# Cluster 2 Gender Proportions are similar statistically from Cluster 1 Gender Proportions
# Hierarchical Clustering concludes that Gender is insignificant differentiator between the 2 clusters.


# Investigate the data via two regression technique
# Linear Regression with Sal94 outcome
data2.dum <- data2
data2.dum$Dept <- factor(data2.dum$Dept)
data2.dum$Gender <- factor(data2.dum$Gender)
data2.dum$Clin <- factor(data2.dum$Clin)
data2.dum$Cert <- factor(data2.dum$Cert)
data2.dum$Rank <- factor(data2.dum$Rank)

m.lin <- lm(Sal94 ~ ., data = data2.dum)
rmse.linreg <- round(sqrt(mean(residuals(m.lin)^2)),0)
summary(m.lin)
# Gender is statistically insignificant. Dept, Cert, Exper, Clin, Rank are significant.


# CART with Sal94 outcome
set.seed(2)
m.cart <- rpart(Sal94 ~ ., data = data2.dum, method = 'anova', control = rpart.control(minsplit = 2, cp = 0))
printcp(m.cart)
plotcp(m.cart)

#[Optional] Extract the Optimal Tree via code instead of eye power
  # Compute min CVerror + 1SE in maximal tree m.cart.
CVerror.cap <- m.cart$cptable[which.min(m.cart$cptable[,"xerror"]), "xerror"] + m.cart$cptable[which.min(m.cart$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree m.cart.
i <- 1; j<- 4
while (m.cart$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(m.cart$cptable[i,1] * m.cart$cptable[i-1,1]), 1)

#  i = 13 shows that the 13th tree is optimal based on 1 SE rule.

# Prune the max tree using a particular CP value
m.cart2 <- prune(m.cart, cp = cp.opt)

rmse.cart1 <- round(sqrt(mean((data2.dum$Sal94 - predict(m.cart))^2)),0)

rmse.cart2 <- round(sqrt(mean((data2.dum$Sal94 - predict(m.cart2))^2)),0)

#  Unpruned CART has overfitted the trainset but we are not predicting future cases but only explaning historical data.

m.cart$variable.importance
#  Dept, Prate, Clin, Experience, Board-Certified are more important than Gender in explaining Salary. 

# Hence, Gender is not a significant predictor of Salary. Hence, no gender discrimination on Salary.


# Investigate data via two classification technique
# Logistic Regression: Is Distribution of rank associated statistically with Gender in the absence of salary?
m.log <- multinom(Rank ~ . -Sal94 , data = data2.dum)
summary(m.log)
OR.CI <- exp(confint(m.log))
OR.CI

#  Gender is statistically significant in distribution of rank but does not necessarily mean promotion discriminatory. Hiring decision?
# CART on Rank
set.seed(2)
m.cart.rk <- rpart(Rank ~ . -Sal94, data = data2.dum, method = 'class', control = rpart.control(minsplit = 2, cp = 0))
printcp(m.cart.rk)
plotcp(m.cart.rk)

# Using the maximal tree as objective is explaining historical data, not predicting future cases.
m.cart.rk$variable.importance
# Experience, Prate and Dept are more important than Gender in explaining Rank.


#  Conclusions:
#  Insufficient evidence of Gender Discrimination on salary.
#  No information to determine Promotion bias as only the current rank is given.
#  Evidence of Gender Discrimination on Rank is mixed. i.e. not conclusive.
#  Dept and Experience are far more important than Gender.

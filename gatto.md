```r
library(stats)

# 4 Unsupervised Learning (UML)
# 4.2 https://lgatto.github.io/IntroMachineLearningWithR/unsupervised-learning.html#k-means-clustering
# k-means clustering
data("iris")

i <- grep("Length", names(iris))
x <- iris[, i]
cl <- kmeans(x, centers = 3, nstart = 10)
plot(x, col = cl$cluster)

# 4.2.1 how does k means work?
set.seed(12)
init <- sample(3, nrow(x), replace = TRUE)
plot(x, col = init)

par(mfrow = c(1,2))
plot(x, col = init)
centres <- sapply(1:3, function(i) colMeans(x[init == i, ])) # finding centres
centres <- t(centres)
points(centres[, 1], centres[, 2], pch = 19, col = 1:3)

tmp <- dist(rbind(centres, x))
tmp <- as.matrix(tmp)[, 1:3]

ki <- apply(tmp, 1, which.min)
ki <- ki[-(1:3)]

plot(x, col = ki)
points(centres[, 1], centres[, 2], pch = 19, col = 1:3)

# 4.2.2 model selection
cl1 <- kmeans(x, centers = 3, nstart = 10)
cl2 <- kmeans(x, centers = 3, nstart = 10)
table(cl1$cluster, cl2$cluster)

cl1 <- kmeans(x, centers = 3, nstart = 1) # dif k means w/ nstart = 1
cl2 <- kmeans(x, centers = 3, nstart = 1)
table(cl1$cluster, cl2$cluster)

set.seed(42) # dif seed
xr <- matrix(rnorm(prod(dim(x))), ncol = ncol(x))
cl1 <- kmeans(xr, centers = 3, nstart = 1)
cl2 <- kmeans(xr, centers = 3, nstart = 1)
table(cl1$cluster, cl2$cluster)

diffres <- cl1$cluster != cl2$cluster # find differences in two clusters
par(mfrow = c(1,2))
plot(xr, col = cl1$cluster, pch = ifelse(diffres, 19, 1))
plot(xr, col = cl2$cluster, pch = ifelse(diffres, 19, 1))

# 4.2.3 how to determine number of clusters
ks <- 1:5
tot_within_ss <- sapply(ks, function(k) {
  cl <- kmeans(x, k, nstart = 10)
  cl$tot.withinss
})
plot(ks, tot_within_ss, type = "b") # choose near elbow

# 4.3 https://lgatto.github.io/IntroMachineLearningWithR/unsupervised-learning.html#hierarchical-clustering
# heirarchical clustering
d <- dist(iris[, 1:4])
hcl <- hclust(d)
hcl
plot(hcl)
cutree(hcl, h = 1.5)
cutree(hcl, k = 2)

# 4.3.2 defining clusters
plot(hcl)
abline(h = 3.9, col = "red")

km <- kmeans(iris[, 1:4], centers = 3, nstart = 10)
hcl <- hclust(dist(iris[, 1:4]))
table(km$cluster, cutree(hcl, k = 3))

par(mfrow = c(1, 2))
plot(iris$Petal.Length, iris$Sepal.Length, col = km$cluster, main = "k-means")
plot(iris$Petal.Length, iris$Sepal.Length, col = cutree(hcl, k = 3), main = "Heirarchical Clustering")

table(iris$Species, km$cluster)
table(iris$Species, cutree(hcl, k = 3))

# 4.4 https://lgatto.github.io/IntroMachineLearningWithR/unsupervised-learning.html#pre-processing
# pre-processing
colMeans(mtcars)
hcl1 <- hclust(dist(mtcars))
hcl2 <- hclust(dist(scale(mtcars))) # scaled version

par(mfrow = c(1, 2))
plot(hcl1, main = "original data")
plot(hcl2, main = "scaled data")

# 4.5 https://lgatto.github.io/IntroMachineLearningWithR/unsupervised-learning.html#principal-component-analysis-pca
# principal component analysis (PCA)
pairs(iris[, -5], col = iris[, 5], pch = 19)

irispca <- prcomp(iris[, -5])
summary(irispca) # proportion of variance is important !

# 4.5.2 visualization
biplot(irispca)

var <- irispca$sdev^2 # extract sd
pve <- var / sum(var)
cumsum(pve)

par(mfrow = c(1, 2))
plot(irispca$x[, 1:2], col = iris$Species)
plot(irispca$x[, 3:4], col = iris$Species)

# 4.5.3 data pre-processing
par(mfrow = c(1, 2))
biplot(prcomp(mtcars, scale = FALSE), main = "no scaling") # prcomp comp scaling vs no scaling
biplot(prcomp(mtcars, scale = TRUE), main = "scaling")

# 4.6 https://lgatto.github.io/IntroMachineLearningWithR/unsupervised-learning.html#data-pre-processing
# t-distributed stochastic neighbour embedding (t-SNE)
library("Rtsne")
uiris <- unique(iris[, 1:5]) # remove duplicates
iristsne <- Rtsne(uiris[, 1:4])
plot(iristsne$Y, col = uiris$Species)


# -------------------------------------------------


# 5 Supervised Learning (SML)
# 5.2 https://lgatto.github.io/IntroMachineLearningWithR/supervised-learning.html#preview
# preview
library(class)

set.seed(12L)
tr <- sample(150, 50) # select sample
nw <- sample(150, 50)

knnres <- knn(iris[tr, -5], iris[nw, -5], iris$Species[tr]) # k nearest neighbors (KNN)
head(knnres)

table(knnres, iris$Species[nw]) # compare pred, actual
mean(knnres == iris$Species[nw])

args(knn) # could maybe have higher k to look at more neighbours for inference
knnres5 <- knn(iris[tr, -5], iris[nw, -5], iris$Species[tr], k = 5) # knn with k = 5
mean(knnres5 == iris$Species[nw])
table(knnres5, knnres) # compare knn for k = 5, k = 1

knnres5prob <- knn(iris[tr, -5], iris[nw, -5], iris$Species[tr], k = 5, prob = TRUE) # knn with prob = true and k > 1
mean(knnres5prob == iris$Species[nw])
table(attr(knnres5prob, "prob"))

# 5.3 https://lgatto.github.io/IntroMachineLearningWithR/supervised-learning.html#model-performance
# model performance

library(caret)
data(diamonds)

model <- lm(price ~ ., diamonds) # model price of diamonds
p <- predict(model, diamonds) # predict price of diamonds

# in-sample rmse
error <- p -diamonds$price # error = predicted - actual
rmse_in <- sqrt(mean(error^2)) # in-sample rmse
rmse_in

# out-of-sample rmse (80/20 split)
set.seed(42)
ntest <- nrow(diamonds) * .8
test <- sample(nrow(diamonds), ntest) # sample training set (80% of data)
model <- lm(price ~ ., diamonds[test, ]) # model price on training set
p <- predict(model, diamonds[-test, ])
error <- p - diamonds$price[-test]
rmse_out <- sqrt(mean(error^2)) # out-of-sample rmse
rmse_out

# 5.3.2 cross-validation
set.seed(42)
model <- train(price ~., diamonds, # cross-validation: creating 10 folds, running linear model
               method = "lm",
               trControl = trainControl(method = "cv",
                                        number = 10,
                                        verboseIter = TRUE))
model

p <- predict(model, diamonds)
error <- p - diamonds$price
rmse_xval <- sqrt(mean(error^2)) #rmse of cross-validation trained lm
rmse_xval

# another example of cross-validation
library(MASS)
data(Boston)
model <- train(medv ~., Boston,
               method = "lm",
               trControl = trainControl(method = "cv",
                                        number = 10,
                                        verboseIter = TRUE))
model
p <- predict(model, Boston)
error <- p - Boston$medv  
rmse_mval <- sqrt(mean(error^2))
rmse_mval

# 5.4 https://lgatto.github.io/IntroMachineLearningWithR/supervised-learning.html#classification-performance
# classification performance

# confusion matrix is good for this

library(mlbench)
data(Sonar)

tr <- sample(nrow(Sonar), round(nrow(Sonar) * .6)) # 60/40 sample split
train <- Sonar[tr, ]
test <- Sonar[-tr, ]

model <- glm(Class ~ ., data = train, family = "binomial") #glm model (logistic classification model)
p <- predict(model, test, type = "response")
summary(p)

Class <- ifelse(p > .5, "M", "R") # set threshold for
cl <- as.data.frame(Class) # formatted cl to df so confmatrix doesn't error on levels
table(Class, test$Class) # confusion matrix comparing pred and actuals (basic table version)

confusionMatrix(cl$Class, test$Class) # caret package confusion matrix (more informative)

# 5.4.2 receiver operating characteristic (ROC) curve
library(caTools)
colAUC(p, test[["Class"]], plotROC = TRUE) #TP and FP rates at each threshold

# 5.4.3 AUC

#create trainControl object
myControl <- trainControl(
  method = "cv", # cross-validation
  number = 10, # 10 fold
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # important
  verboseIter = FALSE
)

# train glm with trainControl object
model <- train(Class ~ ., Sonar,
               method = "glm",
               trControl = myControl)
model

# 5.5 https://lgatto.github.io/IntroMachineLearningWithR/supervised-learning.html#random-forest
# random forest

library(rpart) # recursive partitioning
m <- rpart(Class ~ ., data = Sonar,
           method = "class") # specified here
p <- predict(m, Sonar, type = "class") # and here
table(p, Sonar$Class) # in sample

# own
str <- sample(nrow(Sonar), round(nrow(Sonar) * .7)) # 70/30 sample
strain <- Sonar[str, ]
stest <- Sonar[-str, ]
ms <- rpart(Class ~ ., data = strain,
            method = "class") # rpart on training data
ps <- predict(ms, stest, type = "class") # predict on testing data
confusionMatrix(ps, stest$Class) # confusion matrix comparing pred on test and actuals on test

# 5.5.2 training a random forest
set.seed(12)
model <- train(Class ~ ., data = Sonar,
               method = "ranger")
model
plot(model)

model <- train(Class ~ ., data = Sonar,
               method = "ranger",
               tuneLength = 5) # sets number of hyperparameter values to test

set.seed(42)
myGrid <- expand.grid(mtry = c(5, 10, 20, 40, 60), # mtry is the number of randomly selected variables used at each split
                      splitrule = c("gini", "extratrees"),
                      min.node.size = 5)
model <- train(Class ~ ., data = Sonar,
               method = "ranger",
               tuneGrid = myGrid,
               trControl = trainControl(method = "cv",
                                        number = 5,
                                        verboseIter = TRUE))
model
plot(model)

# challenge
set.seed(42)
model <- train(Class ~., data = Sonar,
               method = "ranger",
               tuneLength = 5,
               trControl = trainControl(method = "cv",
                                        number = 5,
                                        verboseIter = TRUE))
model
plot(model)

# 5.6 https://lgatto.github.io/IntroMachineLearningWithR/supervised-learning.html#data-pre-processing-1
# data pre-processing

data(mtcars)
mtcars[sample(nrow(mtcars), 10),  "hp"] <- NA # replacing with NAs for demonstration
y = mtcars$mpg # target variable
x = mtcars[, 2:4] # predictors

try(train(x, y))

# 5.6.2 median imputation
train(x, y, preProcess = "medianImpute") # imputes using median
# works well if data is missing at random
# if cv, will impute unique for each fold

# 5.6.3 knn imputation
train(x, y, preProcess = "knnImpute") # works well if systemic bias in missing values

# 5.7 https://lgatto.github.io/IntroMachineLearningWithR/supervised-learning.html#scaling-and-scaling
# scaling and scaling
train(x, y, preProcess = "scale") # scale is devision by sd
train(x, y, preProcess = "center") # center is subtraction of the mean
train(x, y, preProcess = "pca") # pca generates set of high-variance and perpendicular predictors, preventing colinearity 

# 5.7.1 multiple pre-processing methods
train(x, y, preProcess = c("knnImpute", "center", "scale", "pca"))
# this represents classical order of operations: imputation, centering, scaling, then pca

# 5.8 https://lgatto.github.io/IntroMachineLearningWithR/supervised-learning.html#model-selection-1
# model selection

library(C50)
data(churn)
table(churnTrain$churn) / nrow(churnTrain) # maintain this ratio of yes/no

myFolds <- createFolds(churnTrain$churn, k = 5)
str(myFolds)

sapply(myFolds, function(i){ # check to make sure all 5 folds maintain ratio
  table(churnTrain$churn[i]) / length(i)
})

myControl <- trainControl( # reusable train control for consistency between models
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)

# 5.8.1 glmnet model
# glmnet is linear with built-in variable selection and coef regulatization
library(glmnet)
glm_model <- train(churn ~ ., churnTrain,
                   metric = "ROC", # to select optimal model
                   method = "glmnet",
                   tuneGrid = expand.grid(
                     alpha = 0:1,
                     lambda = 0:10/10),
                   trControl = myControl)
glm_model
plot(glm_model)

# random forest model challenge
rf_model <- train(churn ~ ., churnTrain,
                  metric = "ROC",
                  method = "ranger",
                  tuneGrid = expand.grid(
                    mtry = c(2, 5, 10, 20),
                    splitrule = c("gini", "extratrees"),
                    min.node.size = 5),
                  trControl = myControl)
rf_model
plot(rf_model)

# knn model challenge
knn_model <- train(churn ~ ., churnTrain,
                   metric = "ROC",
                   method = "knn",
                   tuneLength = 20,
                   trControl = myControl)
knn_model
plot(knn_model)

# 5.8.4 support vector machine model
library(kernlab)
svm_model <- train(churn ~ ., churnTrain,
                   metric = "ROC",
                   method = "svmRadial",
                   tuneLength = 10,
                   trControl = myControl)
svm_model
plot(svm_model)

# 5.8.5 naive bayes
library(naivebayes)
nb_model <- train(churn ~ ., churnTrain,
                  metric = "ROC",
                  method = "naive_bayes",
                  trControl = myControl)
nb_model
plot(nb_model)

# 5.8.6 comparing models
# use resamples() to select model with highest AUC and lowest AUC sd
library(lattice)
model_list <- list(glmnet = glm_model,
                   rf = rf_model,
                   knn = knn_model,
                   svm = svm_model,
                   nb = nb_model)
resamp <- resamples(model_list)
resamp

summary(resamp)
bwplot(resamp, metric = "ROC") # rf appears to be the best

# 5.8.7 pre-processing
# pre-process svm model and compare
svm_model1 <- train(churn ~ ., churnTrain,
                    metric = "ROC",
                    method = "svmRadial",
                    tuneLength = 20,
                    trControl = myControl)
svm_model1

svm_model2 <- train(churn ~ .,
                    churnTrain[, c(2, 6:20)],
                    metric = "ROC",
                    method = "svmRadial",
                    preProcess = c("scale", "center", "pca"),
                    tuneLength = 10,
                    trControl = myControl)
svm_model2

model_list <- list(svm1 = svm_model1,
                   svm2 = svm_model2)
resamp <- resamples(model_list)
summary(resamp)
bwplot(resamp, metric = "ROC") # looks like pre-processing is better (svm2)

# 5.8.8 predict using the best model
p <- predict(rf_model, churnTest)
confusionMatrix(p, churnTest$churn) # damn

```

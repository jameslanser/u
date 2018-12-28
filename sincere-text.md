```r
# Kaggle Competition
# https://www.kaggle.com/c/quora-insincere-questions-classification/data
# Tutorial
# https://www.youtube.com/watch?v=4vuw0AsHeGw

train <- read.csv("Kaggle/Quora/train.csv", stringsAsFactors = FALSE)
length(which(!complete.cases(train))) # check for missing data

table(train$target) # how many of each target 0/1
prop.table(table(train$target)) # as percent
train$length <- nchar(train$question_text) # create length var
summary(train$length)

library(caret)
# reduce size because this would be huge
index <- createDataPartition(train$target, times = 1,
                             p = .02, list = FALSE)
train <- train[index,]

par(mar = c(0, 0, 0, 0)) # reset plot

train$target <- as.factor(train$target) # set var 'target' as factor

# this is a distribution of length of text by target
library(ggplot2)
ggplot(train, aes(x = length, fill = target)) +
  theme_bw() +
  geom_histogram(binwidth = 10) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Length with Class Labels")

library(quanteda)
# pre process pipeline (tokenize, stem, dfm, as.matrix, df, model)
# tokenize question text
train.tokens <- tokens(train$question_text, what = "word",
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)
train.tokens <- tokens_tolower(train.tokens) # make lower case

# remove stop words
train.tokens <- tokens_select(train.tokens, stopwords(),
                             selection = "remove")
# word stem to condense repeated words
train.tokens <- tokens_wordstem(train.tokens, language = "english")

# create document frequency matrix
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE, remove = stopwords())

# transform to matrix
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:100]) # data gets huge, fast
dim(train.tokens.matrix)

# investigate the effects of stemming
colnames(train.tokens.matrix)[1:50]

# set up feature data frame with target (dependent variable), since model cannot be run on matrix
train.tokens.df <- cbind(target = train$target, as.data.frame(train.tokens.dfm))
train.tokens.df <- subset(train.tokens.df, select = -document) # rm line number character column

# clean up names
names(train.tokens.df) <- make.names(names(train.tokens.df))

# use caret to create stratified folds for 10 fold cv
# repeat this 3 times
set.seed(48743)
cv.folds <- createMultiFolds(train$target, k = 10, times = 3)
cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)

library(doSNOW)
start.time <- Sys.time() # time code execution for elapsed processing time

# create cluster to work on 10 logical cores
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# single decision tree algorithm
# may also use method = "rf" or method = "xgbTree"
rpart.cv.1 <- train(target ~ ., data = train.tokens.df, method = "rpart",
                    trControl = cv.cntrl, tuneLength = 7) # 7 dif configs for rpart, uses best (called hyperparameter tuning)
# processing done, stop cluster
stopCluster(cl)

# total time of execution time
total.time <- Sys.time() - start.time
total.time

# check results
rpart.cv.1


# -------------------------------------------------
# TF - IDF
# note this is done manually for detail, and it is better to use a packaged version for better runtime

# function for relative term frequency (TF) [row centric]
term.frequency <- function(row) {
  row / sum(row)
}

# function for inverse document frequency (IDF) [column centric]
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

# these two functions should yield same results as library(quanteda) package tfidf function, if normalization = true

# function for calculating TF - IDF
tf.idf <- function(tf, idf) {
  tf * idf
}

# first, normalize all documents via TF
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
View(train.tokens.df[1:20, 1:100])

# second, calculate IDF vector for training data and testing data
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)

# lastly, calculate TF-IDF for training corpus (combine TF and IDF values)
train.tokens.tfidf <- apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

# transpose matrix to original representation
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:50, 1:50])

# check for incomplete cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]

# to fix
train.tokens[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

# make clean data frame like in line 57-59
train.tokens.tfidf.df <- cbind(target = train$target, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

library(doSNOW) # repeated call
# run model on new tf-idf weighted data
start.time <- Sys.time()

# create cluster to work on 10 logical cores
cl <- makeCluster(3, type = "SOCK")
registerDoSnow(cl)

# single decision tree algorithm
rpart.cv.2 <- train(target ~ ., data = train.tokens.tfidf.df, method = "rpart",
                    trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() - start.time
total.time
rpart.cv.2


# -------------------------------------------------
# N-grams

# add bigrams to feature matrix
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)

# same pipeline
# transform to dfm, then matrix
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
train.tokens.dfm

# normalize all documents via TF
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)

# calculate IDF vector for training data and test data
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)

# calculate TF-IDF for training corpus
train.tokens.tfidf <- apply(train.tokens.df, 2, tf.idf,
                            idf = train.tokens.idf)

# transpose matrix
train.tokens.tfidf <- t(train.tokens.tfidf)

# fix incompletes (there will be none so skipping this step)

# make clean data frame
train.tokens.tfidf.df <- cbind(target = train$target, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

# clean up unused objects in memory
gc() # important in text analytics


# -------------------------------------------------
# VSM, latent semantic analysis (LSA), SVD

# time the code execution
start.time <- Sys.time()

# leverage single decision trees to evaluate if adding bigrams improves effectiveness of model
rpart.cv.3 <- train(target ~ ., data = train.tokens.tfidf.df, method = "rpart",
                    trControl = cv.cntrl, tuneLength = 7)
# ERROR: protect(): protection stack overflow
# execution time
total.time <- Sys.time() - start.time
total.time
rpart.cv.3
```


library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(e1071)

setwd("D:\\CoursERA\\Course8ML\\Assignment")
set.seed(543781)

# data Loading

urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download the datasets
training <- read.csv(url(urlTrain))
testing  <- read.csv(url(urlTest))

# create a partition with the training dataset
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
trainDat <- training[inTrain, ]
testDat  <- training[-inTrain, ]

dim(trainDat)
dim(testDat)

# remove variables with Nearly Zero Variance
nearZvar <- nearZeroVar(trainDat)
trainDat <- trainDat[, -nearZvar]
testDat  <- testDat[, -nearZvar]
dim(trainDat)
dim(testDat)

# Keeping variables with mean > .95
table(data.frame(sapply(trainDat, function(x) mean(is.na(x)))))
varNA    <- sapply(trainDat, function(x) mean(is.na(x))) > 0.95
trainDat <- trainDat[, varNA==FALSE]
testDat  <- testDat[, varNA==FALSE]



dim(trainDat)
dim(testDat)

# remove identification only variables (columns 1 to 5)
trainDat <- trainDat[, -(1:5)]
testDat  <- testDat[, -(1:5)]
dim(trainDat)
dim(testDat)

#i) Method: Random Forest

# model fit

set.seed(543781)
conRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRF <- train(classe ~ ., data=trainDat, method="rf",
                          trControl=conRF)
modFitRF$finalModel

# prediction on Test dataset

predictRF <- predict(modFitRF, newdata=testDat)
confMRF <- confusionMatrix(predictRF, testDat$classe)
confMRF

#ii) Method: Decision Trees

# model fit
set.seed(543781)
modFitDT <- rpart(classe ~ ., data=trainDat, method="class")
fancyRpartPlot(modFitDT)

# prediction on Test dataset
predictDT <- predict(modFitDT, newdata=testDat, type="class")
confMDT   <- confusionMatrix(predictDT, testDat$classe)
confMDT

#iii) Method: Generalized Boosted Model

# model fit
set.seed(543781)
conGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=trainDat, method = "gbm",
                    trControl = conGBM, verbose = FALSE)
modFitGBM$finalModel


# prediction on Test dataset
predictGBM <- predict(modFitGBM, newdata=testDat)
confMGBM <- confusionMatrix(predictGBM, testDat$classe)
confMGBM


# Applying RF Model to the Test Data


prdTEST <- predict(modFitRF, newdata=testing)
prdtTEST


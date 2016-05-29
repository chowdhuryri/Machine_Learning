###Practical Machine Learning Course Project: Prediction Assignment Writeup   
#### Rafiqul Chowdhury ####  
   
     
#### I. Background ####  

Availability of modern computer equipment made easier to capture live human 
activity data. Human activity recognition (HAR) is an emerging area of computer
vision research and applications. The goal of the activity recognition is an 
automated analysis/ interpretation of ongoing events related to the various
human activity. The objective of this analysis is to identify a good prediction
model based on the human activity dataset using machine learning techniques to 
predict the outcome of a test data set.   

A group of enthusiasts tech geeks who took measurements about the personal 
activity of themselves regularly to improve their health, to find patterns 
in their behavior. They collected data from accelerometers on the belt, 
forearm, arm, and dumbells of 6 participants. Participants were asked to one 
set of ten repetitions of the Unilateral  Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A); throwing the elbows
to the front (Class B); lifting the dumbbell only halfway (Class C); 
lowering the dumbbell only halfway (Class D) and throwing the hips to the front 
(Class E).

#### II. Data Preparation ####

Following R code chunk shows the required R packages used for this project.


```r
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(e1071)
```

*i) Reading and Splitting Data* 

Following R codes reads the data and partitioned into training (trainDat-70% )  validation (testDat-30 %) data set.


```r
setwd("D:\\CoursERA\\Course8ML\\Assignment")
set.seed(543781)

urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(urlTrain))
testing  <- read.csv(url(urlTest))

# create a partition with the training dataset
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
trainDat <- training[inTrain, ]
testDat  <- training[-inTrain, ]

dim(trainDat)
```

```
## [1] 13737   160
```

```r
dim(testDat)
```

```
## [1] 5885  160
```

*ii) Removing Nearly Zero Variance Predictors and with NA's*   

The data set consists of one class label and 159 features. Many of the features
have NA's. The caret function  *nearZeroVar* is used to diagnoses predictors that
have one unique value (i.e. are zero variance predictors). Now the total number of columns in the dat sets reduces to 104.


```r
nearZvar <- nearZeroVar(trainDat)
trainDat <- trainDat[, -nearZvar]
testDat  <- testDat[, -nearZvar]
dim(trainDat)
```

```
## [1] 13737   104
```

```r
dim(testDat)
```

```
## [1] 5885  104
```

Following R codes are used to identify which features has mean zero after removing NA's and selected features without zero means. Also, first, five features are removed as those are the part of ID. Now we have total 54 columns in both the datasets.


```r
table(data.frame(sapply(trainDat, function(x) mean(is.na(x)))))
```

```
## 
##                 0 0.979762684720099 
##                59                45
```

```r
varNA    <- sapply(trainDat, function(x) mean(is.na(x))) > 0.95
trainDat <- trainDat[, varNA==FALSE]
testDat  <- testDat[, varNA==FALSE] 
trainDat <- trainDat[, -(1:5)]
testDat  <- testDat[, -(1:5)]

dim(trainDat)
```

```
## [1] 13737    54
```

```r
dim(testDat)
```

```
## [1] 5885   54
```

#### III. Models for Predictions ####   

To identify best prediction model three models have been fitted. Tose are i) Random Forest, ii) Decision Tree and iii)  Generalized Boosted Model. Following are R codes fits these models.

*i) Random Forest*   


```r
set.seed(543781)
conRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRF <- train(classe ~ ., data=trainDat, method="rf",
                          trControl=conRF)
modFitRF$finalModel   
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.2%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    5 2651    2    0    0 0.0026335591
## C    0    6 2389    1    0 0.0029215359
## D    0    0    5 2246    1 0.0026642984
## E    0    1    0    5 2519 0.0023762376
```

Following R codes are used for prediction on test dataset using fitted Random 
Forest model and compute confusion matrix.


```r
predictRF <- predict(modFitRF, newdata=testDat)
confMRF <- confusionMatrix(predictRF, testDat$classe)
confMRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    1    0    0    0
##          B    0 1137    2    0    0
##          C    0    1 1024    2    0
##          D    0    0    0  962    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9988          
##                  95% CI : (0.9976, 0.9995)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9982   0.9981   0.9979   0.9991
## Specificity            0.9998   0.9996   0.9994   0.9998   1.0000
## Pos Pred Value         0.9994   0.9982   0.9971   0.9990   1.0000
## Neg Pred Value         1.0000   0.9996   0.9996   0.9996   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1932   0.1740   0.1635   0.1837
## Detection Prevalence   0.2846   0.1935   0.1745   0.1636   0.1837
## Balanced Accuracy      0.9999   0.9989   0.9987   0.9989   0.9995
```

*ii) Decision Trees*


```r
set.seed(543781)
modFitDT <- rpart(classe ~ ., data=trainDat, method="class")
#fancyRpartPlot(modFitDT)
```

Following R codes are used for prediction on test dataset using fitted Decision
Tree model and compute confusion matrix.


```r
predictDT <- predict(modFitDT, newdata=testDat, type="class")
confMDT   <- confusionMatrix(predictDT, testDat$classe)
confMDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1492  207   43   59   51
##          B   57  664   40   70  110
##          C   21   72  831  139  101
##          D   88  134   53  640  138
##          E   16   62   59   56  682
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7322          
##                  95% CI : (0.7207, 0.7435)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6603          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8913   0.5830   0.8099   0.6639   0.6303
## Specificity            0.9145   0.9416   0.9315   0.9161   0.9598
## Pos Pred Value         0.8056   0.7056   0.7139   0.6078   0.7794
## Neg Pred Value         0.9549   0.9039   0.9587   0.9329   0.9202
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2535   0.1128   0.1412   0.1088   0.1159
## Detection Prevalence   0.3147   0.1599   0.1978   0.1789   0.1487
## Balanced Accuracy      0.9029   0.7623   0.8707   0.7900   0.7951
```

*iii) Generalized Boosted Model*


```r
set.seed(543781)
conGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=trainDat, method = "gbm",
                    trControl = conGBM, verbose = FALSE)
#modFitGBM$finalModel
```

Following R codes are used for prediction on test dataset using fitted Generalized Boosted Model model and compute confusion matrix.


```r
predictGBM <- predict(modFitGBM, newdata=testDat)
confMGBM <- confusionMatrix(predictGBM, testDat$classe)
confMGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673   13    0    1    0
##          B    1 1112    7    0    3
##          C    0   11 1016   17    1
##          D    0    3    1  946    8
##          E    0    0    2    0 1070
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9884         
##                  95% CI : (0.9854, 0.991)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9854         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9763   0.9903   0.9813   0.9889
## Specificity            0.9967   0.9977   0.9940   0.9976   0.9996
## Pos Pred Value         0.9917   0.9902   0.9722   0.9875   0.9981
## Neg Pred Value         0.9998   0.9943   0.9979   0.9963   0.9975
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1890   0.1726   0.1607   0.1818
## Detection Prevalence   0.2867   0.1908   0.1776   0.1628   0.1822
## Balanced Accuracy      0.9980   0.9870   0.9921   0.9894   0.9942
```

####IV. Prediction of class for 20 test samples####

The accuracy of the Random Forest Model is the highest (0.9888), followed by GBM (.9884) and Decision Tree (.7322), respectively. Random Forest model is used to predict 20 test cases. Following codes produce the necessary prediction.


```r
prdTEST <- predict(modFitRF, newdata=testing)   
prdTEST
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


#No-Show Predictive Model
#Author: Jeevan Bandreddi
#Data Source: https://www.kaggle.com/joniarroba/noshowappointments/version/1

#Installing required libraries
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(Hmisc)) install.packages("Hmisc")
if (!require(caret)) install.packages("caret")
if (!require(kknn)) install.packages("kknn")
if (!require(pROC)) install.packages("pROC")
if (!require(MASS)) install.packages("MASS")
if (!require(e1071)) install.packages("e1071")
if (!require(randomForest)) install.packages("randomForest")


#Pulling required libraries
library(tidyverse)
library(Hmisc)
library(caret)
library(pROC)
library(kknn)
library(MASS)
library(e1071)
library(randomForest)


################Pulling data into R and data setup#####################
noshow <- read_delim('No-show-Issue-300k.csv', delim = ';')
describe(noshow)

noshow$Gender <- factor(noshow$Gender)
noshow$DayOfTheWeek <- factor(noshow$DayOfTheWeek, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
noshow$AwaitingTime <- abs(noshow$AwaitingTime)
noshow$Diabetes <- factor(noshow$Diabetes)
noshow$Alcoolism <- factor(noshow$Alcoolism)
noshow$HiperTension <- factor(noshow$HiperTension)
noshow$Handcap <- factor(noshow$Handcap)
noshow$Smokes <- factor(noshow$Smokes)
noshow$Scholarship <- factor(noshow$Scholarship)
noshow$Tuberculosis <- factor(noshow$Tuberculosis)
noshow$Sms_Reminder <- factor(noshow$Sms_Reminder)
noshow$Status <- factor(ifelse(noshow$Status == "No-Show", 1, 0), levels = c(1, 0))

noshow <- subset(noshow, select = -c(AppointmentRegistration, ApointmentData))


###############Exploratory Analysis####################################
describe(noshow$Gender)

#find no-show rate by a variable 
noshow %>%
  group_by(Gender) %>%
  summarise(count = n(),
            noshow = sum(ifelse(Status == 1, 1, 0)),
            rate = noshow/count)

#graphing continous various against Status field 
ggplot(data = noshow, aes(x = Status, y = Age, fill = Status)) + geom_boxplot() + 
  ggtitle("Age Distribution by No-Show Status") + theme(plot.title = element_text(hjust = 0.5))

ggplot(data = noshow, aes(x = Status, y = AwaitingTime, fill = Status)) + geom_boxplot() + 
  ggtitle("AwaitingTime Distribution by No-Show Status") + theme(plot.title = element_text(hjust = 0.5))

#variable variance check 
variancecheck <- nearZeroVar(noshow, saveMetrics= TRUE)
variancecheck

##############Model Setup#############################################
#Post exploratory analysis processing
noshow <- noshow[noshow$Age >= 0,]
noshow$Handcap <- factor(ifelse(noshow$Handcap == 0, 0, 1))
noshow$Sms_Reminder <- factor(ifelse(noshow$Sms_Reminder == 0, 0, 1))
noshow$Tuberculosis <- NULL

#Creating sets
set.seed(333)

##scale numeric variables
noshow$Age <- scale(noshow$Age)
noshow$AwaitingTime <- scale(noshow$AwaitingTime)

##create training and hold sets
trainIndex <- createDataPartition(noshow$Status, p = 0.8, list = FALSE, times = 1)
noshow.train <- noshow[trainIndex,]
noshow.holdout <- noshow[-trainIndex,]

##down-sample training set 
noshow.train <- downSample(noshow.train[, -4], y = noshow.train$Status)
noshow.train$Status <- noshow.train$Class
noshow.train$Class <- NULL
noshow.holdout$Age <- as.numeric(noshow.holdout$Age)
noshow.holdout$AwaitingTime <- as.numeric(noshow.holdout$AwaitingTime)

describe(noshow.train$Status)
describe(noshow.holdout$Status)

####################Models###############################################
Ctrl <- trainControl(method = "cv", number = 5)

#logistic regression
logm.1 <- train(
  form = Status ~ .,
  data = noshow.train,
  trControl = Ctrl,
  method = "glm",
  family = "binomial"
)
logm.1
log.predict <- predict(logm.1, newdata = noshow.holdout)
confusionMatrix(log.predict, noshow.holdout$Status)


#knn
knnm.1 <- train(
  form = Status ~ .,
  data = noshow.train,
  trControl = Ctrl,
  method = "knn",
  tuneGrid = expand.grid(k = c(1, 3, 5))
)
knnm.1
knn.predict <- predict(knnm.1, newdata = noshow.holdout)
confusionMatrix(knn.predict, noshow.holdout$Status)


#LDA
ldam.1 <- train(
  form = Status ~ .,
  data = noshow.train,
  trControl = Ctrl,
  method = "lda"
)
ldam.1
lda.predict <- predict(ldam.1, newdata = noshow.holdout)
confusionMatrix(lda.predict, noshow.holdout$Status)


#QDA
qdam.1 <- train(
  form = Status ~ .,
  data = noshow.train,
  trControl = Ctrl,
  method = "qda"
)
qdam.1
qda.predict <- predict(qdam.1, newdata = noshow.holdout)
confusionMatrix(qda.predict, noshow.holdout$Status)

#random forest
randomm.1 <- train(
  form = Status ~ .,
  data = noshow.train,
  trControl = Ctrl,
  method = "rf"
)
randomm.1
random.predict <- predict(randomm.1, newdata = noshow.holdout)
confusionMatrix(random.predict, noshow.holdout$Status)

#boosted tree
boostedm.1 <- train(
  form = Status ~ .,
  data = noshow.train,
  trControl = Ctrl,
  method = "gbm"
)
boostedm.1
boosted.predict <- predict(boostedm.1, newdata = noshow.holdout)
confusionMatrix(boosted.predict, noshow.holdout$Status)

#svm-linear
svmlinearm.1 <- tune(svm, Status ~ ., data = noshow.train, kernel = "linear",
                     ranges = list(cost=c(0.1, 1, 10, 100)))
svmlinear.predict <- predict(svmlinearm.1$best.model, noshow.holdout[, -4])
confusionMatrix(svmlinear.predict, noshow.holdout$Status)

#svm-radial
svmradialm.1 <- tune(svm, Status ~ ., data = noshow.train, kernel = "radial",
                     ranges = list(cost=c(0.1, 1, 10, 100),
                                   gamma=c(0.5, 1, 2, 3)))
svmradial.predict <- predict(svmradialm.1$best.model, noshow.holdout[, -4])
confusionMatrix(svmradial.predict, noshow.holdout$Status)


#################Predictor Analysis#########################################
#find coefficent magnitudes and importance 
sort(abs(logm.1$finalModel$coefficients), decreasing = TRUE)
varImp(boostedm.1)
varImp(randomm.1)



library(caret)
install.packages("party")
library(rpart)
library(randomForest)
library(dplyr)
library(caTools)
install.packages("gbm")
library(gbm)
library(plyr)
# load data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# combine train and test data
casual <- train$casual
registered <- train$registered
count <- train$count
train <- train[,-c(10:12)]
bike <- rbind(train,test) 

# seperate date and time variable
Hours <- format(as.POSIXct(strptime(bike$datetime,"%d/%m/%Y %H:%M:%S",tz="")) ,format = "%H:%M:%S")
date <- as.POSIXct(bike$datetime)
df_date <- data.frame(date = date, Year = as.numeric(format(date, format = "%Y")), Month = as.numeric(format(date, format = "%m")), Day = as.numeric(format(date, format = "%d")), Hour = as.numeric(format(date, format = "%H")))

# create weekday variable
df_date$weekday <- as.factor(weekdays(df_date$date))
df_date <- df_date[,2:6]
bike <- cbind(bike, df_date)
str(bike)

# create weekend variable
bike$weekend <- ifelse(bike$weekday == "Saturday" | bike$weekday == "Sunday", "1", "0")

# distribution of numerical variable
bike$season <- as.numeric(bike$season)
hist(bike$season)
bike$holiday <- as.numeric(bike$holiday)
hist(bike$holiday)
bike$workingday <- as.numeric(bike$workingday)
hist(bike$workingday)
hist(bike$temp)
bike$weather <- as.numeric(bike$weather)
hist(bike$weather)
bike$weekend <- as.numeric(bike$weekend)
hist(bike$weekend)
hist(bike$humidity)

# transform variables
table(bike$season)
bike$season <- as.factor(bike$season)
bike$Hour <- as.factor(bike$Hour)
bike$holiday <- as.factor(bike$holiday)
bike$workingday <- as.factor(bike$workingday)
bike$weather <- as.factor(bike$weather)
bike$Year <- as.factor(bike$Year)
bike$Month <- as.factor(bike$Month)
bike$weekend <- as.factor(bike$weekend)
str(bike)

# group Hour variable into 4 groups: 0-6, 7-15, 16-19, 20-23
bike$Hour <- as.numeric(bike$Hour)
hour <- ifelse(bike$Hour < 7, "0-6", ifelse( bike$Hour >= 7 & bike$Hour < 16, "7-15", ifelse(bike$Hour >= 16 & bike$Hour < 20, "16-19", "20-23")))
bike <- cbind(bike, hour)
table(bike$hour)
bike <- bike[,-13]
bike$hour <- as.factor(bike$hour)
str(bike)

# group Month variable into 3 groups: 1-5, 6-10, 11-12
bike$Month <- as.numeric(bike$Month)
month <- ifelse(bike$Month < 6, "1-5", ifelse(bike$Month >= 6 & bike$Month < 11, "6-10", "11-12"))
bike <- cbind(bike, month)
table(bike$month)
bike <- bike[,-11]
bike$month <- as.factor(bike$month)

# normalize variables: temp, atemp, humidity, windspeed

dat <- bike %>% mutate_each_(funs(scale), vars = c("temp", "atemp", "humidity", "windspeed"))

# data preparation, we split train data into 2 sets to evaluate the model before submit
train <- bike[1:10886,]
test <- bike[10887:17379,]
train_dat <- cbind(train, casual)
train_dat <- cbind(train_dat, registered)
train_dat <- cbind(train_dat, count)
# split data
set.seed(1)
split = sample.split(train_dat$count, SplitRatio = 0.7)
subTrain = subset(train_dat, split == TRUE)
subTest = subset(train_dat, split == FALSE)

### building model
# create formular
casual <- casual ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed + Year + Day + weekday + weekend + hour + month
registered <- registered ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed + Year + Day + weekday + weekend + hour + month

##### simple tree model
casual_rpart <- rpart(casual, data = subTrain)
casual_pred <- round(predict(casual_rpart, newdata = subTest))
registered_rpart <- rpart(registered, data = subTrain)
registered_pred <- round(predict(registered_rpart, newdata = subTest))
count_rpart <- casual_pred + registered_pred

# calculate RMSE
rmse_rpart <- sqrt(mean((count_rpart - subTest$count)^2))
rmse_rpart
# rmse = 115

# simple tree model with "caret" package
set.seed(1)
fitControl1 <- trainControl(method = "cv", number = 10)
Grid <- expand.grid(cp = seq(0,0.05,0.005))
# predict casual and registered
casual_rpartcv <- train(casual, data = subTrain, method = "rpart", trControl = fitControl1, tuneGrid = Grid, metric = "RMSE", maximize = FALSE)
casual_predcv <- predict(casual_rpartcv, newdata = subTest)
registered_rpartcv <- train(registered, data = subTrain, method = "rpart", trControl = fitControl1, tuneGrid = Grid, metric = "RMSE", maximize = FALSE)
registered_predcv <- predict(registered_rpartcv, newdata = subTest)
count_rpartcv <- round(casual_predcv + registered_predcv)

# calculate RMSE 
rmse_rpartcv <- sqrt(mean((count_rpartcv - subTest$count)^2))
rmse_rpartcv
# rmse = 113

# using caret package for CART tree does not improve the result much

##### try random forest
set.seed(1)
fitControl2 <- trainControl(method = "cv", number = 10)
Grid1 <- expand.grid(mtry = seq(4,16,4))

# predict casual and registered
casual_rf <- train(casual, data = subTrain, method = "rf", trControl = fitControl2, metric = "RMSE", maximize = FALSE, tuneGrid = Grid1, ntree = 250, verbose = FALSE) 
casual_rf
casual_predrf <- predict(casual_rf, newdata = subTest)

registered_rf <- train(registered, data = subTrain, method = "rf", trControl = fitControl2, metric = "RMSE", maximize = FALSE, tuneGrid = Grid1, ntree = 250)
registered_rf
registered_predrf <- predict(registered_rf, newdata = subTest)
count_rf <- round(casual_predrf + registered_predrf)
  
# calculate RMSE 
rmse_rf <- sqrt(mean((count_rf - subTest$count)^2))
rmse_rfrf
# rmse = 100
# compared with cart model, random forest do better with rmse = 100 comapred with 113, the two most important parameter of random forest is mtry and ntree, in the case above, i just tune the mtry parameter cause of the running time of computer

##### try gradient boosting algorithm
set.seed(1)
fitControl3 <- trainControl(method = "cv", number = 3)
Grid3 <- expand.grid(shrinkage = 0.01, interaction.depth = 8, n.minobsinnode = 10, n.trees = 2500)

# predict casual and registered
casual_gbm <- train(casual ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed + Year + Day + weekday + weekend + hour + month, data = subTrain, method = "gbm", trControl = fitControl3, metric = "RMSE", maximize = FALSE, tuneGrid = Grid3 )
casual_gbm
casual_predgbm <- predict(casual_gbm, newdata = subTest)

registered_gbm <- train(registered ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed + Year + Day + weekday + weekend + hour + month, data = subTrain, method = "gbm", trControl = fitControl3, metric = "RMSE", maximize = FALSE, tuneGrid = Grid3)
registered_gbm
registered_predgbm <- predict(registered_gbm, newdata = subTest)

count_gbm <- casual_predgbm + registered_predgbm

# calculate RMSE 
rmse_gbm <- sqrt(mean((count_gbm - subTest$count)^2))
rmse_gbm

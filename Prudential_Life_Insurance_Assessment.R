#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Prudential Life Insurance Assessment
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------
# Business Problem

# Picture this. You are a data scientist in a start-up culture with the potential to have a very large impact on the business. 
# Oh, and you are backed up by a company with 140 years' business experience.

# Curious? Great! You are the kind of person we are looking for.
# 
# Prudential, one of the largest issuers of life insurance in the USA, is hiring passionate data scientists to join a newly-formed Data Science group solving complex challenges and identifying opportunities. The results have been impressive so far but we want more. 
# 
# The Challenge
# In a one-click shopping world with on-demand everything, the life insurance application process is antiquated. 
# Customers provide extensive information to identify risk classification and eligibility, including scheduling medical exams, a process that takes an average of 30 days.
# 
# The result? People are turned off. That's why only 40% of U.S. households own individual life insurance. 
# Prudential wants to make it quicker and less labor intensive for new and existing customers to get a quote while maintaining privacy boundaries.
# 
# By developing a predictive model that accurately classifies risk using a more automated approach, you can greatly impact public perception of the industry.
# 
# The results will help Prudential better understand the predictive power of the data points in the existing assessment, enabling us to significantly streamline the process.
#------------------------------------------------------------------------------------------------------------------------------------------

# install.packages("pillar")
# install.packages("dplyr")
# install.packages("tibble")
# install.packages("pdflatex")
# install.packages("ggpubr")
# install.packages("neuralnet")
# install.packages("ada")
# install.packages("zoo")
# install.packages("ade4")
# install.packages("gtools")
# install.packages("xgboost")
# install.packages("forecast")
# install.packages("mlbench")
# install.packages("caret")
# install.packages("mlr")
# install.packages("data.table")
# install.packages("Metrics")

library(caret)
library(corrplot)
library(xgboost)
library(stats)
library(knitr)
library(ggplot2)
library(Matrix)
library(plotly)
library(htmlwidgets)
library(readr)
library(randomForest)
library(data.table)
library(h2o)
library(dplyr)
library(tidyr)
library(Metrics)

########################################################################################################################
# Importing the data
########################################################################################################################

train1<-read.csv(file = "C:/Users/puj83/OneDrive/Portfolio/Prudential_Life_Insurance_Assessment/train.txt", header = T, sep = ",")
test1<-read.csv(file = "C:/Users/puj83/OneDrive/Portfolio/Prudential_Life_Insurance_Assessment/test.txt", header = T, sep = ",")

train<-train1
test<-test1

##### Remove id
train$Id<-NULL
test$Id<-NULL
# identify number of classes
num.class = length(levels(factor(unlist(train[,"Response"]))))
y = as.matrix(as.integer(unlist(train[,"Response"]))-1)

#####  Remove columns with NA, use test data as referal for NA
cols.without.na = colSums(is.na(train)) == 0
train = train[, cols.without.na]
cols.without.na = colSums(is.na(test)) == 0
test = test[, cols.without.na]
##### Check for zero variance
zero.var = nearZeroVar(train, saveMetrics=F)

train<-train[,-zero.var]
test<-test[, -zero.var]

##### Simple visualization
#x<-as.data.frame(head(train[,c("BMI","Ht","Wt","Ins_Age","Product_Info_3")],100))
x<-as.data.frame(head(train[,c("BMI","Ht","Wt")],100))
y1<-factor(unlist(head(train[,"Response"],100)))
trellis.par.set(theme = col.whitebg(), warn = FALSE)
featurePlot(x, y1, "box",auto.key = list(columns = 3))

featurePlot(x, y1, "density",
            #      scales = list(x = list(relation="free"), 
            #                    y = list(relation="free")), 
            #      adjust = 1.5, 
            #     pch = "|", 
            #      layout = c(4, 2), 
            auto.key = list(columns = 3))

corrplot.mixed(cor(train[,c(2:20)]), lower="circle", upper="color", 
               tl.pos="lt", tl.cex=0.6, diag="n", order="hclust", hclust.method="complete")

##### convert data to matrix
train$Response = NULL
train.matrix = as.matrix(train)
mode(train.matrix) = "numeric"
test.matrix = as.matrix(test)
mode(test.matrix) = "numeric"

param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "merror",    
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 8,    # maximum depth of tree 
              "eta" = 0.1,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample"           = 0.7,
              "colsample_bytree"    = 0.7,
              "min_child_weight"    = 3
)

set.seed(789)

nround.cv = 10
system.time( bst.cv <- xgb.cv(param=param, data=train.matrix, label=y, 
                              nfold=10, nrounds=nround.cv, prediction=TRUE, verbose=T
                              #    callbacks = list(cb.cv.predict(save_models = FALSE))
))

bst.cv$evaluation_log %>%
  select(-contains("std")) %>%
  gather(TestOrTrain, merror,-iter) %>%
  ggplot(aes(x = iter, y = merror, group = TestOrTrain, color = TestOrTrain)) + 
  geom_line() + 
  theme_bw()

col.names<-colnames(bst.cv$evaluation_log)
setnames(bst.cv$evaluation_log, old = col.names, new = c("iter","train.merror.mean","train.merror.std","test.merror.mean","test.merror.std" ))

min.merror.idx = which.min(bst.cv$evaluation_log[, test.merror.mean]) 

bst.cv$dt=bst.cv$evaluation_log
bst.cv$dt[min.merror.idx,]

pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")

y<-factor(y+1)
pred.cv<-factor(pred.cv)

confusionMatrix(y, pred.cv)

train<-train1
test<-test1

# All features shared, making feature transformations simultaneously. 
response <- train$Response
train$training <- 1
test$training  <- 0

data <- rbind(train[-c(1,128)], test[-1])
colnames(data)

prop.table(table(response))


feature.names <- names(data[-127])
for( f in feature.names ){
  if(class(data[[f]]) == "character"){
    print(class(data[[f]]))
    levels <- unique(c(train[[f]],test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]]), levels = levels)
    test[[f]] <- as.integer(factor(test[[f]]), levels = levels)
    data[[f]] <- as.integer(factor(data[[f]]), levels = levels)
    
  }
}

data.roughfix <- na.roughfix(data)
y = as.matrix(as.integer(unlist(response))-1)
# Using training data to identify most important features with xgboost.
system.time(model_xgboost <- xgboost(data = data.matrix(data.roughfix[data.roughfix$training==1,]), 
                                     label  = y, 
                                     nround  = 10, 
                                     objective = "multi:softprob",    
                                     eval_metric = "merror",
                                     num_class=8,
                                     eta = 0.01,  # learning rate                                                 
                                     max.depth = 3,  
                                     missing = NaN,
                                     verbose = TRUE,                                         
                                     print_every_n = 1,
                                     early_stopping_rounds = 10 ))

model_dump <- xgb.dump(model_xgboost, with_stats = T)
importance.matrix <- xgb.importance(names(data.roughfix), model_xgboost)
xgb.plot.importance(importance.matrix[1:30])

medkeywords <- apply(data.roughfix[,79:126], 1, sum)
data.roughfix$medkeywords <- as.integer(medkeywords)
partition <- createDataPartition(response, times = 1, p = 0.75)
training <- data.roughfix[data.roughfix$training==1,]

y_train <- y[partition$Resample1,] 
y_test <- y[-partition$Resample1,] 

training_train <- training[partition$Resample1,-127]
training_test <- training[-partition$Resample1,-127]
system.time(model_xgboost <- xgboost(data = data.matrix(training_train), 
                                     label  = y_train, 
                                     nround  = 100, 
                                     objective = "multi:softprob",    
                                     eval_metric = "merror",
                                     num_class=8,
                                     eta = 0.01,                                                
                                     max.depth = 3,  
                                     missing = NaN,
                                     verbose = TRUE,                                         
                                     print_every_n = 1,
                                     early_stopping_rounds = 10))

pred <- predict(model_xgboost, data.matrix(training_test), missing=NaN)
pred_m<- matrix(pred, nrow=length(pred)/num.class, ncol=num.class)
pred_m = max.col(pred_m, "last")
confusionMatrix(factor(y_test+1), factor(pred_m))

model_dump <- xgb.dump(model_xgboost, with_stats = T)
importance.matrix <- xgb.importance(names(data.roughfix), model_xgboost)
xgb.plot.importance(importance.matrix[1:30])

categorical_string <- as.character("Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41")
categorical_names <- unlist(strsplit(categorical_string, split = ", "))
top30features <- importance.matrix$Feature[1:30]
which(top30features %in% categorical_names)

top30categorical_names <- top30features[which(top30features %in% categorical_names)]
# One-hot encoding top 15 categorical variables
top30categorical_factor <- as.data.frame(apply(data.roughfix[,top30categorical_names],2,as.factor))
categorical_one_hot <- as.data.frame(model.matrix(~.-1, top30categorical_factor[-8])) # Except Medical_History_2 which has too many levels.
categorical_one_hot2 <- as.data.frame(sapply(categorical_one_hot,as.factor))
str(categorical_one_hot2)

data.roughfix2 <- cbind(data.roughfix, categorical_one_hot2)

system.time(model2 <- xgboost(data = data.matrix(data.roughfix2[data.roughfix2$training==1,]), 
                              label  = y, 
                              nround  = 100, 
                              objective = "multi:softprob",    
                              eval_metric = "merror",
                              num_class=8,
                              eta = 0.01,                                             
                              max.depth = 3,  
                              missing = NaN,
                              verbose = TRUE,                                         
                              print_every_n = 1,
                              early_stopping_rounds = 10 ))

model_dump <- xgb.dump(model2, with_stats = T)
importance.matrix <- xgb.importance(names(data.roughfix2), model2)
xgb.plot.importance(importance.matrix[1:30])

folds <- createFolds(response, 2)
training <- data.roughfix[data.roughfix$training == 1,]
cv_results <- lapply(folds, function(x){
  train <- data.matrix(training[-x,])
  test <- data.matrix(training[x,])
  model <- xgboost(data = train,
                   label = y[-x],
                   nround  = 100, 
                   objective = "multi:softprob",    
                   eval_metric = "merror",
                   num.class=8,
                   eta = 0.01,                                             
                   max.depth = 3,  
                   missing = NaN,
                   verbose = TRUE,                                         
                   print_every_n = 1,
                   early_stopping_rounds = 10
  )
  
  model_pred <- predict(model, test, missing=NaN)
  pred_m<- matrix(model_pred, nrow=length(model_pred)/num.class, ncol=num.class)
  pred_m = max.col(pred_m, "last")
  actual <- response[x]
  qwkappa <- Metrics::ScoreQuadraticWeightedKappa(actual, pred_m)
  print(qwkappa)
  return(qwkappa)
})

cv_results


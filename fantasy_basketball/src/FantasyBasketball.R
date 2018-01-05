# Libraries ---------------------------------------------------------------
library(dplyr)
library(mice)
library(leaps)
library(DAAG)
library(pls)
library(dplyr)
library(glmnet)
library(FNN)
library(randomForest)
library(caret)
library(gbm)
library(tidyverse)
library(MASS)
library(matrixStats)

# Clean Data --------------------------------------------------------------

## Load Data
load("Users/JuliusHigiro/FantasyBasketball/src/stat_learning_proj.Rdata")

## Check for missing values
Missing_Data_Rows <- yearly_training[!complete.cases(yearly_training), ]

# It looks like the missing values are for percentage points, where people didn't get any shots and thus no percentage.


## Remove Copies of Players. # I also removed the teams variable and the effective field goals variable.

yearly_training_1 <- yearly_training %>% group_by(player,age,start_year,end_year) %>%
  summarise(games = sum(g),
            games_started = sum(gs),
            minutes_played = sum(mp),
            field_goal = sum(fg),
            field_goal_attempts = sum(fga), 
            field_goal_pct = sum(fg_pct),
            three_point = sum(three_p),
            three_point_attempts = sum(three_pa),
            three_point_pct = sum(three_p_pct),
            two_point = sum(two_p),
            two_point_attempts = sum(two_pa),
            two_point_pct = sum(two_p_pct),
            free_throw = sum(ft),
            free_throw_attempts = sum(fta),
            free_throw_pct = sum(ft)/sum(fta),
            offense_rebound = sum(orb),
            defense_rebound = sum(drb),
            total_reb = sum(orb) + sum(drb),
            assists = sum(ast),
            steals = sum(stl),
            blocks = sum(blk),
            turnovers = sum(tov),
            personal_foul = sum(pf),
            total_points = sum(pts))

## See how many copies of each player remain. # Now only the yearly performances of players remain.
## We note that the number of Players with 2 copies is 396

table(table(yearly_training_1$player)) 

## Get separate data frames for 2014 and 2015
train_fourteen <- yearly_training_1[yearly_training_1$start_year == 2014,]
train_fifteen <- yearly_training_1[yearly_training_1$start_year == 2015,]

## Set row names equal to the players. I couldn't do this before because row names can't be duplicated
## Set the rownames equal to the players
row.names(train_fourteen) <- train_fourteen$player
row.names(train_fifteen) <- train_fifteen$player

## Remove columns that will be duplicated when I recombine the data frames
train_fifteen$player <- NULL
train_fifteen$start_year <- NULL
train_fifteen$end_year <- NULL

## Add a _15 to all of the column names for the train_fifteen dataframe
colnames(train_fourteen) <- paste(colnames(train_fourteen),"14",sep = "_")
colnames(train_fifteen) <- paste(colnames(train_fifteen),"15",sep = "_")

## Now to recombine the data frames, making sure the rows are the same

clean_yearly_training <- merge(train_fourteen,train_fifteen, by=0, all=TRUE)

## Those who didn't play both years have NAs in their data now. However, those who didn't get points in a category
# have NAs because lack of points means NAs in the percentage categories. In order to eliminate those who are NA because of 
# not playing that season, we remove those with NAs in another category --> games played

clean_yearly_training <- clean_yearly_training[complete.cases(clean_yearly_training$games_14)&
                                                 complete.cases(clean_yearly_training$games_15), ]



## Check manipulation. The number of players left is 396, which concurs with what we found on line 44. 
table(table(clean_yearly_training$player)) 

## Now to clean up the clean_yearly_training data frame

clean_yearly_training$Row.names <- NULL # Don't care about this anymore
clean_yearly_training$start_year_14 <- NULL 
clean_yearly_training$end_year_14 <- NULL 
clean_yearly_training$start_year_15 <- NULL 
clean_yearly_training$end_year_15 <- NULL 


rownames(clean_yearly_training) <- clean_yearly_training$player_14

# Add Height as a Predictor -----------------------------------------------

heights <- data.frame(players_training[,c(4)])
rownames(heights) <- players_training$names

cl_yearly_training <- merge(clean_yearly_training,heights, by=0, all=TRUE)

cl_yearly_training$Row.names <- NULL

colnames(cl_yearly_training)[52] <- "Heights"

clean_yearly_training <- cl_yearly_training[! is.na(cl_yearly_training$player_14),]

ht_ft <- as.numeric(substr(clean_yearly_training$Heights,1,1))*12
ht_in <- as.numeric(substr(clean_yearly_training$Heights,3,4))

ht <- ht_ft + ht_in

clean_yearly_training$Heights <- ht

# Multiple Imputation to get a data set without NAs -----------------------------------------------------

set.seed(1760)

tempData <- clean_yearly_training[2:ncol(clean_yearly_training)]

tempData <- mice(tempData ,m=5,maxit=5)

imputed <- tempData$imp

completedData <- mice::complete(tempData,1)

clean_yearly_training[,2:ncol(clean_yearly_training)] <- completedData

rm(list=setdiff(ls(), c("clean_yearly_training","yearly_training","games_training",
                        "players_training")))

# Create Data Frame to Store Results, Define useful vectors --------------------------------------

Appendix <- data.frame(matrix(data = NA, nrow = 7, ncol = 23, byrow = FALSE,
                   dimnames = NULL), 
                   row.names = c("total_points_15","total_reb_15",
                                 "steals_15",
                                 "assists_15",
                                 "three_point_15",
                                 "free_throw_pct_15",
                                 "blocks_15"))
colnames(Appendix) <- c("Forward_CV","Number Forward Predictors",
                        "Backward_CV","Number Backward Predictors",
                        "Exhaustive_CV","Number Exhaustive Predictors",
                               "PCR_CV","Number PCR Components", 
                               "PLSR_CV","Number PLSR Components",
                               "Ridge_CV","Ridge Lambda",
                               "Lasso_CV","Lasso Lambda",
                               "KNN_CV","KNN K",
                               "Rand_For_CV","Rand_For MTry",
                               "Boosting_CV","Boosting_NT","Boosting_id",
                        "Mean CV","Median CV")

categories <- c("total_points_15","total_reb_15","steals_15","assists_15","three_point_15",
                "free_throw_pct_15","blocks_15")

predictors <- c("age_14","games_14","games_started_14", "minutes_played_14", 
                "field_goal_14", "field_goal_attempts_14", "field_goal_pct_14", 
                "three_point_14", "three_point_attempts_14", "three_point_pct_14", 
                "two_point_14", "two_point_attempts_14", "two_point_pct_14", 
                "free_throw_14", "free_throw_attempts_14", "free_throw_pct_14", 
                "offense_rebound_14", "defense_rebound_14", "assists_14", 
                "steals_14", "blocks_14", "turnovers_14", "personal_foul_14", 
                "total_points_14", "Heights")

# Lists to Save Best Models for MLR Models --------------------------------

Forward_List <- list() # List to store formula of best models for Forward Stepwise Selection
Backward_List <- list() # List to store best models for Backward Stepwise Selection
Exhaustive_List <- list() # ""

# Cross Validation for Forward Stepwise Linear Regression -------------------------

for(j in 1:length(categories)){
  print(c("Forward Stepwise",j))
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Combine Predictors
  predictor_formula <- paste(predictors, collapse = " + ")
  
  ## Full Model Formula
  formula_step <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  ## Stepwise Selection
  
  fit_step<- regsubsets(formula_step,
                        data = clean_yearly_training, 
                        method = "forward", 
                        intercept = TRUE,
                        nvmax = 100,
                        really.big=T)
  
  # Get logicals indicating which predictors are selected
  logical_indicators <- as.logical(summary(fit_step)$which[which.min(summary(fit_step)$bic),])
  
  # Eliminate intercept term
  logical_indicators <- logical_indicators[2:length(logical_indicators)]
  
  # Get predictors that are useful from the original character vector
  selected_predictors <- predictors[logical_indicators]
  
  Appendix$`Number Forward Predictors`[j] <- length(selected_predictors)
  
  
  # Proceed as we did before to create a model formula
  ## Combine Predictors
  selected_predictor_formula <- paste(selected_predictors, collapse = " + ")
  
  ## Full Model Formula
  selected_formula_step <- as.formula(paste(category_name, selected_predictor_formula , sep = " ~ "))
  
  ## Proceed to LOOCV
  mses <- rep(NA,nrow(clean_yearly_training))
  
  for(i in 1:nrow(clean_yearly_training)){
    train = clean_yearly_training[-i,]
    test = clean_yearly_training[i,]
    model_lm <- lm(selected_formula_step, data= train)
    predicted <- as.numeric(predict.lm(model_lm, test))
    actual <- clean_yearly_training[i,category_name]
    
    mses[i] <- (predicted - actual)^2
  }
  Forward_List[[j]] <- selected_formula_step
  Appendix$Forward_CV[j] <- mean(mses, na.rm = TRUE)
}

# Cross Validation for Backward Stepwise Linear Regression -------------------------

for(j in 1:length(categories)){
  print(c("Backward Stepwise",j))
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Combine Predictors
  predictor_formula <- paste(predictors, collapse = " + ")
  
  ## Full Model Formula
  formula_step <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  ## Stepwise Selection
  
  fit_step<- regsubsets(formula_step,
                        data = clean_yearly_training, 
                        method = "backward", 
                        intercept = TRUE,
                        nvmax = 100,
                        really.big=T)
  
  # Get logicals indicating which predictors are selected
  logical_indicators <- as.logical(summary(fit_step)$which[which.min(summary(fit_step)$bic),])
  
  # Eliminate intercept term
  logical_indicators <- logical_indicators[2:length(logical_indicators)]
  
  # Get predictors that are useful from the original character vector
  selected_predictors <- predictors[logical_indicators]
  
  Appendix$`Number Backward Predictors`[j] <- length(selected_predictors)
  
  
  # Proceed as we did before to create a model formula
  ## Combine Predictors
  selected_predictor_formula <- paste(selected_predictors, collapse = " + ")
  
  ## Full Model Formula
  selected_formula_step <- as.formula(paste(category_name, selected_predictor_formula , sep = " ~ "))
  
  ## Proceed to LOOCV
  mses <- rep(NA,nrow(clean_yearly_training))
  
  for(i in 1:nrow(clean_yearly_training)){
    train = clean_yearly_training[-i,]
    test = clean_yearly_training[i,]
    model_lm <- lm(selected_formula_step, data= train)
    predicted <- as.numeric(predict.lm(model_lm, test))
    actual <- clean_yearly_training[i,category_name]
    
    mses[i] <- (predicted - actual)^2
  }
  Backward_List[[j]] <- selected_formula_step
  Appendix$Backward_CV[j] <- mean(mses, na.rm = TRUE)
}

# Cross Validation for Exhaustive Stepwise Linear Regression -------------------------

for(j in 1:length(categories)){
  print(c("Exhaustive Stepwise",j))
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Combine Predictors
  predictor_formula <- paste(predictors, collapse = " + ")
  
  ## Full Model Formula
  formula_step <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  ## Stepwise Selection
  
  fit_step<- regsubsets(formula_step,
                        data = clean_yearly_training, 
                        method = "exhaustive", 
                        intercept = TRUE,
                        nvmax = 100,
                        really.big=T)
  
  # Get logicals indicating which predictors are selected
  logical_indicators <- as.logical(summary(fit_step)$which[which.min(summary(fit_step)$bic),])
  
  # Eliminate intercept term
  logical_indicators <- logical_indicators[2:length(logical_indicators)]
  
  # Get predictors that are useful from the original character vector
  selected_predictors <- predictors[logical_indicators]
  
  Appendix$`Number Exhaustive Predictors`[j] <- length(selected_predictors)
  
  
  # Proceed as we did before to create a model formula
  ## Combine Predictors
  selected_predictor_formula <- paste(selected_predictors, collapse = " + ")
  
  ## Full Model Formula
  selected_formula_step <- as.formula(paste(category_name, selected_predictor_formula , sep = " ~ "))
  
  ## Proceed to LOOCV
  mses <- rep(NA,nrow(clean_yearly_training))
  
  for(i in 1:nrow(clean_yearly_training)){
    train = clean_yearly_training[-i,]
    test = clean_yearly_training[i,]
    model_lm <- lm(selected_formula_step, data= train)
    predicted <- as.numeric(predict.lm(model_lm, test))
    actual <- clean_yearly_training[i,category_name]
    
    mses[i] <- (predicted - actual)^2
  }
  Exhaustive_List[[j]] <- selected_formula_step
  
  Appendix$Exhaustive_CV[j] <- mean(mses, na.rm = TRUE)
}

# Cross Validation for Principal Components Regression  -------------------------

set.seed(4272017)
for(j in 1:length(categories)){
  print(c("PCR",j))
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Combine Predictors
  predictor_formula <- paste(predictors, collapse = " + ")
  
  ## Full Model Formula
  formula_pcr <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  pcr_fit <- pcr(formula_pcr, data=clean_yearly_training,
                 validation="LOO", scale=TRUE)
  
  Appendix$`Number PCR Components`[j] <- which.min(pcr_fit$validation$adj)
  Appendix$PCR_CV[j] <- min(pcr_fit$validation$adj)
}

# Cross Validation for Partial Least Squares Regression -------------------
set.seed(4272018)
for(j in 1:length(categories)){
  print(c("PLSR",j))
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Combine Predictors
  predictor_formula <- paste(predictors, collapse = " + ")
  
  ## Full Model Formula
  formula_plsr <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  plsr_fit <- plsr(formula_plsr, data=clean_yearly_training,
                 validation="LOO", scale=TRUE)
  
  Appendix$`Number PLSR Components`[j] <- which.min(plsr_fit$validation$adj)
  Appendix$PLSR_CV[j] <- min(plsr_fit$validation$adj)
}

# Cross Validation for Ridge  -------------------
set.seed(427)
for(j in 1:length(categories)){
  print(c("Ridge",j))
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Get Design Matrix
  X <- as.matrix(clean_yearly_training[,predictors])
  Y <- clean_yearly_training[,category_name]

  ridge_mods <- cv.glmnet(X,Y,alpha=0,lambda=seq(0,40,by=.01))
  

  Appendix$`Ridge Lambda`[j] <- ridge_mods$lambda[which.min(ridge_mods$cvm)]
  Appendix$Ridge_CV[j] <- min(ridge_mods$cvm)
}

# Cross Validation for Lasso  -------------------
set.seed(427345)
for(j in 1:length(categories)){
  print(c("Lasso",j))
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Get Design Matrix
  X <- as.matrix(clean_yearly_training[,predictors])
  Y <- clean_yearly_training[,category_name]
  
  lasso_mods <- cv.glmnet(X,Y,alpha=1,lambda=seq(0,10,by=.01))
  
  
  Appendix$`Lasso Lambda`[j] <- lasso_mods$lambda[which.min(lasso_mods$cvm)]
  Appendix$Lasso_CV[j] <- min(lasso_mods$cvm)
}

# Cross Validation for KNN -------------------
set.seed(12356)
for(j in 1:length(categories)){
  print(c("KNN",j))
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Get Design Matrix
  X <- scale(as.matrix(clean_yearly_training[,predictors]), center = TRUE, scale = TRUE)
  
  ## Get Dependent Variable
  Y <- clean_yearly_training[,category_name]
  
  ## Maximum K to examine
  K_Max <- 100
  
  # Get Data frame to store estimated values of test MSE for each K
  K_Store <- data.frame(1:K_Max,rep(NA,2))
  colnames(K_Store) <- c("k","MSE")
  
  # Loop over K's
  for(k in 1:K_Max){
  MSEs <- rep(NA,nrow(X)) # Create vector to save individual estimates of Test MSE
  
  # LOOCV for KNN
  for(i in 1:nrow(X)){
  knn_mod <- knn.reg(train = X[-i, ],
                     test = X[i, ],
                     y = Y[-i], k = k, algorithm = "brute")
  MSEs[i] <- (Y[i]-knn_mod$pred)^2
  }
  K_Store[k,1] <- k
  K_Store[k,2] <- mean(MSEs)
  }
  # Save data
  Appendix$`KNN K`[j] <- K_Store$k[which.min(K_Store$MSE)]
  Appendix$KNN_CV[j] <- min(K_Store$MSE)
}

# Cross Validation for Random Forests -------------------------------------
set.seed(1235638)
for(j in 1:length(categories)){
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Get the Model Formula
  category_name <- categories[j]
  
  ## Combine Predictors
  predictor_formula <- paste(predictors, collapse = " + ")
  
  ## Full Model Formula
  formula_randf <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  ## Save MSEs based on mTRY
  mses_rf_pts <- rep(NA,24)

  ## Number of folds
  num_fols <- 10
  ## Loop through different mtry
  
    for(m in 1:24){
    MSE_m <- rep(NA,num_fols)

    # Create folds for 10 fold cross validation
    flds <- createFolds(clean_yearly_training[,category_name], k = num_fols, list = TRUE, returnTrain = FALSE)
    names(flds)[1] <- "train"
    
    for(i in 1:num_fols){
      print(c("Random Forest",j,m,i))
      
    modPts <- randomForest(formula_randf, data=clean_yearly_training[-flds[[i]],], mtry=m)
    rfPredPts <- predict(modPts, clean_yearly_training[flds[[i]],])
 
    MSE_m[i] <- mean((rfPredPts - clean_yearly_training[flds[[i]],category_name])^2)
  }  
    mses_rf_pts[m] <- mean(MSE_m)
    }
  
  # Save data
  Appendix$`Rand_For MTry`[j] <- which.min(mses_rf_pts)
  Appendix$Rand_For_CV[j] <- min(mses_rf_pts)
}

# Cross Validation for Boosting  -------------------
set.seed(42704321)
for(j in 1:length(categories)){
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Combine Predictors
  predictor_formula <- paste(predictors, collapse = " + ")
  formula_predict <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  ## Parameter Values to Cross Validate
  boost_tune <- expand.grid(nt=seq(1000,10000,by=1000),id=1:5)
  boost_tune$MSE <- NA
  ## Full Model Formula
  formula_boost <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  ## Set the Number of folds
  num_folds <- 10
  
  for(i in 1:nrow(boost_tune)){
    flds <- createFolds(clean_yearly_training[,category_name], k = num_folds, list = TRUE, returnTrain = FALSE)
    names(flds)[1] <- "train"
    Save_MSEs <- rep(NA,num_folds)
    for(k in 1:num_folds){
    print(c("Boosting",j,i))
    boost_mod <- gbm(formula_predict, data=clean_yearly_training[-flds[[k]],] , 
                     n.trees=boost_tune$nt[i], distribution = "gaussian",
                     interaction.depth = boost_tune$id[i])
    Boost_Preds <- predict(boost_mod, clean_yearly_training[flds[[k]],], n.trees=boost_tune$nt[i])
    Save_MSEs[k] <- mean((Boost_Preds - clean_yearly_training[flds[[k]],category_name])^2)
    }
    boost_tune$MSE[i] <- mean(Save_MSEs)
  }
  
  Appendix$Boosting_NT[j] <- boost_tune[which.min(boost_tune$MSE),1]
  Appendix$Boosting_id[j] <- boost_tune[which.min(boost_tune$MSE),2]
  
  Appendix$Boosting_CV[j] <- min(boost_tune$MSE)
}

# Cross Validaiton for the Mean Model -------------------------------------
set.seed(4222222222)

## Set the Number of folds
num_folds <- 10

for(j in 1:length(categories)){
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Combine Predictors
  predictor_formula <- paste(predictors, collapse = " + ")
  formula_predict <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  ## Save Estimates of Test MSE
  MSEs <- rep(NA,num_folds)
  
  ## Create Folds
  flds <- createFolds(clean_yearly_training[,category_name], k = num_folds, list = TRUE, returnTrain = FALSE)

    ## Loop through the folds
  for(i in 1:num_folds){
    print(c("Mean Model",j,i))
    train_set <- clean_yearly_training[-flds[[i]],]
    test_set <- clean_yearly_training[flds[[i]],]
    X <- as.matrix(train_set[,predictors])
    X_Test <- as.matrix(test_set[,predictors])
    Y <- train_set[,category_name]
        ## Fit Models that I need to fit in order to get predictions
    # Fit Best Forward Stepwise Model
    Best_Forward <- lm(Forward_List[[j]], data= train_set)
    
    # Fit Best Backward Stepwise Model
    Best_Backward <- lm(Backward_List[[j]], data= train_set)
    
    # Fit Best Exhaustive 
    Best_Exhaustive <- lm(Exhaustive_List[[j]], data= train_set)
    
    # Fit Best Ridge
    Best_Ridge <- lm.ridge(formula_predict, data= train_set, lambda = Appendix$`Ridge Lambda`[j])
    

    # Fit Best Lasso
    Best_Lasso <- glmnet(X,Y,alpha=1,lambda= Appendix$`Lasso Lambda`[j]) 
    
    # Best KNN
    Best_KNN <- knn.reg(train = X,
                       test = X_Test,
                       y = Y, k = Appendix$`KNN K`[j], algorithm = "brute")
    # Best Random Forest
    Best_Forest <- randomForest(formula_predict , data=train_set, mtry=Appendix$`Rand_For MTry`[j])
    
    # Best Boosted
    Best_Boost <- gbm(formula_predict, data=train_set , 
                                       n.trees=Appendix$Boosting_NT[j], distribution = "gaussian",
                                       interaction.depth = Appendix$Boosting_id[j])
    # Get Predictions for each Model
    
    Predictions <- data.frame(cbind(predict(Best_Forward,test_set),
                              predict(Best_Backward,test_set),
                              predict(Best_Exhaustive,test_set),
                              predict(pcr_fit, ncomp = Appendix$`Number PCR Components`[j], newdata=test_set),
                              predict(plsr_fit, ncomp = Appendix$`Number PLSR Components`[j], newdata=test_set),
                              as.matrix(cbind(const=1,X)) %*% coef(Best_Ridge),
                              as.numeric(predict(Best_Lasso,as.matrix(test_set[,predictors]))),
                              Best_KNN$pred,
                              predict(Best_Forest, test_set),
                              predict(Best_Boost, test_set, n.trees=Appendix$Boosting_NT[j])))
    
    colnames(Predictions) <- c("Forward","Backward","Exhaustive","PCR","PLSR",
                               "Ridge","Lasso","KNN","Random Forest","Boost")
    
    Mean_Predictions <- as.numeric(rowMeans(Predictions))
    MSEs[i] <- mean((Mean_Predictions-train_set[,category_name])^2)
  }
  Appendix$`Mean CV`[j] <- mean(MSEs)
}

# Cross Validaiton for the Median Model -------------------------------------
set.seed(422222)

## Set the Number of folds
num_folds <- 10

for(j in 1:length(categories)){
  ## Dependent Variable
  category_name <- categories[j]
  
  ## Combine Predictors
  predictor_formula <- paste(predictors, collapse = " + ")
  formula_predict <- as.formula(paste(category_name, predictor_formula , sep = " ~ "))
  
  ## Save Estimates of Test MSE
  MSEs <- rep(NA,num_folds)
  
  ## Create Folds
  flds <- createFolds(clean_yearly_training[,category_name], k = num_folds, list = TRUE, returnTrain = FALSE)
  
  ## Loop through the folds
  for(i in 1:num_folds){
    print(c("Mean Model",j,i))
    train_set <- clean_yearly_training[-flds[[i]],]
    test_set <- clean_yearly_training[flds[[i]],]
    X <- as.matrix(train_set[,predictors])
    X_Test <- as.matrix(test_set[,predictors])
    Y <- train_set[,category_name]
    ## Fit Models that I need to fit in order to get predictions
    # Fit Best Forward Stepwise Model
    Best_Forward <- lm(Forward_List[[j]], data= train_set)
    
    # Fit Best Backward Stepwise Model
    Best_Backward <- lm(Backward_List[[j]], data= train_set)
    
    # Fit Best Exhaustive 
    Best_Exhaustive <- lm(Exhaustive_List[[j]], data= train_set)
    
    # Fit Best Ridge
    Best_Ridge <- lm.ridge(formula_predict, data= train_set, lambda = Appendix$`Ridge Lambda`[j])
    
    
    # Fit Best Lasso
    Best_Lasso <- glmnet(X,Y,alpha=1,lambda= Appendix$`Lasso Lambda`[j]) 
    
    # Best KNN
    Best_KNN <- knn.reg(train = X,
                        test = X_Test,
                        y = Y, k = Appendix$`KNN K`[j], algorithm = "brute")
    # Best Random Forest
    Best_Forest <- randomForest(formula_predict , data=train_set, mtry=Appendix$`Rand_For MTry`[j])
    
    # Best Boosted
    Best_Boost <- gbm(formula_predict, data=train_set , 
                      n.trees=Appendix$Boosting_NT[j], distribution = "gaussian",
                      interaction.depth = Appendix$Boosting_id[j])
    # Get Predictions for each Model
    
    Predictions <- data.frame(cbind(predict(Best_Forward,test_set),
                                    predict(Best_Backward,test_set),
                                    predict(Best_Exhaustive,test_set),
                                    predict(pcr_fit, ncomp = Appendix$`Number PCR Components`[j], newdata=test_set),
                                    predict(plsr_fit, ncomp = Appendix$`Number PLSR Components`[j], newdata=test_set),
                                    as.matrix(cbind(const=1,X)) %*% coef(Best_Ridge),
                                    as.numeric(predict(Best_Lasso,as.matrix(test_set[,predictors]))),
                                    Best_KNN$pred,
                                    predict(Best_Forest, test_set),
                                    predict(Best_Boost, test_set, n.trees=Appendix$Boosting_NT[j])))
    
    colnames(Predictions) <- c("Forward","Backward","Exhaustive","PCR","PLSR",
                               "Ridge","Lasso","KNN","Random Forest","Boost")
    
    Median_Predictions <- rowMedians(as.matrix(Predictions))
    
    MSEs[i] <- mean((Median_Predictions-train_set[,category_name])^2)
  }
  Appendix$`Median CV`[j] <- mean(MSEs)
}

## Generate predictions for individual categories

#------------------------------------------------TOTAL POINTS-----------------------------------------------

## Combine Predictors
predictorFormula <- paste(predictors, collapse = " + ")

predictors15 <- c("age_15","games_15","games_started_15", "minutes_played_15", 
                "field_goal_15", "field_goal_attempts_15", "field_goal_pct_15", 
                "three_point_15", "three_point_attempts_15", "three_point_pct_15", 
                "two_point_15", "two_point_attempts_15", "two_point_pct_15", 
                "free_throw_15", "free_throw_attempts_15", "free_throw_pct_15", 
                "offense_rebound_15", "defense_rebound_15", "assists_15", 
                "steals_15", "blocks_15", "turnovers_15", "personal_foul_15", 
                "total_points_15", "Heights")


## Full model formula for predicting total points
formula_totPts_pcr <- as.formula(paste("total_points_15", predictorFormula, sep = " ~ "))

# Fit PCR model
totPts.fit = pcr(formula_totPts_pcr, data = clean_yearly_training, validation="LOO", scale=TRUE)

# Show summary
summary(totPts.fit)

# Number of pcr components
ncPts = which.min(totPts.fit$validation$adj)

# PCR_CV
min(totPts.fit$validation$adj)

# Make predictions
totPts.pred <- predict(totPts.fit, ncomp = ncPts, newData = clean_yearly_training[predictors15])
totPts.pred
#------------------------------------------------THREE POINTS-----------------------------------------------

## Full model formula for predicting three points
formula_threePts_pcr <- as.formula(paste("three_point_15", predictorFormula, sep = " ~ "))

# Fit PCR model
threePts.fit = pcr(formula_threePts_pcr, data = clean_yearly_training, validation="LOO", scale=TRUE)

# Show summary
summary(threePts.fit)

# Number of pcr components
nc3pt = which.min(threePts.fit$validation$adj)

# PCR_CV
min(threePts.fit$validation$adj)

# Make predictions
threePts.pred = predict(threePts.fit, ncomp = nc3pt, newData = clean_yearly_training[predictors15])
threePts.pred
#------------------------------------------------BLOCKS-----------------------------------------------

## Full model formula for predicting total points
formula_blocks_pcr <- as.formula(paste("blocks_15", predictorFormula, sep = " ~ "))

# Fit PCR model
blocks.fit = pcr(formula_blocks_pcr, data = clean_yearly_training, validation="LOO", scale=TRUE)

# Show summary
summary(blocks.fit)

# Number of pcr components
ncBlk = which.min(blocks.fit$validation$adj)

# PCR_CV
min(blocks.fit$validation$adj)

# Make predictions
blocks.pred = predict(blocks.fit, ncomp = ncBlk, newData = clean_yearly_training[predictors15])
blocks.pred
#------------------------------------------------FREE-THROW %-----------------------------------------------

## Combine Predictors
predictor_formula <- paste(predictors, collapse = " + ")

## Full Model Formula
formula_randf <- as.formula(paste("free_throw_pct_15", predictor_formula , sep = " ~ "))

## Save MSEs based on mTRY
mses_rf_FThrow <- rep(NA,24)
length(predictors)
## Loop through different mtry
for(m in 1:24){
  MSE_m <- rep(NA,10)
  # Create folds for 10 fold cross validation
  flds <- createFolds(clean_yearly_training[,"free_throw_pct_15"], k = 10, list = TRUE, returnTrain = FALSE)
  names(flds)[1] <- "train"
  
  for(i in 1:10){
    modFThrow <- randomForest(formula_randf, data=clean_yearly_training[-flds[[i]],], mtry=m)
    rfPredFThrow <- predict(modFThrow, clean_yearly_training[flds[[i]],])
    MSE_m[i] <- mean((rfPredFThrow - clean_yearly_training[flds[[i]],"free_throw_pct_15"])^2)
    }  
    mses_rf_FThrow[m] <- mean(MSE_m)
}
  
# RF_CV
min(mses_rf_FThrow)

# Identify min
m = which.min(mses_rf_FThrow)
m

# Fit RandomForest model 
FThrow.fit = randomForest(formula_randf, data=clean_yearly_training, mtry=m)

# Make predictions
FThrow.pred = predict(FThrow.fit, clean_yearly_training)

#------------------------------------------------ REBOUNDS -----------------------------------------------

## Full model formula for predicting Rebounds
formula_plsr <- as.formula(paste("total_reb_15", predictor_formula , sep = " ~ "))

rebounds.fit <- plsr(formula_plsr, data=clean_yearly_training,
                 validation="LOO", scale=TRUE)

ncRb = which.min(rebounds.fit$validation$adj)

min(rebounds.fit$validation$adj)

# Make predictions
rebounds.pred = predict(rebounds.fit, ncomp = ncRb, newData = clean_yearly_training[predictors15])
rebounds.pred

#------------------------------------------------ STEALS -----------------------------------------------

## Full model formula for predicting Rebounds
formula_plsr <- as.formula(paste("steals_15", predictor_formula , sep = " ~ "))

steals.fit <- plsr(formula_plsr, data=clean_yearly_training,
                     validation="LOO", scale=TRUE)

ncRb = which.min(steals.fit$validation$adj)

min(steals.fit$validation$adj)

# Make predictions
steals.pred = predict(steals.fit, ncomp = ncRb, newData = clean_yearly_training[predictors15])


#------------------------------------------------ ASSISTS -----------------------------------------------

## Full model formula for predicting Rebounds
formula_plsr <- as.formula(paste("assists_15", predictor_formula , sep = " ~ "))

assists.fit <- plsr(formula_plsr, data=clean_yearly_training,
                     validation="LOO", scale=TRUE)

ncAs = which.min(assists.fit$validation$adj)

min(assists.fit$validation$adj)

# Make predictions
assists.pred = predict(assists.fit, ncomp = ncAs, newData = clean_yearly_training[predictors15])
assists.pred 

# Write to CSV file
x <- data.frame(cbind(clean_yearly_training$player_14, totPts.pred, threePts.pred, FThrow.pred, blocks.pred, rebounds.pred, steals.pred, assists.pred))
write.csv(x, file="Performance.csv", row.names=FALSE)


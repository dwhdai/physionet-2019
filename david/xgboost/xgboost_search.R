#'   ***********************************************************************
#'   ***********************************************************************
#'   ***********************************************************************
#'   Script name: xgboost_search
#'   Initially created by: David Dai
#'   Date created: June 20, 2019
#'   Maintainer information: daiw@smh.ca
#'
#'   Script contents: This script contains the function to perform a random
#'   grid search for the xgboost model.
#'
#'   ***********************************************************************
#'   ***********************************************************************
#'   ***********************************************************************

grid_search <- function(train, valid, n_param_set, ncluster, params,
                        train_outcome, valid_outcome, model_vars, split){
  
  require(doParallel)
  require(xgboost)
  require(tidyverse)
  require(caret)
  
  # conflict_prefer("lift", "caret")
  
  cl <- makePSOCKcluster(ncluster)
  registerDoParallel(cl)
  
  results <-  foreach(i = 1:n_param_set, 
                      .packages = c("xgboost", "tidyverse", "caret"), 
                      .combine = rbind) %dopar% {
                        
                        # Create xgb matrices
                        dtrain <- xgb.DMatrix(
                          data = train %>% 
                            select(one_of(as.character(model_vars))) %>%
                            as.matrix(),
                          label =  train_outcome %>% 
                            as.matrix())
                        
                        dvalid <- xgb.DMatrix(
                          data = valid %>% 
                            select(one_of(as.character(model_vars)))%>%
                            as.matrix(),
                          label = valid_outcome %>%
                            as.matrix())
                        
                        # Set train/evaluation sets for monitoring
                        watchlist <- list(train = dtrain, eval = dvalid)
                        
                        fit <- xgboost::xgb.train(params = params[i,], 
                                                  data = dtrain, 
                                                  nrounds = params[i,]$nrounds,
                                                  watchlist,
                                                  early_stopping_rounds = 10,
                                                  metrics = list("logloss"),
                                                  verbose = F,
                                                  # print_every_n = 10,
                                                  maximize = F)    
                        
                        res <- c(fit$params %>% unlist,
                                 fit$best_score,
                                 split = split)
                        
                      }
  
  stopCluster(cl)
  return(results)
}































# Alternative way to run grid search

# tictoc::tic()
# 
# debug_msg("Refitting best model")
# 
# dtrain <- xgb.DMatrix(
#     data = train %>% 
#         select(one_of(model_vars))%>%
#         as.matrix(),
#     label =  train_outcome %>% 
#         as.matrix())
# 
# dvalid <- xgb.DMatrix(
#     data = valid %>% 
#         select(one_of(model_vars))%>%
#         as.matrix(),
#     label = valid_outcome %>%
#         as.matrix())
# 
# for (i in 1:n_param_set){
#     
#     # Set train/evaluation sets for monitoring
#     watchlist <- list(train = dtrain, eval = dvalid)
#     
#     fit <- xgboost::xgb.train(params = params[i,], 
#                               data = dtrain, 
#                               nrounds = params[i,]$nrounds,
#                               watchlist,
#                               early_stopping_rounds = 10,
#                               metrics = list("logloss"),
#                               verbose = T,
#                               print_every_n = 10,
#                               maximize = F)   
# }
# 
# tictoc::toc()
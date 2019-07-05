#! /usr/bin/env Rscript
# !C:\Users\daiw\Documents\R\R-3.6.0\bin\Rscript.exe --vanilla
# NOTE: run chmod +x scriptname.R to allow script to be executable

library(argparser)
library(xgboost)
suppressMessages(library(tidyverse))
suppressMessages(library(here))
library(conflicted)

conflict_prefer("lag", "dplyr")
conflict_prefer("filter", "dplyr")
conflict_prefer("slice", "dplyr")

# Add arguments -----------------------------------------------------------

p <- arg_parser("parser name")

# Datasets
p <- add_argument(p, "--train", help = "Training dataset")
p <- add_argument(p, "--valid", help = "Validation dataset")

# Outcome label
p <- add_argument(p, "--outcome", help = "Name of outcome variable")

# Add a debug flag
p <- add_argument(p, "--debug", help="Enable debug mode", flag=TRUE)

# Add a flag to tune hyperparameters
p <- add_argument(p, "--search", help = "Hyperparameter search", flag = TRUE)

# YAML file containing hyperparameters for search
p <- add_argument(p, "--yaml", help = "YAML file containing hyperparameters")

# Parse arguments ---------------------------------------------------------

args <- parse_args(p)

if (!is.na(args$yaml)){
    args <- suppressWarnings(yaml::yaml.load_file(args$yaml))
}

# function to display debug output
debug_msg <- function(...){
    if (isTRUE(args$debug)){
        cat(paste0("DEBUG: ",...,"\n"))
    }
}

debug_msg("Debug mode is ON")


# Import data -------------------------------------------------------------

train <- 
    # read_csv(here("physionet_data", "small_dataset", "train.csv"))
    read_csv(args$train) 
  
valid <-   
  # read_csv(here("physionet_data", "small_dataset", "test.csv"))
    read_csv(args$valid) 

# Variables to include in model
model_vars <- train %>% 
    select(-one_of("subject", "SepsisLabel", "EtCO2", "Bilirubin_direct")) %>% 
    names()

# Parse outcome
train_outcome <- eval(parse(text = paste0("train$", args$outcome)))
valid_outcome <- eval(parse(text = paste0("valid$", args$outcome)))

# Create hyperparameter grid ----------------------------------------------

parser <- function(x){
    eval(parse(text = x))
}

if (args$search == "yes"){
    
    # Create hyperparameter grid
    params <- expand.grid(objective = "binary:logistic",
                          booster = "gbtree",
                          eval_metric = "logloss",
                          nrounds = args$nrounds,
                          max_depth = parser(args$max_depth),
                          eta = parser(args$eta),
                          subsample = parser(args$subsample),
                          colsample_bytree = parser(args$colsample_bytree),
                          min_child_weight = parser(args$min_child_weight),
                          max_delta_step = parser(args$max_delta_step)) %>% 
        filter(max_depth == 4 & min_child_weight == 1 |
                   max_depth == 10 & min_child_weight == 10 |
                   max_depth == 7) %>% 
      slice(1:2)
    
    # Run grid search here
    source(here("david", "xgboost", "xgboost_search.R"))
    tictoc::tic()
    
    # Loop through all 10 splits
    for (i in 1:1){
      results <- grid_search(train,
                             valid,
                             n_param_set = nrow(params),
                             ncluster =  args$ncluster,
                             params,
                             train_outcome,
                             valid_outcome,
                             model_vars,
                             split = i)
    }
    
    debug_msg("Grid search complete.")
    
    tictoc::toc()
    
    # Return best params here
    results_tb <- results %>% 
        as_tibble() %>% 
        janitor::clean_names() %>% 
        arrange(eval_logloss) %>% 
        select(-objective, -booster, -eval_metric, -metrics) %>% 
        mutate_all(as.numeric)
    
    # Export search results
    write_csv(results_tb, parser(args$search_results))
    debug_msg("Hyperparameter search results saved at ",
              args$search_results)
    
    best_params <- results_tb %>% 
        slice(1)
    
    final_params <- list(objective = "binary:logistic",
                         booster = "gbtree",
                         eval_metric = "logloss",
                         nrounds = best_params$nrounds,
                         max_depth = best_params$max_depth,
                         eta = best_params$eta,
                         subsample = best_params$subsample,
                         colsample_bytree = best_params$colsample_bytree,
                         min_child_weight = best_params$min_child_weight,
                         max_delta_step = best_params$max_delta_step)
    
} else {
    
    final_params <- list(objective = "binary:logistic",
                         booster = "gbtree",
                         eval_metric = "logloss",
                         nrounds = args$nrounds,
                         max_depth = args$max_depth,
                         eta = args$eta,
                         subsample = args$subsample,
                         colsample_bytree = args$colsample_bytree,
                         min_child_weight = args$min_child_weight,
                         max_delta_step = args$max_delta_step)
}

# Fit final model ---------------------------------------------------------

# Set up xgboost matrices -------------------------------------------------

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

# Train model -------------------------------------------------------------

debug_msg("Training final model.")

# Set train/evaluation sets for monitoring
watchlist <- list(train = dtrain, eval = dvalid)

final_fit <- xgb.train(params = final_params,
                       data = dtrain,
                       nrounds = final_params$nrounds,
                       watchlist,
                       early_stopping_rounds = 10,
                       metrics = list("logloss"),
                       verbose = T,
                       print_every_n = 10,
                       maximize = F)

xgb.save(final_fit, parser(args$model))

debug_msg("Final model saved at ",
          args$model)

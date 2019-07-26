#!/usr/bin/Rscript

get_sepsis_score = function(data, model){
    require(magrittr)
    # data <- current %>% 
    #     as.data.frame() %>% 
    #     select(-one_of("subject", "SepsisLabel", "EtCO2", "Bilirubin_direct"))
    # xgb.mat <- xgb.DMatrix(data = data.matrix(current_data))
    
    # data_last_row <- data[nrow(data),]
    
    score <- predict(model, data)[nrow(data)]
    
    # score = 1 - exp(-l_exp_bx)
    label = score > 0.1
    predictions = c(score, label)
    return(predictions)
}

load_sepsis_model = function(){
    model <- xgboost::xgb.load(here::here("david", "xgboost", "xgboost.model"))
    return(model)
}
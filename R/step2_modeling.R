################################################################################
# Step 2: Modeling
################################################################################

################################################################################
# Setup
################################################################################

# Set working directory
setwd(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))

# Libraries
library(tidyverse)

# Load Data
banking_churn <- data.table::fread(file.path("data", "modeling_data.csv"))

################################################################################
# Feature transformation
################################################################################

scaling_variables <- banking_churn %>% 
  dplyr::select(c('credit_score', 'age', 'balance', 'estimated_salary')) %>% 
  scale(.) %>% 
  data.frame() %>% 
  rename(credit_score_scaled = credit_score,
         age_scaled = age,
         balance_scaled = balance,
         estimated_salary_scaled = estimated_salary)

modeling_df <- banking_churn %>% 
  mutate(geography_factor = factor(geography),
         gender_factor = factor(gender),
         exited_factor = factor(ifelse(exited == 0, 'remain', 'churn')),
         tenure_factor = factor(tenure, 
                                levels = 0:10, 
                                labels = c("Zero Year", "One Year", "Two Year", "Three Year",
                                           "Four Year", "Five Year", "Six Year", 
                                           "Seven Year", "Eight Year", "Nine Year",
                                           "Ten Year"), 
                                ordered = TRUE),
         num_of_products_factor = factor(num_of_products, 
                                         levels = 1:4, 
                                         labels = c("One Product", "Two Products", "Three Products", "Four Products"), 
                                         ordered = TRUE),
         has_cr_card_factor = factor(has_cr_card),
         is_active_member_factor = factor(is_active_member),
         ) %>% 
  cbind(., scaling_variables) %>% 
  dplyr::select(customer_id, credit_score_scaled, estimated_salary_scaled, 
                age_scaled, balance_scaled, geography_factor, gender_factor,
                tenure_factor, num_of_products_factor, has_cr_card_factor,
                is_active_member_factor, exited_factor)

################################################################################
# Create train-test splits
################################################################################

train_split <- 0.8
test_split <- 0.2

set.seed(42)
train_ids <- sample(modeling_df$customer_id, train_split * nrow(modeling_df))
test_ids <- setdiff(modeling_df$customer_id, train_ids)

training_data <- modeling_df %>% 
  filter(customer_id %in% train_ids)

testing_data <- modeling_df %>% 
  filter(customer_id %in% test_ids)

print(paste0("Number of IDs in training set: ", length(train_ids)))
print(paste0("Number of IDs in testing set: ", length(test_ids)))

################################################################################
# Model 1: Dummy Mode
################################################################################

outcome_mode <- names(which.max(table(modeling_df$exited_factor)))

prediction_df <- testing_data %>% 
  dplyr::select(customer_id, exited_factor) %>% 
  mutate(dummy_pred = outcome_mode)

prediction_df$dummy_pred <- factor(prediction_df$dummy_pred, levels = levels(testing_data$exited_factor))

################################################################################
# Model 2: Logistic Regression
################################################################################

# Define the trainControl object for 5-fold cross-validation
train_control <- caret::trainControl(
  method = "cv", 
  number = 5, 
  savePredictions = TRUE, 
  classProbs = TRUE
)

# Fit the logistic regression model using caret
model <- caret::train(
  exited_factor ~ . -customer_id,
  data = training_data,
  method = "glm",
  family = binomial,
  trControl = train_control
)

# Print the model summary
print(model)

# Extract the coefficients from the trained model
simple_lr_coefficients <- data.frame(summary(model$finalModel)$coefficients) %>% 
  janitor::clean_names(.)
print(simple_lr_coefficients)

# Predict on test data using caret
predictions <- predict(model, newdata = testing_data)
prediction_df$simple_lr_pred = predictions

################################################################################
# Model 3: Ridge Regression
################################################################################

# Define the tuning grid for lambda
tune_grid <- expand.grid(.alpha = 0, .lambda = seq(0.001, 0.1, by = 0.001))

# Prepare the data matrix and response vector
x_train <- model.matrix(exited_factor ~ . -customer_id, data = training_data)[, -1]
y_train <- training_data$exited_factor

# Fit the Ridge regression model using caret and glmnet
ridge_model <- caret::train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = tune_grid,
  family = "binomial"
)

# Extract the best tuning parameters
best_lambda <- ridge_model$bestTune$lambda
best_alpha <- ridge_model$bestTune$alpha

# Fit the final Ridge regression model using the best lambda
final_ridge_model <- glmnet::glmnet(
  x = x_train,
  y = y_train,
  alpha = best_alpha,
  lambda = best_lambda,
  family = "binomial"
)

# Extract coefficients
ridge_coefficients <- coef(final_ridge_model)

# Convert the coefficients to a data frame
ridge_coefficients_df <- as.data.frame(as.matrix(ridge_coefficients))

# Prepare the test data matrix
x_test <- model.matrix(exited_factor ~ . -customer_id, data = testing_data)[, -1]

# Predict on test data using the best tuned model
ridge_predictions <- predict(final_ridge_model, newx = x_test, s = best_lambda, type = "response")

# Convert probabilities to binary outcomes
ridge_predictions_binary <- ifelse(ridge_predictions > 0.5, 'remain', 'churn')

# Convert binary outcomes to factor with the same levels as the original outcome
ridge_predictions_factor <- factor(ridge_predictions_binary, levels = levels(testing_data$exited_factor))

prediction_df$ridge_lr_pred <-  ridge_predictions_factor

################################################################################
# Model 4: Lasso Regression
################################################################################

# Define the tuning grid for lambda
tune_grid <- expand.grid(.alpha = 1, .lambda = seq(0.001, 0.1, by = 0.001))

# Fit the Lasso regression model using caret and glmnet
lasso_model <- caret::train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = tune_grid,
  family = "binomial"
)

# Print the model summary
print(lasso_model)

# Extract the best tuning parameters
best_lambda <- lasso_model$bestTune$lambda
best_alpha <- lasso_model$bestTune$alpha

# Fit the final Lasso regression model using the best lambda
final_lasso_model <- glmnet::glmnet(
  x = x_train,
  y = y_train,
  alpha = best_alpha,
  lambda = best_lambda,
  family = "binomial"
)

# Extract coefficients
lasso_coefficients <- coef(final_lasso_model)

# Convert the coefficients to a data frame
lasso_coefficients_df <- as.data.frame(as.matrix(lasso_coefficients))

# Prepare the test data matrix
x_test <- model.matrix(exited_factor ~ . -customer_id, data = testing_data)[, -1]

# Predict on test data using the best tuned model
lasso_predictions <- predict(final_lasso_model, newx = x_test, s = best_lambda, type = "response")

# Convert probabilities to binary outcomes
lasso_predictions_binary <- ifelse(lasso_predictions > 0.5, 'remain', 'churn')

# Convert binary outcomes to factor with the same levels as the original outcome
lasso_predictions_factor <- factor(lasso_predictions_binary, levels = levels(testing_data$exited_factor))

prediction_df$lasso_lr_pred <-  lasso_predictions_factor

################################################################################
# Model 5: Elastic Net Regression
################################################################################

# Define the tuning grid for lambda
tune_grid <- expand.grid(.alpha = seq(0, 1, by = 0.1), .lambda = seq(0.001, 0.1, by = 0.001))

# Fit the elastic_net regression model using caret and glmnet
elastic_net_model <- caret::train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = tune_grid,
  family = "binomial"
)

# Print the model summary
print(elastic_net_model)

# Extract the best tuning parameters
best_lambda <- elastic_net_model$bestTune$lambda
best_alpha <- elastic_net_model$bestTune$alpha

# Fit the final elastic_net regression model using the best lambda
final_elastic_net_model <- glmnet::glmnet(
  x = x_train,
  y = y_train,
  alpha = best_alpha,
  lambda = best_lambda,
  family = "binomial"
)

# Extract coefficients
elastic_net_coefficients <- coef(final_elastic_net_model)

# Convert the coefficients to a data frame
elastic_net_coefficients_df <- as.data.frame(as.matrix(elastic_net_coefficients))

# Prepare the test data matrix
x_test <- model.matrix(exited_factor ~ . -customer_id, data = testing_data)[, -1]

# Predict on test data using the best tuned model
elastic_net_predictions <- predict(final_elastic_net_model, newx = x_test, s = best_lambda, type = "response")

# Convert probabilities to binary outcomes
elastic_net_predictions_binary <- ifelse(elastic_net_predictions > 0.5, 'remain', 'churn')

# Convert binary outcomes to factor with the same levels as the original outcome
elastic_net_predictions_factor <- factor(elastic_net_predictions_binary, levels = levels(testing_data$exited_factor))

prediction_df$elastic_net_lr_pred <-  elastic_net_predictions_factor

################################################################################
# Model 6: Decision Tree
################################################################################

# Define the tuning grid for complexity parameter (cp)
tune_grid <- expand.grid(.cp = seq(0.001, 0.05, by = 0.001))

# Fit the decision tree model using caret with cross-validation
decision_tree_model <- caret::train(
  exited_factor ~ . -customer_id,
  data = training_data,
  method = "rpart",
  trControl = train_control,
  tuneGrid = tune_grid
)

# Print the model summary
print(decision_tree_model)

rpart.plot::rpart.plot(decision_tree_model$finalModel)

# Predict on the test data
x_test <- model.matrix(exited_factor ~ . -customer_id, data = testing_data)[, -1]
decision_tree_predictions <- predict(decision_tree_model, newdata = testing_data)

prediction_df$decision_tree_pred <- decision_tree_predictions

################################################################################
# Model 7: Random Forest
################################################################################

# Define the tuning grid for mtry (number of variables randomly sampled as candidates at each split)
tune_grid <- expand.grid(.mtry = seq(1, ncol(training_data) - 2, by = 1)) # Excluding 'customer_id' and 'exited_factor'

# Fit the Random Forest model using caret with cross-validation
random_forest_model <- caret::train(
  exited_factor ~ . -customer_id,
  data = training_data,
  method = "rf",
  trControl = train_control,
  tuneGrid = tune_grid,
  importance = TRUE
)

# Plot the model to see the performance for different mtry values
plot(random_forest_model)

# Plot variable importance
var_imp <- varImp(random_forest_model, scale = FALSE)

# Predict on the test data
random_forest_predictions <- predict(random_forest_model, newdata = testing_data)
prediction_df$rf_pred <- random_forest_predictions

################################################################################
# Model 8: XGBoost
################################################################################

# Prepare the data matrix and response vector
x_train <- model.matrix(exited_factor ~ . -customer_id, data = training_data)[, -1]
y_train <- ifelse(training_data$exited_factor == "churn", "yes", "no")  # Convert to binary factor

# Calculate class weights
neg_pos_ratio <- sum(y_train == "no") / sum(y_train == "yes")

# Set up train control for 5-fold cross-validation using caret
train_control <- caret::trainControl(
  method = "cv",
  number = 5,
  summaryFunction = twoClassSummary,  # Necessary for ROC, Sensitivity, Specificity
  classProbs = TRUE  # Necessary for ROC, Sensitivity, Specificity
)

# Define a more extensive tuning grid for xgboost
tune_grid <- expand.grid(
  nrounds = c(50, 100, 200),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 0.1, 0.5),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.5, 0.7, 1)
)


# Fit the XGBoost model using caret with cross-validation
xgb_model <- caret::train(
  x = x_train,
  y = factor(y_train, levels = c("no", "yes")),  # Ensuring proper factor levels
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "Sensitivity",  # "Sensitivity" should also work, if properly set
  scale_pos_weight = 1/neg_pos_ratio
)

# Convert predictions to factor
xgb_predictions <- predict(xgb_model, newdata = x_test, type = "prob")[,2]

# Convert probabilities to binary predictions
xgb_predictions_binary <- factor(ifelse(xgb_predictions > 0.5, "churn", "remain"), levels=levels(modeling_df$exited_factor))
prediction_df$xgb_pred <- xgb_predictions_binary

################################################################################
# Create Coefficient DF
################################################################################

all_coefficients <- simple_lr_coefficients %>% 
  dplyr::select(estimate) %>% 
  mutate(variable = row.names(simple_lr_coefficients)) %>% 
  rename(simple_lr = estimate)

all_coefficients$ridge_lr <- ridge_coefficients[,1]
all_coefficients$lasso_lr <- lasso_coefficients[,1]
all_coefficients$elastic_net <- elastic_net_coefficients[,1]

all_coefficients <- all_coefficients %>% 
  dplyr::select(variable, everything())

row.names(all_coefficients) <- NULL

long_coefficients <- all_coefficients %>% 
  pivot_longer(., -variable, names_to = 'model', values_to = 'coefficient')

long_coefficients %>% 
  ggplot() +
    geom_point(aes(x = coefficient, y = variable, color = model), alpha=0.5) +
    labs(x = 'Coefficient',
         y = 'Variable',
         color = 'Model',
         title = 'Model Coefficients Comparisons')

################################################################################
# Performance
################################################################################

performance_df <- prediction_df %>% 
  pivot_longer(., -c(customer_id, exited_factor), names_to = 'model', values_to = 'prediction') %>% 
  mutate(correct = ifelse(exited_factor == prediction, 1, 0)) %>% 
  group_by(model) %>% 
  summarize(number_correct = sum(correct, na.rm=TRUE))

# Function to calculate metrics
calculate_metrics <- function(predictions, true_values) {
  confusion <- confusionMatrix(predictions, true_values)
  accuracy <- confusion$overall['Accuracy']
  f1 <- confusion$byClass['F1']
  recall <- confusion$byClass['Recall']
  precision <- confusion$byClass['Precision']
  
  table <- confusion$table
  
  return(list('confusion' = confusion,
              'performance' = data.frame(accuracy = accuracy, 
                                       f1 = f1, 
                                       recall = recall, 
                                       precision = precision),
              'table' = table))
}

# Function to plot confusion matrix
plot_cm <- function(cm_table, model_name) {
  ggplot(data = cm_table, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = "blue") +
    geom_text(aes(label = Freq), vjust = 1) +
    theme_minimal() +
    labs(title = paste("Confusion Matrix -", model_name), x = "Predicted", y = "Actual") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# True values
true_values <- prediction_df$exited_factor

# Calculate metrics for each model
dummy_results <- calculate_metrics(prediction_df$dummy_pred, true_values)
simple_lr_results <- calculate_metrics(prediction_df$simple_lr_pred, true_values)
ridge_lr_results <- calculate_metrics(prediction_df$ridge_lr_pred, true_values)
lasso_lr_results <- calculate_metrics(prediction_df$lasso_lr_pred, true_values)
en_lr_results <- calculate_metrics(prediction_df$elastic_net_lr_pred, true_values)
dt_lr_results <- calculate_metrics(prediction_df$decision_tree_pred, true_values)
rf_lr_results <- calculate_metrics(prediction_df$rf_pred, true_values)
xgb_lr_results <- calculate_metrics(prediction_df$xgb_pred, true_values)

# Combine results into a single dataframe
metrics_df <- rbind(
  cbind(model = "Dummy", dummy_results$performance),
  cbind(model = "Simple LR", simple_lr_results$performance),
  cbind(model = "Ridge LR", ridge_lr_results$performance),
  cbind(model = "Lasso LR", lasso_lr_results$performance),
  cbind(model = "Elastic Net LR", en_lr_results$performance),
  cbind(model = "Decision Tree", dt_lr_results$performance),
  cbind(model = "Random Forest", rf_lr_results$performance),
  cbind(model = "XGBoost", xgb_lr_results$performance)
)

row.names(metrics_df) <- NULL

metrics_df %>% 
  pivot_longer(., -model, names_to = 'Metric', values_to = 'Value') %>% 
  ggplot() +
    geom_point(aes(x = Value, y = Metric, color = model), alpha = 0.5)

dummy_plot <- plot_cm(data.frame(dummy_results$table), "Dummy Model")
simple_lr_plot <- plot_cm(data.frame(simple_lr_results$table), "Simple LR Model")
ridge_lr_plot <- plot_cm(data.frame(ridge_lr_results$table), "Ridge LR Model")
lasso_lr_plot <- plot_cm(data.frame(lasso_lr_results$table), "Lasso LR Model")
en_lr_plot <- plot_cm(data.frame(en_lr_results$table), "EN LR Model")
dt_plot <- plot_cm(data.frame(dt_lr_results$table), "Decision Tree Model")
rf_plot <- plot_cm(data.frame(rf_lr_results$table), "Random Forest Model")

# Combine all plots into a single plot with two columns
combined_plot <- cowplot::plot_grid(
  dummy_plot, simple_lr_plot,
  ridge_lr_plot, lasso_lr_plot,
  en_lr_plot, dt_plot, rf_plot,
  ncol = 3
)

ggsave("plots/model_cm.png", plot = combined_plot, width = 10, height = 10, units = "in", dpi = 300)


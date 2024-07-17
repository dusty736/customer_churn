################################################################################
# Step 1: EDA
################################################################################

################################################################################
# Setup
################################################################################

# Set working directory
setwd(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))

# Libraries
library(tidyverse)

# Load Data
banking_churn <- data.table::fread(file.path("data", "Churn_Modelling.csv")) %>% 
  janitor::clean_names()

################################################################################
# Quick Summary
################################################################################

numeric_columns <- c('credit_score', 'age', 'tenure', 'balance', 'num_of_products',
                     'has_cr_card', 'is_active_member', 'estimated_salary')

total_numeric_df <- banking_churn %>% 
  dplyr::select(all_of(numeric_columns)) %>% 
  mutate(churn_status = 'all') %>% 
  pivot_longer(., -churn_status, names_to = 'variable', values_to = 'value')

split_numeric_df <- banking_churn %>% 
  dplyr::select(all_of(c(numeric_columns, 'exited'))) %>% 
  mutate(churn_status = ifelse(exited == 1, 'churned', 'retained')) %>% 
  dplyr::select(-exited) %>% 
  pivot_longer(., -churn_status, names_to = 'variable', values_to = 'value')

numeric_summary <- rbind(total_numeric_df, split_numeric_df) %>% 
  group_by(churn_status, variable) %>% 
  summarize(n = n(),
            n_missing = sum(is.na(value)),
            min_value = min(value, na.rm=TRUE),
            q25_value = quantile(value, 0.25, na.rm=TRUE),
            q50_value = quantile(value, 0.50, na.rm=TRUE),
            mean_value = mean(value, na.rm=TRUE),
            median_value = median(value, na.rm=TRUE),
            q75_value = quantile(value, 0.75, na.rm=TRUE),
            max_value = max(value, na.rm=TRUE)) %>% 
  ungroup(.) %>% 
  mutate_if(is.numeric, round, 2) %>% 
  arrange(variable, churn_status)

################################################################################
# Pair Plot
################################################################################

# Create pairplot
pairplots <- banking_churn %>% 
  dplyr::select(all_of(c(numeric_columns, 'exited'))) %>% 
  GGally::ggpairs(aes(color = factor(exited), alpha = 0.5))

ggsave("plots/variable_pairplot.png", plot = pairplots, width = 10, height = 10, units = "in", dpi = 300)

################################################################################
# Variance Inflation Factor
################################################################################

# VIF quantifies how much the variance of a regression coefficient is inflated due to 
# multicollinearity with other predictors. A VIF value above 10 is often used as a 
# rule of thumb to indicate high multicollinearity, although some sources suggest a threshold of 5.

# Fit a linear model
model <- lm(exited ~ credit_score + age + tenure + balance + num_of_products + has_cr_card + is_active_member + estimated_salary, data = banking_churn)

# Calculate VIF
vif_values <- car::vif(model)

# Print VIF values
print(vif_values)

################################################################################
# Non-Numeric Variables
################################################################################

table(banking_churn$geography)
table(banking_churn$surname)

################################################################################
# Findings
################################################################################

# 1. There is no missing data.
# 2. The numeric variables vary in total scale, so we will min-max scale them.
# 3. Variables that seem helpful in predicting churn include: credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member
# 4. There is no issue of multicollinearity
# 5. surname will all be excluded to avoid discriminatory classifications

################################################################################
# Save Modeling Data
################################################################################

data.table::fwrite(banking_churn %>%
                     dplyr::select(-c(row_number, surname)), 
                   file.path("data", "modeling_data.csv"))


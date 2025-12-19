#Phase 3 (Model Training and fine tuning manually)

# Install and load necessary packages
if(!require(xgboost)) install.packages("xgboost")
if(!require(caTools)) install.packages("caTools")
if(!require(caret)) install.packages("caret")
if(!require(dplyr)) install.packages("dplyr") 
if(!require(Matrix)) install.packages("Matrix")

library(xgboost)
library(caTools)
library(caret)
library(dplyr)
library(Matrix)


OUT_DIR <- "Dhaka_Tribune"
cat("--- Loading prepared data from Phase 2 ---\n")


dtm_tfidf <- readRDS(file.path(OUT_DIR, "dtm_tfidf_bigram.rds"))
print(class(dtm_tfidf))
clean_articles <- readRDS(file.path(OUT_DIR, "clean_articles.rds"))

cat(" Data loaded successfully.\n\n")


#PREPARE LABELS AND DATA SPLITS
cat("--- Preparing labels for the model ---\n")
label_factor <- factor(clean_articles$Section)
label_levels <- levels(label_factor)
labels_num <- as.integer(label_factor) - 1

cat("Label mapping created:\n")
print(data.frame(Section = label_levels, NumericLabel = 0:(length(label_levels)-1)))



#Create a Stratified Train/Test Split
cat("\n--- Splitting data into 80% training and 20% testing ---\n")
set.seed(123)
train_idx <- createDataPartition(label_factor, p = 0.8, list = FALSE, times = 1)
test_idx  <- seq_len(nrow(clean_articles))[-train_idx]
cat("Data splitting complete.\n")



#CREATE FINAL DATASETS FOR XGBOOST
cat("--- Creating final data subsets and DMatrix objects ---\n")
sparse_mat <- as(dtm_tfidf, "sparseMatrix")

train_x <- sparse_mat[train_idx, ]
test_x  <- sparse_mat[test_idx, ]

train_y <- labels_num[train_idx]
test_y  <- labels_num[test_idx]

dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest  <- xgb.DMatrix(data = test_x, label = test_y)
cat(" Data is now fully prepared for XGBoost.\n\n")


#TRAIN THE XGBOOST MODEL WITH TUNED PARAMETERS

#Define Tuned Model Parameters
cat("--- Defining TUNED XGBoost parameters ---\n")
params_tuned <- list(
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = length(label_levels),
  eta = 0.05,
  max_depth = 8,
  subsample = 0.8,
  colsample_bytree = 0.7
)

#Find Best Rounds with CV
cat("--- Running 5-fold cross-validation ---\n")
set.seed(123)
xgb_cv <- xgb.cv(
  params = params_tuned,
  data = dtrain,
  nrounds = 500,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 1
)
best_nrounds <- xgb_cv$best_iteration
cat("\n Cross-validation complete. Best number of rounds:", best_nrounds, "\n")


#Train Final Model
cat("--- Training the final XGBoost model ---\n")
set.seed(123)
xgb_model <- xgboost(
  params = params_tuned,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 1
)
cat(" Model training complete!\n\n")


#EVALUATE FINAL MODEL
cat("--- Evaluating final model performance ---\n")
pred_probs <- predict(xgb_model, newdata = dtest)
pred_matrix <- matrix(pred_probs, ncol = length(label_levels), byrow = TRUE)
pred_labels_num <- max.col(pred_matrix) - 1

pred_labels <- factor(label_levels[pred_labels_num + 1], levels = label_levels)
true_labels <- factor(label_levels[test_y + 1], levels = label_levels)

cm <- confusionMatrix(data = pred_labels, reference = true_labels)
print(cm)
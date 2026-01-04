# 📰 Dhaka Tribune News Classification Engine

**Scrape, Prepare, and Classify Bangladeshi News Articles with Machine Learning**

---

## 🎯 Badges

![R Version](https://img.shields.io/badge/R-%3E%3D4.0-276DC3?style=flat-square&logo=r)
![ML Framework](https://img.shields.io/badge/XGBoost-Latest-336791?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Project Type](https://img.shields.io/badge/Project%20Type-NLP%2FText%20Classification-blue?style=flat-square)

---

## 📌 Project Description

**Dhaka Tribune News Classification Engine** is a complete end-to-end machine learning pipeline for **scraping, processing, and automatically classifying news articles** from [Dhaka Tribune](https://www.dhakatribune.com/), a major Bangladeshi news outlet. 

The project demonstrates a full workflow in R:
- 🕷️ **Web Scraping** via API and HTML parsing
- 🧹 **Data Cleaning & Tokenization** with advanced NLP techniques
- 📊 **Exploratory Data Analysis** with visualizations
- 🤖 **Multi-class Text Classification** using XGBoost and Tidymodels
- 📝 **Automatic Text Summarization** using LexRank algorithm

### 📚 My Learning Journey

This project started as a challenge to classify 1,600+ news articles and evolved into a powerful lesson about machine learning fundamentals:

**The Hard Lesson: Manual Tuning Limits** 📉  
To understand fundamentals, I built my first XGBoost model manually:
- **Manual XGBoost Approach:** 54% accuracy (0.5465) ❌
- **Root Cause:** 129,000+ features created from unigrams + bigrams—pure noise
- **The Realization:** More data doesn't mean better results. I was drowning in features.

**The Turning Point: Smart Feature Selection** 🚀  
I switched to R's **tidymodels framework** with intelligent feature selection:
- **Tidymodels Approach:** 94% accuracy (0.9373) ✅
- **The Magic:** A single `step_tokenfilter()` filtered out 99% of noise
- **Key Insight:** "Smart feature selection is everything in machine learning. It's not about having the most data, but the right data."

This **40-point accuracy jump** from 54% → 94% wasn't just about numbers—it taught me why ML frameworks exist over manual fine-tuning.

**Why This Project Matters:**
- Learn practical NLP and ML pipeline development beyond textbooks
- Understand the real-world struggle: cleaning messy web data
- See why frameworks matter: compare manual XGBoost (54%) vs tidymodels (94%)
- Implement scalable text classification in R
- Discover that data quality > data quantity

**Key Use Cases:**
- Automatically categorize news articles into sections (Bangladesh, World, Business, Sports)
- Generate automatic summaries of long articles
- Train custom classifiers for news categorization
- Learn why feature engineering and selection are crucial for ML success

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-----------|
| **Language** | R 4.0+ |
| **Web Scraping** | rvest, httr, jsonlite |
| **Data Processing** | dplyr, stringr, tidyr, readr |
| **NLP & Tokenization** | tidytext, tm, textstem, SnowballC, tokenizers |
| **ML Frameworks** | XGBoost, Caret, tidymodels, textrecipes |
| **Text Summarization** | lexRankr, textrank |
| **Visualization** | ggplot2, ggraph, wordcloud, igraph |
| **Data Structures** | Matrix, sparseMatrix |

---

## ✨ Features

- ✅ **Automated Web Scraping** - Reverse-engineered API calls to scrape 1,600+ articles from Dhaka Tribune
- ✅ **Multi-Section Coverage** - Collects articles from Bangladesh, World, Business, and Sports sections
- ✅ **Smart API Pagination** - Simulates browser clicks to load more content via hidden API endpoints
- ✅ **Advanced Text Preprocessing** - URL removal, stopword filtering, lemmatization, lowercasing
- ✅ **Feature Engineering** - Unigrams + Bigrams with TF-IDF scoring (129,000+ initial features)
- ✅ **Dual Model Implementations** - Compare manual XGBoost (54% ❌) vs Tidymodels framework (94% ✅)
- ✅ **Intelligent Feature Selection** - Filters 99% of noise to improve accuracy from 54% → 94%
- ✅ **Cross-Validation** - 5-fold CV for optimal hyperparameter selection
- ✅ **Comprehensive EDA** - Word frequency analysis, word clouds, bigram networks
- ✅ **Text Summarization** - LexRank-based automatic multi-sentence summaries
- ✅ **Production-Ready** - Modular functions for prediction on new articles
- ✅ **Educational Value** - Learn why frameworks beat manual tuning through real-world comparison

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **R** (version 4.0 or higher) - [Download here](https://www.r-project.org/)
- **RStudio** (recommended) - [Download here](https://www.rstudio.com/products/rstudio/download/)
- **Git** - [Download here](https://git-scm.com/)
- **Internet Connection** - Required for web scraping from Dhaka Tribune

### Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/TanmoyGG/Dhaka_Tribune-Scraping-and-Classification-XGBoost.git
cd Dhaka_Tribune-Scraping-and-Classification-XGBoost
```

#### Step 2: Open in RStudio
Open RStudio and set the working directory to the project folder:
```r
setwd("path/to/Dhaka_Tribune-Scraping-and-Classification-XGBoost")
```

#### Step 3: Install Required Packages
All required packages are automatically installed within each script if not already present. The scripts use:
```r
if(!require(package_name)) install.packages("package_name")
```

However, you can pre-install all dependencies at once:
```r
# Core packages
install.packages(c("rvest", "httr", "jsonlite", "dplyr", "readr", "stringr"))

# NLP packages
install.packages(c("tm", "tidytext", "SnowballC", "textstem", "tokenizers"))

# ML packages
install.packages(c("xgboost", "caTools", "caret", "Matrix"))

# Tidymodels ecosystem
install.packages(c("tidymodels", "textrecipes"))

# Text summarization
install.packages(c("lexRankr", "textrank"))

# Visualization
install.packages(c("ggplot2", "ggraph", "wordcloud", "igraph", "tidyr"))
```

#### Step 4: Verify Installation
Run this test script to verify all packages load correctly:
```r
library(rvest)
library(xgboost)
library(tidymodels)
cat("✓ All packages loaded successfully!\n")
```

---

## 🎓 Key Learnings: Why This Project Matters

This project teaches critical lessons that textbooks often skip:

### ❌ **Problem: Manual XGBoost (54% Accuracy)**
```
129,000 features (unigrams + bigrams)
↓
Tuned hyperparameters manually
↓
54% accuracy ❌
↓
Model drowns in noise
```

### ✅ **Solution: Tidymodels Framework (94% Accuracy)**
```
129,000 features
↓
step_tokenfilter() keeps only top 1,000
↓
99.2% of noise removed
↓
94% accuracy ✅
↓
Model learns signal, not noise
```

### 💡 **The Core Insight**
> "Smart feature selection is everything in machine learning. It's not about having the most data, but the RIGHT data."

**This 40-point accuracy jump (54% → 94%) demonstrates:**
- Why ML frameworks exist (they handle feature engineering intelligently)
- Why raw data quality > data quantity
- The power of reproducible, automated pipelines
- That hyperparameter tuning alone can't fix bad features

---

## 📖 Usage

### Project Workflow

The project is designed to run in **4 sequential phases**. Each phase builds on the outputs of the previous one.

#### **Phase 1: Data Scraping** (`1_Data_Scraping.R`)

Scrapes news articles from Dhaka Tribune across 4 sections using their API.

```r
# Run the entire script:
source("1_Data_Scraping.R")

# What happens:
# - Scrapes Bangladesh, World, Business, and Sports sections
# - Uses Dhaka Tribune's internal API for pagination
# - Extracts article URLs, titles, and content
# - Cleans HTML and normalizes text
# - Outputs: Dhaka_Tribune/dhakatribune_articles.csv (~3000+ articles)
```

**Configuration in Script:**
```r
MAX_CLICKS <- 25          # Number of API calls per section
ARTICLES_PER_CLICK <- 16  # Articles per API response
```

**Output Structure:**
```
Dhaka_Tribune/
└── dhakatribune_articles.csv
    ├── URL (article link)
    ├── Title (article headline)
    ├── Text (article content)
    └── Section (category: bangladesh, world, business, sport)
```

---

#### **Phase 2: Data Preparation & EDA** (`2_Data_Preparation.R`)

Cleans data, performs tokenization, and generates exploratory visualizations.

```r
# Run the entire script:
source("2_Data_Preparation.R")

# What happens:
# - Removes duplicates and invalid articles
# - Cleans URLs, mentions, special characters
# - Lemmatizes words for standardization
# - Creates unigrams (single words) and bigrams (word pairs)
# - Computes TF-IDF scoring for feature importance
# - Generates EDA visualizations
# - Outputs: Clean data + DTM (Document-Term Matrix)
```

**Cleaning Steps:**
```r
# URLs and social media mentions removed
# Text converted to lowercase
# Lemmatization: "running" → "run", "better" → "good"
# TF-IDF normalization for ML-friendly features
```

**Visualizations Generated:**
```
Dhaka_Tribune/
├── Bar Plot(Top 20).png              # Top 20 words bar chart
├── wordcloud.png                      # Overall word cloud
├── wordcloud_bangladesh.png           # Section-specific clouds
├── wordcloud_world.png
├── wordcloud_business.png
├── wordcloud_sport.png
├── Top word per section(Top 15).png   # Top 15 words per category
├── bigram_network.png                 # Word co-occurrence network
├── clean_articles.rds                 # Cleaned article data
└── dtm_tfidf_bigram.rds              # Feature matrix for ML
```

---

#### **Phase 3A: Model Training - Manual XGBoost** (`3_Model_Training.R`)

⚠️ **The "Hard Way" - Manual hyperparameter tuning with 129,000+ features**

```r
# Run the entire script:
source("3_Model_Training.R")

# What happens:
# - Loads cleaned data and DTM from Phase 2 (129,000+ features)
# - Splits data: 80% train, 20% test
# - Performs 5-fold cross-validation
# - Trains XGBoost with manually tuned parameters
# - Evaluates with confusion matrix
```

**Model Parameters:**
```r
params_tuned <- list(
  objective = "multi:softprob",    # Multi-class classification
  eval_metric = "mlogloss",        # Cross-entropy loss
  num_class = 4,                   # 4 news categories
  eta = 0.05,                      # Learning rate (conservative)
  max_depth = 8,                   # Tree depth
  subsample = 0.8,                 # Row sampling
  colsample_bytree = 0.7           # Feature sampling
)
```

**❌ Results: 54% Accuracy (0.5465)**

```
Manual XGBoost Performance
- Feature Count: 129,000+ (unfiltered unigrams + bigrams)
- Accuracy: 54% ❌
- Problem: Drowning in noisy, irrelevant features
- Key Lesson: More features ≠ Better predictions
```

**Why It Failed:**
- No intelligent feature filtering
- All 129,000 features treated equally (signal + noise)
- Model overfits on noise, poor generalization
- Manual parameter tuning couldn't overcome feature noise

**Confusion Matrix Screenshot:**

![Result of Manual XGBoost](https://i.postimg.cc/q7WtctJx/Screenshot-2025-09-11-061633.png)

---

#### **Phase 3B: Model Training - Tidymodels Framework** (`4_Model_Training_with_Tidymodels_and_TextRank_Summmarizer.R`)

🚀 **The "Smart Way" - Intelligent feature selection with tidymodels**

```r
# Run the entire script:
source("4_Model_Training_with_Tidymodels_and_TextRank_Summmarizer.R")

# What happens:
# - Uses tidymodels workflow for reproducibility
# - Intelligent recipe with step_tokenfilter() (keeps only top 1,000 features)
# - Filters out 99% of noise automatically
# - Integrates recipe (preprocessing) + model + engine
# - Trains XGBoost on CLEAN features only
# - Evaluates performance
# - Demonstrates text summarization with LexRank
```

**Key Difference - The Recipe (Feature Selection):**
```r
text_recipe <- recipe(Section ~ Text, data = train_data) %>%
  step_tokenize(Text) %>%
  step_stopwords(Text) %>%
  step_tokenfilter(Text, max_tokens = 1000) %>%  # ← THE MAGIC: Only top 1,000 features!
  step_tfidf(Text)
```

**✅ Results: 94% Accuracy (0.9373) - 40 Point Improvement!**

```
Tidymodels Performance
- Feature Count: 1,000 (intelligently selected)
- Accuracy: 94% ✅ (40-point improvement from 54%!)
- Filtering: Removed 99.2% of noise
- Key Lesson: Smart selection beats manual tuning + raw data
- Execution Time: Faster than manual approach
```

**Why It Worked:**
- `step_tokenfilter()` kept only most informative features
- Model trained on signal, not noise
- Better generalization to new articles
- Reproducible & maintainable pipeline

**Confusion Matrix Screenshot:**
![Tidymodel_XGBoost](https://i.postimg.cc/8PkC6dd8/Screenshot-2025-09-11-062122.png)

**Making Predictions on New Text:**
```r
predict_category <- function(texts, fitted_workflow) {
  new_data <- tibble(Text = texts)
  predictions <- predict(fitted_workflow, new_data = new_data)
  
  results_df <- tibble(
    'Sample Text' = texts,
    'Predicted Category' = predictions$.pred_class
  )
  return(results_df)
}

# Use on new articles
sample_articles <- c(
  "Bangladesh government announces new digital policy...",
  "World leaders meet at international climate summit..."
)

predictions <- predict_category(sample_articles, xgb_fit)
print(predictions)
```

---

#### **Text Summarization with LexRank**

Automatic multi-sentence summary generation:

```r
summarize_article_lexrank <- function(article_text, num_sentences = 3) {
  summary_sentences <- lexRank(
    text = article_text, 
    docId = "doc1", 
    n = num_sentences, 
    returnTies = FALSE 
  )
  
  ordered_summary <- summary_sentences %>% arrange(sentenceId)
  summary_text <- paste(ordered_summary$sentence, collapse = " ")
  return(summary_text)
}

# Example usage
full_article <- articles_df$Text[1]
summary <- summarize_article_lexrank(full_article, num_sentences = 3)
```

---

### Complete Execution Flow

**Run all phases sequentially:**

```r
# Phase 1: Scrape data (10-30 minutes)
source("1_Data_Scraping.R")

# Phase 2: Prepare & visualize (2-5 minutes)
source("2_Data_Preparation.R")

# Phase 3A: Train XGBoost (5-10 minutes)
source("3_Model_Training.R")

# Phase 3B: Train with Tidymodels + Summarize (5-10 minutes)
source("4_Model_Training_with_Tidymodels_and_TextRank_Summmarizer.R")
```

---

## 📂 Project Structure

```
Dhaka_Tribune-Scraping-and-Classification-XGBoost/
│
├── 1_Data_Scraping.R                          # Phase 1: Web scraping
│   ├── API Configuration for 4 news sections
│   ├── Pagination simulation (API calls)
│   └── Output: dhakatribune_articles.csv
│
├── 2_Data_Preparation.R                       # Phase 2: Cleaning & EDA
│   ├── Data inspection & quality checks
│   ├── Text cleaning & tokenization
│   ├── TF-IDF computation
│   ├── EDA visualizations
│   └── Outputs: clean_articles.rds, dtm_tfidf_bigram.rds
│
├── 3_Model_Training.R                         # Phase 3A: XGBoost training
│   ├── Train/test split (80/20)
│   ├── 5-fold cross-validation
│   ├── Hyperparameter tuning
│   └── Output: Confusion matrix & metrics
│
├── 4_Model_Training_with_Tidymodels...R      # Phase 3B: Tidymodels + Summarization
│   ├── Recipe-based feature engineering
│   ├── Workflow setup
│   ├── Model training & prediction
│   └── LexRank text summarization
│
├── Dhaka_Tribune/                             # Output directory
│   ├── dhakatribune_articles.csv              # Raw scraped data (3000+ articles)
│   ├── clean_articles.rds                     # Cleaned data (R object)
│   ├── dtm_tfidf_bigram.rds                   # Feature matrix (sparse)
│   ├── Bar Plot(Top 20).png                   # Word frequency chart
│   ├── wordcloud.png                          # Overall word cloud
│   ├── wordcloud_[section].png                # Section-specific clouds
│   ├── Top word per section(Top 15).png       # Category-wise top words
│   └── bigram_network.png                     # Word co-occurrence network
│
└── README.md                                  # This file
```

---
## 📈 Project Performance Metrics

### Comparison: Manual XGBoost vs Tidymodels

| Metric | Manual XGBoost | Tidymodels | Winner |
|--------|---|---|---|
| **Accuracy** | 54% (0.5465) ❌ | **94% (0.9373)** ✅ | **Tidymodels +40%** |
| **Features Used** | 129,000+ (all) | 1,000 (filtered) | **Tidymodels (99.2% reduction)** |
| **Feature Selection** | None | `step_tokenfilter()` | **Tidymodels** |
| **Reproducibility** | Low | **High** | **Tidymodels** |
| **Maintenance** | Manual tuning | **Automated pipeline** | **Tidymodels** |
| **Training Speed** | 10-15 min | **5-10 min** | **Tidymodels** |

---

### Overall Project Metrics

| Metric | Value |
|--------|-------|
| **Total Articles Scraped** | ~1,600+ |
| **News Sections** | 4 (Bangladesh, World, Business, Sports) |
| **Initial Features (raw)** | 129,000+ (unigrams + bigrams) |
| **Final Features (optimized)** | 1,000 (tidymodels filtered) |
| **Feature Reduction** | 99.2% |
| **Train/Test Split** | 80/20 (stratified) |
| **Best Model Accuracy** | 94% (Tidymodels) ✅ |
| **Manual Model Accuracy** | 54% (XGBoost manual) ❌ |
| **Accuracy Improvement** | **+40 percentage points** |
| **Cross-Validation Folds** | 5 |
| **Best ML Framework** | Tidymodels (XGBoost engine) |
| **Average Scrape Time** | 30-45 minutes |
| **Preprocessing Time** | 3-5 minutes |
| **Training Time (Manual)** | 10-15 minutes |
| **Training Time (Tidymodels)** | 5-10 minutes (40% faster!) |
| **Total Pipeline Time** | 45-65 minutes |

---

## ⚠️ Challenges & Limitations

### Challenges Faced During Development

1. **Website Structure Changes**
   - Dhaka Tribune periodically updates their HTML structure
   - **Solution:** Use robust CSS selectors and comprehensive error handling with try/catch blocks
   - **Impact:** Scraping may fail if website structure changes significantly

2. **Rate Limiting & IP Blocking**
   - Servers may block aggressive scraping attempts
   - **Implemented:** Random delays (1-3.5 seconds) between requests, Custom User-Agent headers
   - **Note:** Always respect `robots.txt` and website terms of service

3. **Data Imbalance**
   - Some news sections have significantly fewer articles than others
   - **Solution:** Use stratified train/test splits to maintain class distribution
   - **Status:** Already implemented in the code

4. **Text Quality Issues**
   - Duplicate articles, malformed HTML, encoding errors, missing content
   - **Solution:** Comprehensive data cleaning pipeline with duplicate detection

5. **Feature Explosion**
   - Unigrams + Bigrams combined create 10,000+ features
   - **Impact:** Memory usage and training time increase significantly
   - **Solution:** TF-IDF scoring + sparse matrix representation (already implemented)

6. **Language-Specific NLP Challenges**
   - English stop-word lists are not optimal for Bangladeshi/Bengali context
   - **Current approach:** Lemmatization + domain-specific filtering
   - **Future improvement:** Bengali-specific tokenizers and stop-word lists

### Project Limitations

1. **Static Classification Schema**
   - Fixed to 4 news categories (Bangladesh, World, Business, Sport)
   - **Extensibility:** Can be retrained with custom categories, but requires new labeled data

2. **Memory Constraints**
   - DTM (3000 × 10,000+) can be memory-intensive on older machines
   - **Solution:** Already using sparse matrix representation

3. **Website Dependency**
   - Scraping requires live website access and operational API endpoints
   - **Risk:** API endpoints may change or become unavailable without notice

4. **Text Summarization Limitations**
   - LexRank is extractive (selects existing sentences) rather than abstractive
   - **Constraint:** Cannot generate new sentences or paraphrase content
   - **Impact:** Summaries may not be as natural as human-written ones

5. **Cross-Validation Time**
   - 5-fold CV with 500 rounds can take 30+ minutes
   - **Trade-off:** Balancing accuracy improvement vs. computational time

6. **Potential Class Imbalance**
   - If one section has disproportionately more articles, model may be biased
   - **Mitigation:** Monitor class distribution; implement XGBoost class weights if needed

7. **Scalability**
   - Current pipeline handles ~3000 articles efficiently
   - **For larger datasets:** Consider distributed computing (Spark, H2O)

8. **No Real-Time Updates**
   - Scraping is a one-time batch process, not continuous
   - **For production:** Would need scheduling (cron jobs, Docker containers, cloud functions)

### Known Issues & Workarounds

| Issue | Workaround |
|-------|-----------|
| Script timeout during scraping | Increase API call timeouts in httr::timeout() |
| Out of memory errors | Reduce MAX_CLICKS or pre-filter data |
| API endpoint unavailable | Check website status; retry scraping at different time |
| Low model accuracy | Increase training rounds or adjust hyperparameters |
| Duplicate text not removed | Run `distinct()` function manually in Phase 2 |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Issues
1. Check [Issues](https://github.com/yourusername/Dhaka_Tribune-Scraping-and-Classification-XGBoost/issues) for duplicates
2. Provide:
   - Clear description of the problem
   - Steps to reproduce
   - R version (`R --version`)
   - Package versions (`packageVersion("package_name")`)
   - Error messages and stack traces

### Submitting Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/TanmoyGG/Dhaka_Tribune-Scraping-and-Classification-XGBoost.git
   cd Dhaka_Tribune-Scraping-and-Classification-XGBoost
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow existing code style and conventions
   - Add comments to explain complex logic
   - Test changes thoroughly

3. **Commit & Push**
   ```bash
   git add .
   git commit -m "Add descriptive message (e.g., 'Add Bengali stop words support')"
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Describe changes and improvements
   - Reference related issues
   - Include before/after metrics if applicable

### Contribution Ideas

- 🌐 Add support for other Bangladeshi news sources (Daily Star, Prothom Alo, etc.)
- 🇧🇩 Implement Bengali-specific NLP (stemmer, stop words, tokenizer)
- 📊 Add advanced visualizations (interactive plots, Shiny dashboard)
- ⚡ Optimize code for better performance and memory efficiency
- 📝 Improve documentation with additional examples
- 🧪 Add unit tests and integration tests
- 🔄 Implement automatic data refresh pipeline with scheduling
- 🌳 Experiment with alternative ML algorithms (Random Forest, SVM, Neural Networks)
- 🇧🇩 Add support for Bengali language text processing
- 📈 Implement model versioning and tracking

---

## 📜 License

This project is licensed under the **MIT License** - see the LICENSE file for details.



## 📞 Support & Contact

- 📧 Email: tanmoydas180719@gmail.com
- 🌟 Star this project if you find it useful!

---

## 🎓 Learning Resources

**Resources used in this project:**

- [Web Scraping with rvest](https://rvest.tidyverse.org/)
- [tidytext: Text Mining in R](https://www.tidytextmining.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [tidymodels Framework](https://www.tidymodels.org/)
- [Natural Language Processing with R](https://www.routledge.com/Text-Mining-with-R/Silge-Robinson/p/book/9781491981641)
- [LexRank Algorithm](https://en.wikipedia.org/wiki/Automatic_summarization)
- [Sparse Matrices in R](https://cran.r-project.org/web/packages/Matrix/vignettes/Intro2Matrix.pdf)

---



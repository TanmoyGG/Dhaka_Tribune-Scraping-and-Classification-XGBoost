#Phase 3 (Model Training with Tidymodels framework)

# Install and load necessary packages
if(!require(tidymodels)) install.packages("tidymodels")
if(!require(textrecipes)) install.packages("textrecipes")
if(!require(xgboost)) install.packages("xgboost")
if(!require(textrank)) install.packages("textrank")
if(!require(dplyr)) install.packages("dplyr")
if(!require(stringr)) install.packages("stringr")
if(!require(tokenizers)) install.packages("tokenizers")
if(!require(readr)) install.packages("readr")
if(!require(lexRankr)) install.packages("lexRankr")

library(tidymodels)
library(textrecipes)
library(xgboost)
library(caret)
library(textrank)
library(dplyr)
library(stringr)
library(tokenizers)
library(readr)
library(lexRankr)


OUT_DIR <- "Dhaka_Tribune"

#LOAD & PREPARE DATA --
cat("--- Loading prepared data ---\n")
clean_articles <- readRDS(file.path(OUT_DIR, "clean_articles.rds"))

#Convert the Section to a factor, which tidymodels requires for classification
clean_articles <- clean_articles %>%
  mutate(Section = factor(Section))


#DATA SPLITTING
cat("--- Splitting data into training and testing sets ---\n")
set.seed(123)
article_split <- initial_split(clean_articles, prop = 0.8, strata = Section)

train_data <- training(article_split)
test_data  <- testing(article_split)

cat("Training articles:", nrow(train_data), "\n")
cat("Testing articles:", nrow(test_data), "\n")


# THE RECIPE: FEATURE ENGINEERING
cat("--- Defining the feature engineering recipe ---\n")

text_recipe <- recipe(Section ~ Text, data = train_data) %>%
  
  step_tokenize(Text) %>%
  
  step_stopwords(Text) %>%

  step_tokenfilter(Text, max_tokens = 1000) %>%

  step_tfidf(Text)


#MODEL SPECIFICATION --
cat("--- Defining the XGBoost model specification ---\n")
xgb_spec <- boost_tree(
  trees = 100,
  tree_depth = 8,
  learn_rate = 0.05
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")


# THE WORKFLOW: BUNDLING RECIPE + MODEL --
cat("--- Creating the workflow object ---\n")

xgb_workflow <- workflow() %>%
  add_recipe(text_recipe) %>%
  add_model(xgb_spec)


#TRAIN THE MODEL --
cat("--- Training the model (fitting the workflow) ---\n")

xgb_fit <- fit(xgb_workflow, data = train_data)
cat("Model training complete!\n\n")


#EVALUATE THE MODEL
cat("--- Evaluating the model on the test set ---\n")

test_predictions <- predict(xgb_fit, new_data = test_data)

results <- bind_cols(
  test_data %>% select(Section),
  test_predictions
) %>%
  rename(truth = Section, estimate = .pred_class)

cm <- conf_mat(results, truth = truth, estimate = estimate)

print(cm)
cat("\n")
print(summary(cm))



#EVALUATE THE MODEL
cat("--- Tidymodels Confusion Matrix ---\n")
print(cm)

cat("\n\n--- Detailed Confusion Matrix Report (from caret) ---\n")

detailed_cm <- caret::confusionMatrix(
  data = results$estimate,
  reference = results$truth
)

print(detailed_cm)



#PREDICT ON NEW, UNSEEN TEXT
predict_category <- function(texts, fitted_workflow) {

  new_data <- tibble(Text = texts)

  predictions <- predict(fitted_workflow, new_data = new_data)
  
  results_df <- tibble(
    'Sample Text' = texts,
    'Predicted Category' = predictions$.pred_class
  )
  
  return(results_df)
}


#EXAMPLE USAGE
sample_articles <- c(
  "A New Zealand rugby player who had wanted his brain to be studied after suffering from the effects of multiple concussions has died aged 39, police said Wednesday.Shane Christie, a former Maori All Blacks player, had campaigned for greater awareness in rugby of the impact of repeated blows to the head. After retiring from the game in 2017 Christie reportedly suffered from headaches, memory lapses, speech problems, depression and mood swings consistent with chronic traumatic encephalopathy (CTE). Local media said the death of Christie, who also played for Otago Highlanders, Canterbury Crusaders and the All Blacks sevens team, may have been suicide. The death will be referred to the coroner and we have no further information or comment we can provide, said police. New Zealand Rugby said Christie was deeply passionate about the sport and would be remembered always. Christie reportedly wanted to donate his brain to the New Zealand sports human brain bank for its studies into CTE, a degenerative disease caused by repetitive head trauma that cannot be detected in living people. Hundreds of American football (NFL) players have been affected by the condition, which is linked to an array of behavioural symptoms including depression. CTE has been cited in a number of violent deaths involving former NFL players. A 2023 study by the Boston University CTE Center said that of 376 brains of former NFL players, 345 of them were found to have CTE.",
  "In the past three days, Israel has launched strikes in Palestine, Lebanon, Syria, Tunisia, Qatar, and Yemen – marking the sixth country targeted in just 72 hours and the seventh since the start of the year, reports Aljazeera. On Tuesday, Israel carried out a targeted air strike in Doha, Qatar, hitting a Hamas leadership compound during a meeting on a US-proposed Gaza ceasefire. Six people were killed, including the son of senior Hamas leader Khalil al-Hayya, his office director, three bodyguards, and a Qatari security officer. Hamas’s top leaders reportedly survived. The strike, nearly 2,000km from Israel, was the first Israeli attack inside Qatar. Explosions shook the capital’s West Bay Lagoon district, home to embassies, schools, and residential compounds. Israel later confirmed the attack. Qatar has long hosted Hamas leaders at the request of the United States, which operates its regional CENTCOM command centre about 35km away. Israeli forces continue heavy bombardments across Gaza, killing at least 150 people and wounding more than 540 since Monday. On Monday, 67 people were killed, including 14 seeking aid, while six others – two of them children – died of famine-related causes. Another 83 were killed and 223 wounded on Tuesday. Israel has intensified its assault on Gaza City, targeting high-rise buildings, destroying infrastructure, and displacing residents. Since October 2023, at least 64,656 people have been killed in Gaza, including 404 who died from starvation. Thousands remain missing under the rubble and are presumed dead.",
  "The White Paper on Bangladesh's economy has revealed that crores of takas were embezzled from the stock market through fraud, manipulation and deceit, particularly in placement shares and IPO processes. The committee, led by economist Dr Debapriya Bhattacharya, presented the findings at a press conference in the NEC conference room in the capital on Sunday. Excessive government tutelage held back market development and constrained responsible institutions from carrying out their mandates. This, combined with strong vested interest, resulted in an entrenched status quo of gambling and swindling, said the report. Laws, rules, and regulations were deliberately deficient in their implementation. Weak and substandard companies came into the market through IPOs. A major manipulation network involving influential entrepreneurs, issue managers, auditors, and a certain class of investors emerged. In many cases, officials of the regulatory body themselves played a role as accomplices by exploiting legal loopholes or providing concessions. Equity market growth is dragged by poor market infrastructure and unwieldy processing cycle for initial public offerings (IPOs). Current market systems are not supportive of a well-functioning market. IPO valuations give the sponsors an upper hand over the general investors in the secondary market. Settlement delays raise the investors’ interest rate and price fluctuation risks. Liquidity is affected by the lengthy IPO cycle. Absence of central counterparty clearing, interoperable information technology infrastructure and adequate trading platforms constrain brokers and clearing houses from transparent market making and trading. Stunted investor confidence: Public perception of the stock market is impaired by the memories of manipulators facing no legal action based on the reports produced by the investigating committees. The Centre for Policy Dialogue (CPD): A study of 71 businessmen in 2023 found 50% of businessmen believe the prevalence of suspicious trading in the secondary market, 53.1% thought BSEC’s regulatory enforcement is weak, 50% found financial reporting anomalous, and 56.3% believed poor companies enter the capital market through initial public offerings (IPOs). The same issues topped the list in 2022. Market rigging is endemic: Several powerful investors and institutions artificially inflate the share prices through a series of trades, mostly among themselves, violating securities laws. They execute circular trades in targeted company shares, where some investors sell shares and others, related to them, buy shares in a series of trades to create the appearance of active trading. The book-building process is manipulated to the extent that it no longer effectively determines the true valuation of a company's shares. Anomalies in IPO valuations (mostly underpricing) give the sponsors an upper hand over the general investors in the secondary market. Some big-ticket mutual funds were taken over by vested interests. Specifically, allegations of embezzlement of unit holders' funds were made against the top two institutions in the closed- end mutual fund sector. BSEC looked the other way. The Khairul Commission extended the duration of all closed-end mutual funds by an additional ten years. Investor confidence plummeted. The increase in the index prompted regulations to raise margin loan ratios, fueling the stock market surge. The BSEC often maintained the index, disregarding rising stock prices, with regulatory action only taking place when prices began to fall.",
  "The Election Commission (EC) has informed the Ministry of Home Affairs that it will not take responsibility for installing CCTV cameras at polling centres during the upcoming 13th parliamentary election planned to be held in early February 2026. In a recent letter to the ministry, the EC conveyed that it has nothing to do with the installation of CCTV cameras in polling stations and body-worn cameras for police personnel. On August 6 last, the Ministry of Home Affairs held a meeting chaired by Home Affairs Adviser Lt Gen (Retd) Md Jahangir Alam Chowdhury, where a number of decisions were taken to ensure the proper conduct of the election. The decisions include monitoring the polling stations through the installation of CCTV cameras in the polling stations and equipping police members with body-worn cameras. The responsibility of implementing the decision was given to either the EC or the police. But the Election Commission later clarified its position to the ministry in this regard. In the letter sent to the ministry on Monday, the EC said: “With reference to agenda item 9 of the minutes of the August 6 meeting on logistical issues and action plan of law enforcement agencies, the commission has decided that there is no responsibility on its part regarding CCTV cameras and body-worn cameras.” Election Commissioner Brig Gen (Retd) Abul Fazal Md Sanaullah also told reporters earlier that the commission was not considering CCTV cameras in the election process. On August 7 last, he said the EC did not find the use of CCTV cameras during elections to be rational."
  )

final_predictions <- predict_category(texts = sample_articles, fitted_workflow = xgb_fit)

print(final_predictions)









#TEXT_SUMMARIZER (with lexRank)

#LOAD ORIGINAL RAW DATA --
OUT_DIR <- "Dhaka_Tribune"
articles_df <- read_csv(file.path(OUT_DIR, "dhakatribune_articles.csv"), show_col_types = FALSE)
articles_df <- articles_df %>% distinct(Text, .keep_all = TRUE)


#LEXRANK SUMMARIZER FUNCTION
summarize_article_lexrank <- function(article_text, num_sentences = 3) {
  
  summary_sentences <- lexRank(
    text = article_text, 
    docId = "doc1", 
    n = num_sentences, 
    returnTies = FALSE 
  )
  
  ordered_summary <- summary_sentences %>%
    arrange(sentenceId)
  
  summary_text <- paste(ordered_summary$sentence, collapse = " ")
  
  return(summary_text)
}

#EXAMPLE USAGE (from our dataset)
sample_article <- articles_df$Text[250]

cat("--- ORIGINAL ARTICLE (first 800 characters) ---\n")
print(str_wrap(substr(sample_article, 1, 800)))

cat("\n\n--- 3-SENTENCE LEXTRANK SUMMARY ---\n")
summary_result <- summarize_article_lexrank(sample_article)
print(str_wrap(summary_result))



#EXAMPLE USAGE (with the long article)

long_article_text <- "OpenAI on Tuesday unveiled its latest flagship artificial intelligence model, dubbed GPT-4o, a significant update to its popular chatbot technology that is now faster and offers enhanced capabilities in text, vision, and audio. The 'o' in GPT-4o stands for 'omni', highlighting its new ability to handle multiple types of media inputs and outputs natively. During a live-streamed demonstration, OpenAI executives showcased the model's new, more natural and human-like conversational abilities. In one demo, the AI voice assistant was able to detect the user's emotion by analyzing their tone of voice and could even be interrupted and respond in real-time, much like a human conversation. Another demonstration showed the model's vision capabilities, where it successfully interpreted a live video feed from a phone's camera to help a user solve a math equation written on a piece of paper. It also demonstrated its ability to act as a real-time translator between two people speaking different languages. Company officials emphasized that a key goal of GPT-4o was to make advanced AI more accessible. The new model will be available for free to all users, though paid users will have higher usage limits. This marks a significant strategy shift, as previous advanced models were largely kept behind a paywall. The move is seen by industry analysts as a direct response to increasing competition from rivals like Google and Anthropic, who have also recently released powerful new models. Technologically, the new model is a single, unified system that was trained across text, vision, and audio data simultaneously. This is a departure from previous versions where voice features were handled by separate, slower models. By integrating these capabilities into one system, OpenAI claims that GPT-4o can respond to audio inputs in as little as 232 milliseconds, a speed comparable to human reaction time in a conversation. The rollout of these new features will be gradual. The enhanced text and vision capabilities are beginning to be deployed to ChatGPT users this week, while the more advanced voice and video features will be available to a select group of alpha testers in the coming weeks before a wider release later in the year. The company stated that safety was a top priority, with new mitigation techniques built-in to guard against potential misuse of the more powerful audio and vision features."

cat("--- ORIGINAL ARTICLE (full text) ---\n")
print(str_wrap(long_article_text))

cat("\n\n--- 3-SENTENCE LEXRANK SUMMARY ---\n")
summary_result <- summarize_article_lexrank(long_article_text)
print(str_wrap(summary_result))

















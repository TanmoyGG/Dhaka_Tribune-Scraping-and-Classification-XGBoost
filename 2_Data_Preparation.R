#Phase 2 (Data Preparation & EDA)

# Install and load necessary packages
if(!require(dplyr)) install.packages("dplyr")
if(!require(readr)) install.packages("readr")
if(!require(stringr)) install.packages("stringr")
if(!require(tm)) install.packages("tm")      
if(!require(tidytext)) install.packages("tidytext")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(ggraph)) install.packages("ggraph")
if(!require(wordcloud)) install.packages("wordcloud")
if(!require(SnowballC)) install.packages("SnowballC")  
if(!require(textstem)) install.packages("textstem")   

library(dplyr)
library(readr)
library(stringr)
library(tm)
library(tidytext)
library(ggplot2)
library(wordcloud)
library(SnowballC)
library(textstem)
library(igraph)
library(ggraph)
library(tidyr)

OUT_DIR <- "Dhaka_Tribune"
if(!dir.exists(OUT_DIR)) dir.create(OUT_DIR)

#Load dataset from Phase 1
file_path <- "Dhaka_Tribune/dhakatribune_articles.csv"
articles_df <- read_csv(file_path)



#Quick Dataset Inspection Function
inspect_dataset <- function(df) {
  str(df)
  summary(df$Section)
  
  cat("\n--- Dataset Structure ---\n")
  glimpse(df)
  
  cat("--- Dataset Dimensions ---\n")
  print(dim(df))
  
  cat("\n Dataset Inspection:\n")
  cat("Rows:", nrow(df), " | Columns:", ncol(df), "\n\n")
  
  cat(" Missing values per column:\n")
  print(colSums(is.na(df)))
  cat("\n")
  
  dup_count <- sum(duplicated(df$URL))
  cat(" Duplicate URLs:", dup_count, "\n\n")
  
  duplicate_rows <- df %>%
    group_by(Text) %>%
    filter(n() > 1) %>%
    summarise(Count = n())
  
  cat("\n--- Count of Duplicate Article Texts ---\n")
  print(paste("Found", nrow(duplicate_rows), "unique articles that appear more than once."))
  
  cat(" Distribution of Sections:\n")
  print(table(df$Section))
  cat("\n")
  
  df$text_length <- nchar(df$Text)
  cat("📝 Average Article Length (chars):", mean(df$text_length), "\n")
  cat("📏 Min:", min(df$text_length), " | Max:", max(df$text_length), "\n")
}

inspect_dataset(articles_df)



#DATA CLEANING
clean_articles <- articles_df %>%
  distinct(Text, .keep_all = TRUE) %>%
  
  mutate(
    Text = str_replace_all(Text, "https?://\\S+|www\\.[^\\s]+", " "), 
    Text = str_replace_all(Text, "@\\w+|#\\w+", " "),                 
    Text = str_replace_all(Text, "[^\\p{L}\\p{N}\\s']", " "),         
    Text = tolower(Text),                                             
    Text = str_squish(Text),                                          
    
    Title   = tolower(Title),
    Title   = str_squish(Title),
    Section = tolower(Section)
  )

cat("\n After comprehensive cleaning:\n")
cat("Rows before:", nrow(articles_df), "\n")
cat("Rows after :", nrow(clean_articles), "\n")

cat("\n--- Preview of Comprehensively Cleaned Data ---\n")
print(head(select(clean_articles, URL, Text), 3))

cat("\n\n--- INSPECTION AFTER CLEANING ---\n")
inspect_dataset(clean_articles)



#TOKENIZATION & FEATURE ENGINEERING (UNIGRAMS + BIGRAMS) 
cat("\n--- Starting text processing with unigrams and bigrams ---\n")

data("stop_words")

#Create Unigrams (single words)
unigrams <- clean_articles %>%
  mutate(ArticleID = row_number()) %>%
  unnest_tokens(output = word, input = Text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(str_detect(word, "[a-z]")) %>%
  filter(nchar(word) >= 3) %>%
  mutate(lemma = textstem::lemmatize_words(word)) %>%
  select(ArticleID, Section, term = lemma)

cat("Unigram processing complete.\n")


#Create Bigrams (word pairs)
bigrams <- clean_articles %>%
  mutate(ArticleID = row_number()) %>%
  unnest_tokens(output = bigram, input = Text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word) %>%
  unite(term, word1, word2, sep = " ") %>%
  select(ArticleID, Section, term)

cat("Bigram processing complete.\n")

combined_terms <- bind_rows(unigrams, bigrams)

cat("Combined unigrams and bigrams into a single feature set.\n")


#create the DTM with the features
cat(" Creating DTM with unigrams and bigrams ")

term_counts <- combined_terms %>%
  count(ArticleID, term, sort = TRUE)

# Calculate TF-IDF
terms_tfidf <- term_counts %>%
  bind_tf_idf(term = term, document = ArticleID, n = n)

# Cast into a Document-Term Matrix
dtm_tfidf_bigram <- terms_tfidf %>%
  cast_dtm(document = ArticleID, term = term, value = tf_idf)

cat("DTM created successfully.\n")
print(dtm_tfidf_bigram)


cat(" Saving the DTM for the modeling script ")

saveRDS(clean_articles, file = file.path(OUT_DIR, "clean_articles.rds"))

saveRDS(dtm_tfidf_bigram, file = file.path(OUT_DIR, "dtm_tfidf_bigram.rds"))

cat(" Successfully saved 'clean_articles.rds' and 'dtm_tfidf_bigram.rds'.\n")




#EDA: WORD FREQUENCY & VISUALIZATION (using Lemmatization) --
word_freq <- unigrams %>%
  count(term, sort = TRUE) %>%
  filter(!is.na(term))

cat("\n--- TOP 20 MOST FREQUENT unigrams ---\n")
print(head(word_freq, 20))


#Bar Plot of Top 20 Words
cat("\n📊 Generating bar plot of top 20 words...\n")
top_words_plot <- word_freq %>%
  slice_max(n, n = 20) %>%
  ggplot(aes(x = reorder(term, n), y = n)) +
  geom_col(fill = "#1D76B5") +
  coord_flip() +
  labs(
    title = "Top 20 Most Frequent Words in Dhaka Tribune Articles",
    subtitle = "Based on unigrams",
    x = "Word (term)",
    y = "Frequency"
  ) +
  theme_minimal()

print(top_words_plot)

ggsave(file.path(OUT_DIR, "Bar Plot(Top 20).png"), width = 7, height = 5, dpi = 300, bg = "white")



#Word Cloud
cat("\n Generating word cloud...\n")
suppressWarnings({
  wordcloud(
    words = word_freq$term,
    freq = word_freq$n,
    max.words = 150,
    random.order = FALSE,
    #scale = c(4, 0.5),
    colors = RColorBrewer::brewer.pal(8, "Dark2")
  )
})

png(file.path(OUT_DIR, "wordcloud.png"), width = 1600, height = 1200, res = 150)
wordcloud(words = word_freq$term, freq = word_freq$n, max.words = 150, random.order = FALSE, colors = RColorBrewer::brewer.pal(8, "Dark2"))
dev.off()




#VISUALS PER CATEGORY
#Top Words per Section
cat("\n📊 Generating bar plot of top 15 words for each section...\n")
section_top_words <- unigrams %>%
  count(Section, term, sort = TRUE) %>%
  group_by(Section) %>%
  slice_max(n, n = 15) %>%
  ungroup()

section_plot <- section_top_words %>%
  ggplot(aes(x = reorder_within(term, n, Section), y = n, fill = Section)) +
  geom_col(show.legend = FALSE) +
  scale_x_reordered() +
  coord_flip() +
  facet_wrap(~Section, scales = "free") + 
  labs(
    title = "Top 15 Most Frequent Words by each Section",
    subtitle = "Highlights the unique vocabulary of each category",
    x = "Word (term)",
    y = "Frequency"
  ) +
  theme_minimal()

print(section_plot)

ggsave(file.path(OUT_DIR, "Top word per section(Top 15).png"), width = 7, height = 5, dpi = 300, bg = "white")
cat("saved plot to Dhaka_Tribute ouput DIR")



#Bigram Network Graph (Word Pairs)
bigrams <- clean_articles %>%
  unnest_tokens(bigram, Text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         nchar(word1) >= 3, nchar(word2) >= 3) %>%
  count(word1, word2, sort = TRUE)

bigram_graph <- bigrams %>%
  filter(n > 20) %>%
  graph_from_data_frame()

set.seed(123)
p_bigram <- ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "steelblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1.5, size = 3) +
  labs(title = "Bigram Network (Word Co-occurrence)") +
  theme_void()

print(p_bigram)
ggsave(file.path(OUT_DIR, "bigram_network.png"), p_bigram, width = 8, height = 6, dpi = 300, bg ="white")



#Word Cloud by Section
sections <- unique(unigrams$Section)

for (sec in sections) {
  wc_df <- unigrams %>%
    filter(Section == sec) %>%
    count(term, sort = TRUE) %>%
    filter(n > 5)
  
  png(file.path(OUT_DIR, paste0("wordcloud_", sec, ".png")), width = 1600, height = 1200, res = 150)
  par(mar = c(1,1,1,1))
  wordcloud(words = wc_df$term, freq = wc_df$n, max.words = 100,
            random.order = FALSE, colors = RColorBrewer::brewer.pal(8, "Dark2"))
  dev.off()
}










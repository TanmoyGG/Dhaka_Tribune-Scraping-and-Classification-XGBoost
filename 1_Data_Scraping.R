#Phase 1 (Data Scraping)

# Install and load necessary packages
if(!require(rvest)) install.packages("rvest")
if(!require(httr)) install.packages("httr")
if(!require(jsonlite)) install.packages("jsonlite")
library(rvest)
library(httr)
library(jsonlite)


#CONFIGURATION
BASE <- "https://www.dhakatribune.com"
OUT_DIR <- "Dhaka_Tribune"
if(!dir.exists(OUT_DIR)) dir.create(OUT_DIR)

#Scraping Parameters 
MAX_CLICKS <- 25     
ARTICLES_PER_CLICK <- 16


#API Configuration Data Frame
#Stores the unique widget and page IDs for each section's API endpoint
SECTIONS_API_CONFIG <- data.frame(
  section_name = c("bangladesh", "world", "business", "sport"),
  widget_id    = c(704, 704, 704, 704),
  page_id      = c(1116, 1129, 1094, 1095)
)


#Custom User-Agent
ua <- httr::user_agent(
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) R/rvest academic scraper Chrome/120 Safari/537.36"
)


#Scrape links using the API
get_section_links_api <- function(section_name, widget_id, page_id) {
  
  #Scrape the initial articles from the main section page
  cat("Scanning initial page for section:", section_name, "\n")
  initial_url <- paste0(BASE, "/", section_name)
  page <- try(httr::GET(initial_url, ua, timeout(20)), silent = TRUE)
  
  if(inherits(page, "try-error") || http_error(page)) {
    initial_links <- character()
  } else {
    initial_links <- read_html(page) %>%
      html_elements("a") %>%
      html_attr("href") %>%
      grep("/[0-9]{5,}/", ., value = TRUE)
  }
  
  all_links <- initial_links
  
  #Loop to call the API for more content, simulating "+ MORE" clicks
  for (p in 1:MAX_CLICKS) {
    start_index <- p * ARTICLES_PER_CLICK
    
    api_url <- paste0(
      "https://www.dhakatribune.com/api/theme_engine/get_ajax_contents?widget=", widget_id,
      "&start=", start_index, "&count=", ARTICLES_PER_CLICK, "&page_id=", page_id
    )
    
    cat("  -> Calling API (click", p, "):", section_name, "\n")
    
    resp <- try(httr::GET(api_url, ua, timeout(20)), silent = TRUE)
    if(inherits(resp, "try-error") || http_error(resp)) next
    
    json_data <- httr::content(resp, as = "parsed", type = "application/json")
    
    if (is.null(json_data$html) || nchar(json_data$html) < 50) next
    
    api_links <- read_html(json_data$html) %>%
      html_elements("a") %>%
      html_attr("href") %>%
      grep("/[0-9]{5,}/", ., value = TRUE)
    
    all_links <- c(all_links, api_links)
    Sys.sleep(runif(1, 1, 2.5)) 
  }
  
  links <- all_links[!is.na(all_links)]
  links <- ifelse(startsWith(links, "/"), paste0(BASE, links), links)
  return(unique(links))
}


#Scrape the content of a single article
get_article <- function(url) {
  Sys.sleep(runif(1, 1.5, 3.5))
  cat("Fetching:", url, "\n")
  
  page <- try(httr::GET(url, ua, timeout(20)), silent = TRUE)
  if(inherits(page, "try-error") || http_error(page)) return(NULL)
  
  doc <- read_html(httr::content(page, type = "text", encoding = "UTF-8"))
  
  title <- doc %>% html_element("title") %>% html_text(trim = TRUE)
  body <- doc %>%
    html_elements("article p, .details-content p, .post-content p") %>%
    html_text(trim = TRUE) %>%
    paste(collapse = "\n")
  
  if(nchar(body) < 50) return(NULL)
  
  data.frame(
    url = url,
    title = title,
    text = body,
    stringsAsFactors = FALSE
  )
}


#MAIN SCRAPING PROCESS --
articles <- list()

for (i in 1:nrow(SECTIONS_API_CONFIG)) {
  s_name <- SECTIONS_API_CONFIG$section_name[i]
  w_id   <- SECTIONS_API_CONFIG$widget_id[i]
  p_id   <- SECTIONS_API_CONFIG$page_id[i]
  
  cat("\n--- Starting Section:", toupper(s_name), "---\n")
  links <- get_section_links_api(
    section_name = s_name,
    widget_id = w_id,
    page_id = p_id
  )
  
  section_articles <- purrr::map_dfr(
    links,
    ~{
      res <- purrr::safely(get_article, otherwise = NULL)(.x)
      if (!is.null(res$result)) {
        res$result$Section <- s_name
        return(res$result)
      } else {
        return(NULL)
      }
    }
  )
  
  articles[[s_name]] <- section_articles
}

articles_df <- dplyr::bind_rows(articles)

colnames(articles_df) <- c("URL", "Title", "Text", "Section")

outfile <- file.path(OUT_DIR, "dhakatribune_articles.csv")
write.csv(articles_df, outfile, row.names = FALSE, fileEncoding = "UTF-8")

cat("\n successfully scraped", nrow(articles_df), "articles across",
    length(unique(articles_df$Section)), "sections.\n")
cat("Saved to:", outfile, "\n")



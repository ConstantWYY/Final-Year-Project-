# Prompts for Preprocessing
## Data Cleaning Prompt - cleaning_prompt.docx
This prompt is responsible for preprocessing and standardising raw news datasets before further analysis.  
It performs the following tasks:
1. Remove news entries with missing values
2. Remove news entries with promotional content from the Finhub API
3. Remove identical news entries
4. Formatting

## Deduplication Prompt - deduplicate_prompt.docx
This prompt is designed to remove semantically similar news entries from the cleaned datasets.  
It performs the following tasks:
1. Group news into Monday-Sunday weekly buckets
2. Use TF-IDF to measure the similarity between news entries
3. Group news entries into clusters with cosine similarity greater than 0.7
4. Retain only one news entry with the longest headline + summary from each cluster

# Prompts used in calculating the sentiment score
<h2>Reminders</h2>
<p>** When using the prompt, you should replace &lt;Company Name&gt; and  &lt;Company Ticker&gt; with the actual company's name and ticker.</p>

<div>
<h2>Relevance Prompt</h2>
<p>The relevance_prompt.docx file is used to check the relevance of the news to the Company.</p>
</div>

<div>
<h2>Directional Sentiment Prompt</h2>
<p>The directional_sentiment_prompt.docx file is used to determine the directional sentiment of the news.</p>
</div>



# Programs

## data_cleaning.py
This is a program based on the [data-cleaning-prompt](../Prompts).  
It performs the following task:
1. Remove rows with missing values
2. Remove all the Zacks.com promotion content from the Finhub API
3. Remove identical news entries
4. Formatting

## relevance.py
This is a program based on the [relevance-prompt](../Prompts).  
It requires a config file and a CSV file as input.  
It uses keyword matching to assign labels and scores to each financial news data.  
We did not choose to use any Machine learning model because we do not have time for labeling the dataset manually.  
It ranges from [0,1], where:
* Directly Related: 0.70 - 1.00
* Indirectly Related: 0.30 - 0.69
* Unrelated: 0.00 - 0.29


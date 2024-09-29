**About the Dataset**
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

**Project Title:** Fraud Detection in Financial Transactions using Self-Organizing Maps

**Description:**
This project aims to detect fraudulent transactions in financial data using a Self-Organizing Map (SOM) neural network. The SOM is trained on a dataset of credit card transactions, and nodes representing fraudulent activities are identified based on their proximity to other nodes.SOMs cluster similar transactions based on their features (e.g., transaction amount, time, location, merchant category). Outliers in the SOM map, which are transactions that are far from other clusters, are often indicative of fraudulent activity. By visualizing the SOM, analysts can identify regions with a high concentration of fraudulent transactions. Additionally, SOMs can be used to create new features based on the proximity of transactions to different nodes on the map, which can be used as input to other machine learning models for fraud detection. Overall, SOMs provide a valuable tool for financial institutions to detect fraudulent transactions and protect their customers from financial losses. 

**Software Requirements:**
google colab
Libraries:
pandas
numpy
matplotlib
seaborn
minisom
scikit-learn

**Hardware Requirements:**
A computer with sufficient RAM and processing power to handle the dataset size. For large datasets, a GPU may be beneficial for faster training.

**Execution steps :**
The model's performance is evaluated using various metrics such as accuracy, precision, recall, and F1-score.
The code first loads and preprocesses the credit card transaction dataset. It then creates and trains a SOM on the data, using the trained SOM to visualize the distribution of transactions and identify nodes with a high density of fraudulent activities. The performance of the model is evaluated using various metrics. This approach effectively utilizes SOMs to detect anomalous transactions and identify potential fraudulent activities in credit card data.

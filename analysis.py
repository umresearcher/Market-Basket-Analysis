# Import necessary libraries
import streamlit as st 
import mlxtend
import plotly

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#add side for uploading file
uploaded_file = st.sidebar.file_uploader("Choose a file")

#default dataset: read the dataset
transactions_df = pd.read_csv('my_transactions.csv')#change the path to the dataset


if uploaded_file is not None:
    transactions_df = pd.read_csv(uploaded_file)
    st.write(transactions_df)
    st.write(transactions_df.columns)
    
#add title to the app
st.title('Market Basket Analysis')

#display the dataset
st.write('Display the dataset')
st.write(transactions_df)

#conver column of transactions to list
transactions = transactions_df['Items'].apply(lambda x: x.split(','))

# Initialize the TransactionEncoder
te = TransactionEncoder()

# Transform the list of transactions into an array of booleans
te_ary = te.fit(transactions).transform(transactions)

# Convert the array into a DataFrame
transactions_encoded = pd.DataFrame(te_ary, columns=te.columns_)
st.write('Display the encoded dataset')
# Display the encoded transaction dataset
st.write(transactions_encoded)


# Apply the Apriori algorithm to find frequent itemsets with min support of 0.4 (40%)
frequent_itemsets = apriori(transactions_encoded, min_support=0.4, use_colnames=True)

# Display the frequent itemsets found
st.write('Display the frequent itemsets')
st.write(frequent_itemsets)

#plot support value of the frequent itemsets as line chart using plotly not plotly express
st.write('Plot the support value of the frequent itemsets')
line_fig = plotly.graph_objs.Figure()
line_fig.add_trace(plotly.graph_objs.Scatter(x=frequent_itemsets.index, y=frequent_itemsets['support']))
st.plotly_chart(line_fig)

#plot histogram of support value of the frequent itemsets
st.write('Plot histogram of the support value of the frequent itemsets')
his_fig = plotly.graph_objs.Figure()
his_fig.add_trace(plotly.graph_objs.Histogram(x=frequent_itemsets['support']))
#set bin = 20
his_fig.update_layout(bargap=0.1)

st.plotly_chart(his_fig)

# Generate association rules from the frequent itemsets with a minimum confidence of 0.5 (50%)
association_rules_df = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# Display the association rules
st.write('Display the association rules')
st.write(association_rules_df)

# Sort rules by highest lift to see the strongest associations at the top
association_rules_df = association_rules_df.sort_values(by='lift', ascending=False)

# Display the sorted association rules
st.write('Display the sorted association rules')
st.write(association_rules_df)

# Filter the association rules to only show rules where the antecedent is 'Bread'
bread_df = association_rules_df[association_rules_df['antecedents'].apply(lambda x: 'Bread' in x)]

# Display the association rules where the antecedent is 'Bread'
st.write('Display the association rules where the antecedent is Bread')
st.write(bread_df)

# Filter the association rules to only show rules where the minimum threshold = 0.5
min_threshold_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display the association rules where the minimum threshold = 0.5
st.write('Display the association rules where the minimum threshold = 0.5')
st.write(min_threshold_rules)


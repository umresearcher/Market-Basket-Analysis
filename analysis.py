# Import necessary libraries
import streamlit as st 
import mlxtend
import plotly

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#add side for uploading file
uploaded_file = st.sidebar.file_uploader("Upload a file")

#default dataset: read the dataset
transactions_df = pd.read_csv('my_transactions.csv')#change the path to the dataset


if uploaded_file is not None:
    transactions_df = pd.read_csv(uploaded_file)
    st.write(transactions_df)
    st.write(transactions_df.columns)
    

#put a download button
st.sidebar.markdown('Download the template') 
st.sidebar.markdown('[The template](https://github.com/TZ2024/Market-Basket-Analysis/blob/main/my_transactions.csv)')#update the link to the dataset
      

#add title to the app
st.title('Market Basket Analysis')


#create pages tabs
tab_intro,tab_encoded,tab_freq,tab_s_value,tab_associa,tab_filter = st.tabs(['Introduction','The encoded dataset','The frequent itemsets','The support value','The association rules','Filter functions'])



    

with tab_intro:
    st.write('Introduction')
    st.write('Market Basket Analysis is a data mining technique used by retailers to understand the purchase behavior of customers. It involves identifying items that are frequently purchased together. This information can be used to improve product placement, promotions, and marketing strategies. In this project, we will use the Apriori algorithm to perform Market Basket Analysis on a dataset of transactions. We will identify frequent itemsets and generate association rules to uncover patterns in the data. We will also visualize the results to gain insights into customer behavior.')
    #display the dataset
    st.write('Display the dataset')
    st.write(transactions_df)

with tab_encoded:
    st.write('The encoded dataset')
    #add some explanation for the encoded dataset
    st.write('Display the encoded dataset')
    #conver column of transactions to list
    transactions = transactions_df['Items'].apply(lambda x: x.split(','))

    # Initialize the TransactionEncoder
    te = TransactionEncoder()

    # Transform the list of transactions into an array of booleans
    te_ary = te.fit(transactions).transform(transactions)

    # Convert the array into a DataFrame
    transactions_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    # Display the encoded transaction dataset
    st.write(transactions_encoded)

with tab_freq:
    st.write('The frequent itemsets')
    
    # Apply the Apriori algorithm to find frequent itemsets with min support of 0.4 (40%)
    frequent_itemsets = apriori(transactions_encoded, min_support=0.4, use_colnames=True)

    # Display the frequent itemsets found

    st.write(frequent_itemsets)

with tab_s_value:
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

with tab_associa:
    st.write('Display the association rules')
    # Generate association rules from the frequent itemsets with a minimum confidence of 0.5 (50%)
    association_rules_df = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

    # Display the association rules
    st.write(association_rules_df)

    # Sort rules by highest lift to see the strongest associations at the top
    association_rules_df = association_rules_df.sort_values(by='lift', ascending=False)

    # Display the sorted association rules
    st.write('Display the sorted association rules')
    st.write(association_rules_df)

with tab_filter:
    
    #read all items in the antecedents column
    all_items = set()
    for items in association_rules_df['antecedents']:
        all_items.update(items)
    all_items = list(all_items)
    
    #create a selectbox for the items
    selected_item = st.selectbox('Select an item:', all_items)
    
    
    # Filter the association rules to only show rules where the antecedent is 'Bread'
    bread_df = association_rules_df[association_rules_df['antecedents'].apply(lambda x: selected_item in x)]

    # Display the association rules where the antecedent is 'Bread'
    st.write('Display the association rules where the antecedent is '+selected_item)
    st.write(bread_df)

    
    #create a slide for the threshold value
    threshold = st.slider('Select the minimum threshold:', 0.0, 1.0, 0.5, 0.1)
    st.write("Values:", threshold)
    
    # Filter the association rules to only show rules where the minimum threshold = 0.5
    min_threshold_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=threshold)

    # Display the association rules where the minimum threshold = 0.5
    st.write('Display the association rules where the minimum threshold = '+str(threshold))
    st.write(min_threshold_rules)


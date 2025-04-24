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
st.sidebar.markdown('[The template](https://github.com/umresearcher/Market-Basket-Analysis/blob/main/my_transactions.csv)')#update the link to the dataset
      

#add title to the app
st.title('Market Basket Analysis')


#create pages tabs
tab_intro,tab_encoded,tab_freq,tab_s_value,tab_associa,tab_filter = st.tabs(['Introduction','The encoded dataset','The frequent itemsets','The support value','The association rules','Filter functions'])


with tab_intro:
    st.header("Introduction")
    st.markdown("""
    **Overview of Market Basket Analysis**  
    Market Basket Analysis is a data mining technique used to uncover patterns in customer purchase data. It identifies which items are frequently bought together and helps businesses optimize product placement, promotions, and marketing strategies. [Reference: https://www.geeksforgeeks.org/market-basket-analysis-in-data-mining/](https://www.geeksforgeeks.org/market-basket-analysis-in-data-mining/)
    
    In this section, you will see a brief introduction to the concept and a display of the raw transaction dataset.
    """)
    #display the dataset
    st.write('Display the dataset')
    #st.write(transactions_df)
    #st.dataframe(transactions_df, index=False)
    st.dataframe(transactions_df, hide_index=True)



with tab_encoded:
    st.header("The Encoded Dataset")
    #add some explanation for the encoded dataset
    st.markdown("""
    **Data Encoding Explanation**  
    To perform Market Basket Analysis, the raw transaction data must be converted into a Boolean matrix. Each row represents a transaction, and each column represents an item. A value of True indicates the item was purchased in that transaction.
    
    This section displays the encoded dataset after processing the 'Items' column.
    """)
    st.write('Display the encoded dataset')
    #conver column of transactions to list -- trim whitespaces
    #transactions = transactions_df['Items'].apply(lambda x: x.split(','))
    transactions = transactions_df['Items'].apply(lambda x: [item.strip() for item in x.split(',')])

    # Initialize the TransactionEncoder
    te = TransactionEncoder()

    # Transform the list of transactions into an array of booleans
    te_ary = te.fit(transactions).transform(transactions)

    # Convert the array into a DataFrame
    transactions_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    # Display the encoded transaction dataset
    st.write(transactions_encoded)

with tab_freq:
    st.header("The Frequent Itemsets")
    st.markdown("""
    **Frequent Itemsets Overview**  
    Frequent itemsets are combinations of items that appear together in a significant portion of transactions.  
    We use the Apriori algorithm with a minimum support threshold of 0.4 (40%). This means only itemsets that appear in at least 40% of all transactions are shown. These frequent itemsets form the basis for generating association rules.
    """)
    
    # Apply the Apriori algorithm to find frequent itemsets with min support of 0.4 (40%)
    frequent_itemsets = apriori(transactions_encoded, min_support=0.4, use_colnames=True)

    # Display the frequent itemsets found

    st.write(frequent_itemsets)

with tab_s_value:
    #plot support value of the frequent itemsets as line chart using plotly not plotly express
    st.header("Support Value Visualization")
    st.markdown("""
    **Understanding Support**  
    Support represents the proportion of transactions in which a specific itemset appears. For example, a support of 0.6 means that 60% of transactions contain that itemset.
    
    Here, we display the support values of frequent itemsets using a line chart and a histogram. These visualizations help you understand how common different item combinations are.
    """)
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
    st.header("Association Rules")
    st.markdown("""
    **Explaining Association Rules**  
    Association rules describe how the presence of one item (antecedent) implies the presence of another item (consequent).  
    - **Support:** The proportion of transactions that include the itemset (both antecedent and consequent).  
    - **Confidence:** The probability that transactions containing the antecedent also contain the consequent.  
    - **Lift:** A measure of how much more likely the consequent is purchased when the antecedent is purchased, compared to random chance.
    
    In this section, association rules are generated with a minimum confidence threshold of 0.5 (50%).
    """)
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
    st.header("Filter Functions and Custom Rules")
    st.markdown("""
    **Interactive Filtering**  
    This section allows you to filter association rules based on specific criteria:
    - Select an item from the dropdown to display rules where that item is in the antecedents.
    - Use the slider to adjust the minimum confidence threshold for the rules.
    
    These filters help you focus on the most relevant rules for your business analysis.
    """)
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
    st.write("Selected confidence threshold:", threshold)
    
    # Filter the association rules to only show rules where the minimum threshold = 0.5
    min_threshold_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=threshold)

    # Display the association rules where the minimum threshold = 0.5
    st.write('Association Rules with Minimum Confidence = '+str(threshold))
    st.write(min_threshold_rules)


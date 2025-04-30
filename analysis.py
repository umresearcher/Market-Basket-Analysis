# Import necessary libraries
import streamlit as st 
import mlxtend
import plotly

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Function to load CSS file
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css('styles.css')

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
#tab_intro,tab_encoded,tab_itemset,tab_freq,tab_s_value,tab_associa,tab_filter = st.tabs(['Introduction','The encoded dataset','ItemSets and Support','Frequent Itemsets','The support value','The association rules','Filter functions'])
tab_intro,tab_encoded,tab_itemset,tab_freq,tab_associa,tab_metrics,tab_other_metrics,tab_filter,tab_references = st.tabs(['Introduction','The encoded dataset','ItemSets and Support','Frequent Itemsets','Association rules','Metrics for Association Rules','Other Metrics','Filter functions','Further Reading'])

with tab_intro:
    #st.header("Introduction")
    st.header("Overview of Market Basket Analysis")

    #st.markdown("""**Overview of Market Basket Analysis**""")

    st.markdown("""
    Market Basket Analysis is a data mining technique used to uncover patterns in data such as customer purchase data. 
    It identifies which items are frequently bought together and helps businesses optimize product placement, promotions, 
    and marketing strategies. 

    """)

    # In this section, you will see a brief introduction to the concept and a display of the raw transaction dataset.
    st.markdown("""
    You can upload your own transactions and items (as a .csv file) instead of the default provided. Any data you enter should be based on the template. The items in a transaction are comma separated, and spaces around an item are trimmed. Each transaction must have 1 or more items. Different transactions can hve different number of items.
    """)
    #display the dataset
    st.write('These are the transactions and the items in each transaction from your data.')
    #st.write(transactions_df)
    st.dataframe(transactions_df, hide_index=True)

with tab_encoded:
    st.header("The Encoded Dataset")
    #add some explanation for the encoded dataset
    #To perform Market Basket Analysis, the raw transaction data must be converted into a Boolean matrix. Each row represents a transaction, and each column represents an item. A value of True indicates the item was purchased in that transaction.
    #This section displays the encoded dataset after processing the 'Items' column.

    st.markdown("""
    **Data Encoding Explanation**  
    Let us view the dataset as a Boolean matrix. Each row represents a transaction, and each column represents an item. A value of True indicates the item was purchased in that transaction.
    
    This section displays the encoded dataset. If you see any errors in the encoded dataset, you may want to go back and check the csv file has the right values and follows the template.

    """)
    #st.write('Display the encoded dataset')
    #conver column of transactions to list -- trim whitespaces
    #transactions = transactions_df['Items'].apply(lambda x: x.split(','))
    transactions = transactions_df['Items'].apply(lambda x: [item.strip() for item in x.split(',')])

    # Initialize the TransactionEncoder
    te = TransactionEncoder()

    # Transform the list of transactions into an array of booleans
    te_ary = te.fit(transactions).transform(transactions)

    # Convert the array into two DataFrames - one for future processing and one for the display on this page
    transactions_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    transactions_encoded_disp = transactions_encoded.copy()

    #transactions_encoded_disp.insert(0, 'Transaction_ID', transactions_df['Transaction_ID'])
    transactions_encoded_disp.insert(0, 'Transaction_ID', transactions_df['Transaction_ID'])

    # Add the Transaction_ID column back to the encoded DataFrame
    #transactions_encoded_disp['Transaction_ID'] = transactions_df['Transaction_ID']
    
    # Reorder columns to display Transaction_ID first
    #transactions_encoded_disp = transactions_encoded_disp[['Transaction_ID'] + [col for col in transactions_encoded_disp.columns if col != 'Transaction_ID']]
    
    # Display the encoded transaction dataset
    #st.write(transactions_encoded)
    #st.dataframe(transactions_encoded_disp, hide_index=True)
    st.dataframe(transactions_encoded_disp, hide_index=True)

with tab_itemset:
    #st.header("Itemsets and Support")

    # HTML code for the definition of itemset
    html_code = """
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <h2>Itemset</h2>
        <p>An <strong>itemset</strong> is simply a collection of one or more items.
        In the context of market basket analysis, an itemset represents a group of items that are frequently purchased together.</p>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    # Your HTML code
    html_code = """
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <h2>Support</h2>
        <p><strong>Support</strong> is a measure used to indicate how frequently an itemset appears in the dataset. It is defined as the proportion of transactions in the dataset that contain the itemset. Mathematically, support is calculated as:</p>
        <p><b>Support(X) = </b> 
        <span class="fraction">
            <span class="numerator">Number of transactions containing X</span>
            <span class="denominator">Total number of transactions</span>
        </span> 
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    st.markdown("""
    Choose itemsets below, and we highlight the transactions where the itemset appears (that is, the transactions that includes all the items in the itemset), and its support.
    """)

    # Display items for selection
    selected_items = st.multiselect('Select items:', options=te.columns_)
    
        # Highlight transactions containing selected items
    highlighted_transactions = transactions_encoded[transactions_encoded[selected_items].all(axis=1)]

    # Function to apply highlighting
    def highlight_rows(row):
        return ['background-color: lightblue' if row.name in highlighted_transactions.index else '' for _ in row]

    # Display the encoded transaction dataset without the index
    st.dataframe(transactions_encoded.style.apply(highlight_rows, axis=1), hide_index=True)

    # Calculate support for the selected items
    support = len(highlighted_transactions) / len(transactions_encoded)
    num_transactions_containing_itemset = len(highlighted_transactions)
    total_transactions = len(transactions_encoded)

    # Highlight transactions
    if not selected_items:
        st.markdown("""
        <p><strong>No</strong> items selected.</p>
        """, unsafe_allow_html=True)

    # Display support at the bottom
    st.markdown(f"""
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <h2>Support for Selected Itemset</h2>
        <p>The support for the selected itemset is: 
            <span class="tooltip"><strong>{support:.2f}</strong>
            <span class="tooltiptext" style="width: 450px;">Calculated as: 
            <span class="fraction">
            <span class="numerator">Number of Transactions containing the Itemset ({num_transactions_containing_itemset})</span>
            <span class="denominator">Total number of Transactions ({total_transactions})</span>
            </span></span></span>
    </div>
    """, unsafe_allow_html=True)

with tab_freq:
    st.header("Frequent Itemsets")
    st.markdown("""
    In the context of market basket analysis, a <strong>frequent itemset</strong> is an itemset that appear together in the set of transactions 
    with a frequency that meets or exceeds a specified threshold. This threshold is known as the <strong>minimum support</strong>, 
    and is set by the analyst.
    
    For instance, suppose we set the minimum support threshold of 0.4 (40%). In this case, the frequent itemsets are the itemsets
    whose support is &gt;= 0.4 (that is, the itemset appears in at least 40% of the transactions).

    For market basket analysis, we are interested in frequent itemsets, as itemsets that occur in only a few transactions are not
    interesting to detect meaningful patterns. 
    """, unsafe_allow_html=True)

    # Input for minimum support threshold
    min_support = st.slider('Set the minimum support threshold:', min_value=0.0, max_value=1.0, value=0.4, step=0.01)

    # Calculate frequent itemsets
    frequent_itemsets = apriori(transactions_encoded, min_support=min_support, use_colnames=True)

    # Check if there are any frequent itemsets
    if frequent_itemsets.empty:
        st.warning(f'There are no itemsets with support >= {min_support:.2f}.')
    else:
        # Sort frequent itemsets by support in descending order
        frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

        #form frequent_itemsets_disp for display purposes
        frequent_itemsets_disp = frequent_itemsets.copy()
        frequent_itemsets_disp['itemsets'] = frequent_itemsets_disp['itemsets'].apply(lambda x: ', '.join(list(x)))
        #Switch columns so that itemset appears first and support second
        frequent_itemsets_disp = frequent_itemsets_disp[['itemsets', 'support']]
        # Apply custom CSS class to support column
        frequent_itemsets_disp['support'] = frequent_itemsets_disp['support'].apply(lambda x: f'<div class="left-align">{x}</div>')

        #Display frequent itemsets in a table
        st.markdown(frequent_itemsets_disp.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Display frequent itemsets in a table
    #st.dataframe(frequent_itemsets_disp, hide_index=True)

    # Display frequent itemsets in a table
    #st.dataframe(frequent_itemsets, hide_index=True)


    # Apply the Apriori algorithm to find frequent itemsets with min support of 0.4 (40%)
    #frequent_itemsets = apriori(transactions_encoded, min_support=0.4, use_colnames=True)

    # Display the frequent itemsets found
    #st.write(frequent_itemsets)

# with tab_s_value:
#     #plot support value of the frequent itemsets as line chart using plotly not plotly express
#     st.header("Support Value Visualization")
#     st.markdown("""
#     **Understanding Support**  
#     Support represents the proportion of transactions in which a specific itemset appears. For example, a support of 0.6 means that 60% of transactions contain that itemset.
    
#     Here, we display the support values of frequent itemsets using a line chart and a histogram. These visualizations help you understand how common different item combinations are.
#     """)
#     line_fig = plotly.graph_objs.Figure()
#     line_fig.add_trace(plotly.graph_objs.Scatter(x=frequent_itemsets.index, y=frequent_itemsets['support']))
#     st.plotly_chart(line_fig)

#     #plot histogram of support value of the frequent itemsets
#     st.write('Plot histogram of the support value of the frequent itemsets')
#     his_fig = plotly.graph_objs.Figure()
#     his_fig.add_trace(plotly.graph_objs.Histogram(x=frequent_itemsets['support']))
#     #set bin = 20
#     his_fig.update_layout(bargap=0.1)

#     st.plotly_chart(his_fig)

with tab_associa:
    st.header("Association Rules")
    st.markdown("""
    An association rule is typically expressed in the form of **"X → Y"**. It can also be expressed in the form of 
    "<strong>If-Then</strong>" statements as If **X**, Then **Y**. Here the presence of one set of items (the 
    <strong>antecedent</strong> itemset, **X**) implies the presence of another set of items (the <strong>consequent</strong> 
    itemset, **Y**) with a certain level of <strong>confidence</strong> and <strong>support</strong>. We
    will examine <strong>confidence</strong>, <strong>support</strong>, and other metrics for association rules in the next section.


    <!--where the presence of one set of items 
    (the antecedent, X) implies the presence of another set of items (the consequent, Y) with a certain level of confidence and support.

                
    An association rule, written as **X → Y**, where **X** and **Y** are itemsets, is a fundamental concept in market basket 
    analysis. It is used to identify relationships between items in the set of transactions. An association rule **X → Y**
    can also be expressed in the form of "<strong>If-Then</strong>" statements as If **X**, Then **Y**. Here the presence of 
    one set of items (the <strong>antecedent</strong> itemset, **X**) implies the presence of another set of items (the 
    <strong>consequent</strong> itemset, **Y**) with a certain level of <strong>confidence</strong> and <strong>support</strong>. We
    will examine <strong>confidence</strong>, <strong>support</strong>, and other metrics for association rules in the next section. -->

    An association rule can have a single item or multiple items in the antecedent (**X**)/consequent (**Y**) itemsets.

    <!-- For example, an association rule might state: "If a customer buys bread, then they are likely to buy butter." This rule 
    indicates that there is a significant relationship between the purchase of bread and butter. -->
    """, unsafe_allow_html=True)

    min_support_exp = 1 / len(transactions_encoded)
    frequent_itemsets_exp = apriori(transactions_encoded, min_support=min_support_exp, use_colnames=True)
    min_threhold_exp = 1 / len(transactions_encoded)

    # Generate association rules
    rules = association_rules(frequent_itemsets_exp, metric="confidence", min_threshold=min_threhold_exp)

    # Select examples
    single_item_rules = rules[(rules['antecedents'].apply(lambda x: len(x) == 1)) & (rules['consequents'].apply(lambda x: len(x) == 1))].sort_values(by=['support', 'confidence'], ascending=False)

    # Check if there is at least one rule
    if single_item_rules.empty:
        st.warning('No association rules found in the given dataset. Please change the dataset.')
    else:
        single_item_rule = single_item_rules.iloc[0]
        # Format rules for display
        single_item_antecedent = ', '.join(list(single_item_rule['antecedents']))
        single_item_consequent = ', '.join(list(single_item_rule['consequents']))
        st.markdown(f"""
        ### Example Association Rules from Dataset

        **Single-item sets:**
        - **Rule:** **{single_item_antecedent} -> {single_item_consequent}** (If a customer buys **{single_item_antecedent}**, then they are likely to buy **{single_item_consequent}**).
        - **Support:** {single_item_rule['support']:.2f} (the proportion of transactions that contain both **{single_item_antecedent}** and **{single_item_consequent}**; these are the transactions highlighted in green below among all transactions)
        - **Confidence:** {single_item_rule['confidence']:.2f} (the proportion of transactions that contain **{single_item_consequent}** among those that contain **{single_item_antecedent}**; these are the transactions highlighted in green below among all highlighted transactions)
        """, unsafe_allow_html=True)

        # Highlight transactions
        def highlight_rows(row):
            antecedent_items = single_item_rule['antecedents']
            consequent_items = single_item_rule['consequents']
            if all(row[antecedent_items]):
                if all(row[consequent_items]):
                    return ['background-color: lightgreen' for _ in row]
                return ['background-color: lightblue' for _ in row]
            return ['' for _ in row]

        # Display the encoded transaction dataset with highlighting
        st.dataframe(transactions_encoded_disp.style.apply(highlight_rows, axis=1), hide_index=True)

        #multi_item_rule = rules[(rules['antecedents'].apply(lambda x: len(x) > 1)) & (rules['consequents'].apply(lambda x: len(x) > 1))].iloc[0]

        multi_item_rules = rules[(rules['antecedents'].apply(lambda x: len(x) > 1)) & 
                         (rules['consequents'].apply(lambda x: len(x) == 1)) & 
                         (rules['confidence'] < 1)].sort_values(by='support', ascending=False)

        if multi_item_rules.empty:        
            multi_item_rules = rules[(rules['antecedents'].apply(lambda x: len(x) > 1)) & (rules['consequents'].apply(lambda x: len(x) > 1))].sort_values(by=['support', 'confidence'], ascending=False)

        # Check if there is at least one rule
        if multi_item_rules.empty:
            multi_item_rules = rules[(rules['antecedents'].apply(lambda x: len(x) > 1)) & 
                         (rules['consequents'].apply(lambda x: len(x) == 1)) & 
                         (rules['confidence'] < 1)].sort_values(by='support', ascending=False)
        if multi_item_rules.empty:
            multi_item_rules = rules[(rules['antecedents'].apply(lambda x: len(x) > 1)) & (rules['consequents'].apply(lambda x: len(x) == 1))].sort_values(by=['support', 'confidence'], ascending=False)
        if not multi_item_rules.empty:
            multi_item_rule = multi_item_rules.iloc[0]

            # Format rules for display
            multi_item_antecedent = ', '.join(list(multi_item_rule['antecedents']))
            multi_item_consequent = ', '.join(list(multi_item_rule['consequents']))

            st.markdown(f"""
            **Multi-item sets:**
            - **Rule:** **{multi_item_antecedent} -> {multi_item_consequent}** (If a customer buys **{multi_item_antecedent}**, then they are likely to buy **{multi_item_consequent}**).
            - **Support:** {multi_item_rule['support']:.2f} (the proportion of transactions that contain **{multi_item_antecedent}**, **{multi_item_consequent}**; these are the transactions highlighted in green below among all transactions)
            - **Confidence:** {multi_item_rule['confidence']:.2f} (the proportion of transactions that contain **{multi_item_consequent}** among those that contain **{multi_item_antecedent}**; these are the transactions highlighted in green below among all highlighted transactions)
            """)

            # Highlight transactions for multi-item rule
            def highlight_multi_item_rows(row):
                antecedent_items = multi_item_rule['antecedents']
                consequent_items = multi_item_rule['consequents']
                if all(row[antecedent_items]):
                    if all(row[consequent_items]):
                        return ['background-color: lightgreen' for _ in row]
                    return ['background-color: lightblue' for _ in row]
                return ['' for _ in row]

            # Display the encoded transaction dataset with highlighting for multi-item rule
            st.dataframe(transactions_encoded_disp.style.apply(highlight_multi_item_rows, axis=1), hide_index=True)
        else:
            st.warning('No multi-set association rules found in the given dataset. Please change the dataset.')



    #multi_item_antecedent = ', '.join(list(multi_item_rule['antecedents']))
    #multi_item_consequent = ', '.join(list(multi_item_rule['consequents']))



    # st.markdown(f"""
    # ### Example Association Rules from Dataset

    # **Single-item sets:**
    # - **Rule:** If a customer buys **{single_item_antecedent}**, then they are likely to buy **{single_item_consequent}**.
    # - **Support:** {single_item_rule['support']:.2f}
    # - **Confidence:** {single_item_rule['confidence']:.2f}

    # **Multi-item sets:**
    # - **Rule:** If a customer buys **{multi_item_antecedent}**, then they are likely to buy **{multi_item_consequent}**.
    # - **Support:** {multi_item_rule['support']:.2f}
    # - **Confidence:** {multi_item_rule['confidence']:.2f}
    # """, unsafe_allow_html=True)

    # st.markdown("""
    # **Explaining Association Rules**  
    # Association rules describe how the presence of one item (antecedent) implies the presence of another item (consequent).  
    # - **Support:** The proportion of transactions that include the itemset (both antecedent and consequent).  
    # - **Confidence:** The probability that transactions containing the antecedent also contain the consequent.  
    # - **Lift:** A measure of how much more likely the consequent is purchased when the antecedent is purchased, compared to random chance.
    
    # In this section, association rules are generated with a minimum confidence threshold of 0.5 (50%).
    # """)
    # # Generate association rules from the frequent itemsets with a minimum confidence of 0.5 (50%)
    # association_rules_df = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

    # # Display the association rules
    # st.write(association_rules_df)

    # # Sort rules by highest lift to see the strongest associations at the top
    # association_rules_df = association_rules_df.sort_values(by='lift', ascending=False)

    # # Display the sorted association rules
    # st.write('Display the sorted association rules')
    # st.write(association_rules_df)

with tab_metrics:
    st.header("Metrics for Association Rules")
    st.markdown("""
    The two primary metrics used for association rules are **Support** and **Confidence**. These metrics help us understand 
    the frequency and reliability of the rules we generate.

    Consider an association rule: **X → Y**. 
    """)
                
    st.markdown("""
    **Support**:
    For the association rule **X → Y**, support measures how frequently all the items in **X** and **Y** appear in the dataset. 
    This is calculated as:
    """)

    html_code = """
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; 
        background-color: #f9f9f9;">
        <p><b>Support(X→Y) = </b> 
        <span class="fraction">
            <span class="numerator">Number of transactions containing all the items in X and Y</span>
            <span class="denominator">Total number of transactions</span>
        </span> 
        </p>
        <p>We are interested in association rules above a certain support threshold because we want itemsets to appear with some 
        frequency (and not very rarely). This ensures that the rules we find are relevant and useful for making decisions.</p>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    st.markdown("""<br>""", unsafe_allow_html=True)

    st.markdown("""
    **Confidence**:
    Confidence indicates how often the rule holds true in the dataset, and measures the reliability of the rule. It is calculated as:
    """)

    html_code = """
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <p><b>Confidence(X→Y) = </b> 
        <span class="fraction">
            <span class="numerator">Number of transactions containing all the items in X and Y</span>
            <span class="denominator">Number of transactions containing all the items in X</span>
        </span> 
        </p>
        <p>This is the same as:
        <b>Confidence(X→Y) = </b> 
        <span class="fraction">
            <span class="numerator">Support(X→Y)</span>
            <span class="denominator">Support(X)</span>
        </span> 
        </p>
        <p>Confidence helps us understand the reliability of the association rule. A higher confidence means that when itemset 
        <b>X</b> appears, itemset <b>Y</b> is likely to appear as well. This metric is crucial for determining how strong the 
        relationship is between the items.</p>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown("""**Illustrating Support and Confidence. Select Antecedent and Consequent**""")

    # Selection widgets within the tab
    items = te.columns_
    antecedent = st.multiselect("Antecedent", items)
    consequent = st.multiselect("Consequent", items)

    # Function to highlight transactions based on user selections
    def highlight_user_selected_rows(row, antecedent, consequent):
        if all(row[antecedent]):
            if all(row[consequent]):
                return ['background-color: lightgreen' for _ in row]
            return ['background-color: lightblue' for _ in row]
        return ['' for _ in row]

    # Calculate support and confidence
    def calculate_support_confidence(transactions_encoded, antecedent, consequent):
        # Calculate combined support
        combined_support = transactions_encoded[antecedent + consequent].all(axis=1).mean()

        # Calculate confidence
        antecedent_support = transactions_encoded[antecedent].all(axis=1).mean()
        confidence = combined_support / antecedent_support if antecedent_support > 0 else 0

        return combined_support, confidence

    # Apply highlighting based on user selections
    if antecedent and consequent:
        styled_df = transactions_encoded_disp.style.apply(lambda row: highlight_user_selected_rows(row, antecedent, consequent), axis=1)
        st.dataframe(styled_df, hide_index=True)

        # Calculate and display support and confidence
        combined_support, confidence = calculate_support_confidence(transactions_encoded, antecedent, consequent)
        antecedent_str = ', '.join(antecedent)
        consequent_str = ', '.join(consequent)
        num_transactions_containing_itemset = transactions_encoded[antecedent + consequent].all(axis=1).sum()
        total_transactions = len(transactions_encoded)

        st.markdown(f"""
        <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <b>Support and Confidence for Selected Association Rule</b><br/><br/>
            <p>The support for the association rule ({antecedent_str} → {consequent_str}) is: 
                <span class="tooltip"><strong>{combined_support:.2f}</strong>
                <span class="tooltiptext" style="width: 450px;">Calculated as: 
                <span class="fraction">
                <span class="numerator">Number of Transactions containing {antecedent_str}, {consequent_str} ({num_transactions_containing_itemset})</span>
                <span class="denominator">Total number of Transactions ({total_transactions})</span>
                </span></span></span>
            </p>
            <p>The confidence for the association rule ({antecedent_str} → {consequent_str}) is: 
                <span class="tooltip"><strong>{confidence:.2f}</strong>
                <span class="tooltiptext" style="width: 450px;">Calculated as: 
                <span class="fraction">
                <span class="numerator">Support ({antecedent_str} -> {consequent_str})</span>
                <span class="denominator">Support ({antecedent_str})</span>
                </span></span></span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.dataframe(transactions_encoded_disp, hide_index=True)


with tab_other_metrics:
    st.header("Other Metrics for Association Rules")

    st.markdown("""
    While support and confidence are the two primary metrics for association rules, several additional metrics have also been defined. 
    These additional metrics may be useful for your use-case. Let us examine three such metrics: Lift, Leverage, and Conviction.
                
    
    Consider an association rule: **X → Y**. 
    """)

    st.markdown("""
    **Lift**:
        Lift measures the strength of the association rule compared to the expected occurrence of the consequent if the antecedent (**X**)
        and the consequent (**Y**) were independent.
    """)

    st.markdown("""        
        If **X** and **Y** are independent, then the proportion of transactions that include **Y** among all the transactions that 
        include **X** (that is, **Confidence(X → Y)**) should be the same as the proportion of transactions that include **Y** among
        all transactions in the dataset (that is, **Support(Y)**).
         
        **Lift(X → Y)** is calculated as:
    """)

    html_code = """
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <p><b>Lift(X→Y) = </b> 
        <span class="fraction">
            <span class="numerator">Confidence(X→Y)</span>
            <span class="denominator">Support(Y)</span>
        </span> 
        <b> = </b> 
        <span class="fraction">
            <span class="numerator">Support(X→Y)</span>
            <span class="denominator">Support(X) &times; Support(Y)</span>
        </span> 
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <b>Interpretation of Lift Values</b><br/><br/>
        <p><b>Lift &gt; 1:</b> Indicates a positive association between the antecedent (<b>X</b>) and the consequent (<b>Y</b>). 
            This means that the occurrence of <b>X</b> increases the likelihood of <b>Y</b> occurring. The higher the Lift value, 
            the stronger the association. The focus in market basket analysis is generally on identifying association rules with 
            Lift &gt; 1.</p>
        <p><b>Lift = 1:</b> Indicates no association between <b>X</b> and <b>Y</b>. The occurrence of <b>X</b> does not affect the 
                likelihood of <b>Y</b> occurring. <b>X</b> and <b>Y</b> are independent.</p>
        <p><b>Lift &lt; 1:</b> Indicates a negative association between <b>X</b> and <b>Y</b>. This means that the occurrence of 
                <b>X</b> decreases the likelihood of <b>Y</b> occurring. The lower the Lift value, the stronger the negative 
                association. While market basket analysis typically focuses on positive Lift values, negative Lift values (<b>Lift 
                &lt; 1</b>) can indicate products that are substitutes for each other.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<br/>""", unsafe_allow_html=True)

    st.markdown("""
    **Leverage**:
    Leverage measures the difference between the observed frequency of the antecedent (**X**) and the consequent (**Y**) appearing 
    together and the expected frequency if they were independent. If **X** and **Y** are independent, the **Expected Support(X → Y)**
    = **Support(X)** &times; **Support(Y)**.
                
    **Leverage(X → Y)** is calculated as:
    """)

    html_code = """
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <p><b>Leverage(X→Y) = Support(X→Y) - (Support(X) &times; Support(Y))</b></p>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; 
                background-color: #f9f9f9;">
        <b>Interpretation of Leverage Values</b><br/><br/>
        <p><b>Leverage > 0:</b> Indicates a positive association between the antecedent (<b>X</b>) and the consequent (<b>Y</b>). 
                This means that the occurrence of <b>X</b> increases the likelihood of <b>Y</b> occurring more than expected if 
                they were independent.</p>
        <p><b>Leverage = 0:</b> Indicates no association between <b>X</b> and <b>Y</b>. The occurrence of <b>X</b> does not affect 
                the likelihood of <b>Y</b> occurring, and their co-occurrence is as expected if they were independent.</p>
        <p><b>Leverage < 0:</b> Indicates a negative association between <b>X</b> and <b>Y</b>. This means that the occurrence of 
                <b>X</b> decreases the likelihood of <b>Y</b> occurring compared to what would be expected if they were independent.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <b>Relationship Between Leverage and Lift</b><br/><br/>
        <p>We can see that:</p>
        <ul>
            <li><b>Leverage = 0</b> exactly when <b>Lift = 1</b></li>
            <li><b>Leverage > 0</b> exactly when <b>Lift > 1</b></li>
            <li><b>Leverage < 0</b> exactly when <b>Lift < 1</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<br/>""", unsafe_allow_html=True)

    st.markdown("""
    **Conviction**:
    Conviction measures the degree of implication of the antecedent (**X**) in the absence of the consequent (**Y**).

    **Conviction(X → Y)** is calculated as:
    """)

    html_code = """
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <p><b>Conviction(X→Y) = (1 - Support(Y)) / (1 - Confidence(X→Y))</b></p>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <b>Interpretation of Conviction Values</b><br/><br/>
        <p><b>Conviction > 1:</b> Indicates a positive association between the antecedent (<b>X</b>) and the consequent (<b>Y</b>). 
            This means that the occurrence of <b>X</b> implies the occurrence of <b>Y</b> more strongly than expected if they were independent.</p>
        <p><b>Conviction = 1:</b> Indicates no association between <b>X</b> and <b>Y</b>. The occurrence of <b>X</b> does not affect 
            the likelihood of <b>Y</b> occurring, and their co-occurrence is as expected if they were independent.</p>
        <p><b>Conviction < 1:</b> Indicates a negative association between <b>X</b> and <b>Y</b>. This means that the occurrence of 
            <b>X</b> implies the absence of <b>Y</b> more strongly than expected if they were independent.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <b>Relationship Between Conviction, Leverage and Lift</b><br/><br/>
        <p>We can see that:</p>
        <ul>
            <li><b>Conviction = 1</b> exactly when <b>Lift = 1</b>, and <b>Leverage = 0</b></li>
            <li><b>Conviction > 1</b> exactly when <b>Lift > 1</b>, and <b>Leverage > 0</b></li>
            <li><b>Conviction < 1</b> exactly when <b>Lift < 1</b>, and <b>Leverage < 0</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Selection widgets within the tab
    items = te.columns_
    antecedent = st.multiselect("Antecedent", items, key="antecedent_multiselect")
    consequent = st.multiselect("Consequent", items, key="consequent_multiselect")

    # Function to highlight transactions based on user selections
    def highlight_user_selected_rows(row, antecedent, consequent):
        if all(row[antecedent]):
            if all(row[consequent]):
                return ['background-color: lightgreen' for _ in row]
            return ['background-color: lightblue' for _ in row]
        return ['' for _ in row]

    # Calculate support, confidence, lift, leverage, and conviction
    def calculate_metrics(transactions_encoded, antecedent, consequent):
        # Calculate combined support
        combined_support = transactions_encoded[antecedent + consequent].all(axis=1).mean()

        # Calculate confidence
        antecedent_support = transactions_encoded[antecedent].all(axis=1).mean()
        confidence = combined_support / antecedent_support if antecedent_support > 0 else 0

        # Calculate lift
        consequent_support = transactions_encoded[consequent].all(axis=1).mean()
        lift = confidence / consequent_support if consequent_support > 0 else 0

        # Calculate leverage
        leverage = combined_support - (antecedent_support * consequent_support)

        # Calculate conviction
        conviction = (1 - consequent_support) / (1 - confidence) if confidence < 1 else float('inf')

        return combined_support, confidence, lift, leverage, conviction

    # Apply highlighting based on user selections
    if antecedent and consequent:
        styled_df = transactions_encoded_disp.style.apply(lambda row: highlight_user_selected_rows(row, antecedent, consequent), axis=1)
        st.dataframe(styled_df, hide_index=True)

        # Calculate and display metrics
        combined_support, confidence, lift, leverage, conviction = calculate_metrics(transactions_encoded, antecedent, consequent)
        antecedent_str = ', '.join(antecedent)
        consequent_str = ', '.join(consequent)
        num_transactions_containing_itemset = transactions_encoded[antecedent + consequent].all(axis=1).sum()
        total_transactions = len(transactions_encoded)

        st.markdown(f"""
        <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <b>Support, Confidence, Lift, Leverage, and Conviction for Selected Association Rule</b><br/><br/>
            <p>The support for the association rule ({antecedent_str} → {consequent_str}) is: 
                <span class="tooltip"><strong>{combined_support:.2f}</strong>
                <span class="tooltiptext" style="width: 450px;">Calculated as: 
                <span class="fraction">
                <span class="numerator">Number of Transactions containing {antecedent_str}, {consequent_str} ({num_transactions_containing_itemset})</span>
                <span class="denominator">Total number of Transactions ({total_transactions})</span>
                </span></span></span>
            </p>
            <p>The confidence for the association rule ({antecedent_str} → {consequent_str}) is: 
                <span class="tooltip"><strong>{confidence:.2f}</strong>
                <span class="tooltiptext" style="width: 450px;">Calculated as: 
                <span class="fraction">
                <span class="numerator">Support ({antecedent_str} → {consequent_str})</span>
                <span class="denominator">Support ({antecedent_str})</span>
                </span></span></span>
            </p>
            <p>The lift for the association rule ({antecedent_str} → {consequent_str}) is: 
                <span class="tooltip"><strong>{lift:.2f}</strong>
                <span class="tooltiptext" style="width: 450px;">Calculated as: 
                <span class="fraction">
                <span class="numerator">Confidence ({antecedent_str} → {consequent_str})</span>
                <span class="denominator">Support ({consequent_str})</span>
                </span></span></span>
            </p>
            <p>The leverage for the association rule ({antecedent_str} → {consequent_str}) is: 
                <span class="tooltip"><strong>{leverage:.2f}</strong>
                <span class="tooltiptext" style="width: 450px;">Calculated as: <br/>
                Support ({antecedent_str} → {consequent_str}) &minus; <br/>(Support ({antecedent_str}) &times; Support ({consequent_str}))</span>
                </span>
            </p>
            <!-- <p>The leverage for the association rule ({antecedent_str} → {consequent_str}) is: 
                <span class="tooltip"><strong>{leverage:.2f}</strong>
                <span class="tooltiptext" style="width: 450px;">Calculated as: 
                <span class="fraction">
                <span class="numerator">Support ({antecedent_str} → {consequent_str})</span>
                <span class="denominator">Support ({antecedent_str}) &times; Support ({consequent_str})</span>
                </span></span></span>
            </p> -->
            <p>The conviction for the association rule ({antecedent_str} → {consequent_str}) is: 
                <span class="tooltip"><strong>{conviction:.2f}</strong>
                <span class="tooltiptext" style="width: 450px;">Calculated as: 
                <span class="fraction">
                <span class="numerator">1 - Support ({consequent_str})</span>
                <span class="denominator">1 - Confidence ({antecedent_str} → {consequent_str})</span>
                </span></span></span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.dataframe(transactions_encoded_disp, hide_index=True)


with tab_filter:
    st.header("Filter Functions and Custom Rules")
    # """ """     st.markdown("""
    #     **Interactive Filtering**  
    #     This section allows you to filter association rules based on specific criteria:
    #     - Select an item from the dropdown to display rules where that item is in the antecedents.
    #     - Use the slider to adjust the minimum confidence threshold for the rules.
        
    #     These filters help you focus on the most relevant rules for your business analysis.
    #     """)
    #     #read all items in the antecedents column
    #     all_items = set()
    #     for items in association_rules_df['antecedents']:
    #         all_items.update(items)
    #     all_items = list(all_items)
        
    #     #create a selectbox for the items
    #     selected_item = st.selectbox('Select an item:', all_items)
        
    #     # Filter the association rules to only show rules where the antecedent is 'Bread'
    #     bread_df = association_rules_df[association_rules_df['antecedents'].apply(lambda x: selected_item in x)]

    #     # Display the association rules where the antecedent is 'Bread'
    #     st.write('Display the association rules where the antecedent is '+selected_item)
    #     st.write(bread_df)

        
    #     #create a slide for the threshold value
    #     threshold = st.slider('Select the minimum threshold:', 0.0, 1.0, 0.5, 0.1)
    #     st.write("Selected confidence threshold:", threshold)
        
    #     # Filter the association rules to only show rules where the minimum threshold = 0.5
    #     min_threshold_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=threshold)

    #     # Display the association rules where the minimum threshold = 0.5
    #     st.write('Association Rules with Minimum Confidence = '+str(threshold))
    #     st.write(min_threshold_rules) """ """

with tab_references:
    st.header("Further Reading")

    st.markdown("""
        - [https://www.geeksforgeeks.org/market-basket-analysis-in-data-mining/](https://www.geeksforgeeks.org/market-basket-analysis-in-data-mining/)
    """)


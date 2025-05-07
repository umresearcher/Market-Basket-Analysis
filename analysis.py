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
    #st.write(transactions_df)
    #st.write(transactions_df.columns)
    
#put a download button
st.sidebar.markdown('Download the template') 
st.sidebar.markdown('[The template](https://github.com/umresearcher/Market-Basket-Analysis/blob/main/my_transactions.csv)')#update the link to the dataset
      
#add title to the app
st.title('Market Basket Analysis')

#create pages tabs
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
    We have provided a default dataset that has five transactions. You may choose to upload your own transactions and items 
    (as a .csv file) instead of the default provided. The template provided shows how your data should be formatted: there must 
    be a header row with column headings as Transaction_ID, and Items. The items in a 
    transaction are comma separated. Each transaction must have 1 or more items. Different transactions can have different number 
    of items. As part of our processing, we trim any spaces around an item. 
    """)
    #display the dataset
    st.write('These are the transactions and the items in each transaction from your data.')
    #st.write(transactions_df)
    st.dataframe(transactions_df, hide_index=True)

with tab_encoded:
    st.header("The Encoded Dataset")
    st.markdown("""
    Let us view the dataset as a Boolean (True/False) matrix. Each row represents a transaction, and each column represents an item. 
    A value of True indicates the item was purchased in that transaction.
    
    This section displays the encoded dataset. If you see any errors in the encoded dataset, you may want to go back and check the 
    csv file has the right values and follows the template.

    """)
    #conver column of transactions to list -- trim whitespaces
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

    # Display the encoded transaction dataset
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
        return ['background-color: lightgreen' if row.name in highlighted_transactions.index else '' for _ in row]

    # Display the encoded transaction dataset without the index
    st.dataframe(transactions_encoded.style.apply(highlight_rows, axis=1), hide_index=True)

    # Calculate support for the selected items
    support = len(highlighted_transactions) / len(transactions_encoded)
    num_transactions_containing_itemset = len(highlighted_transactions)
    total_transactions = len(transactions_encoded)

    # Highlight transactions
    # Define the common HTML structure
    html_structure = """
    <div style="font-family: Arial, sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <h2>Support for Selected Itemset</h2>
        <p>{content}</p>
    </div>
    """

    # Highlight transactions
    if not selected_items:
        content = "<strong>No</strong> items selected."
    else:
        content = f"""
        The support for the selected itemset is: 
        <span class="tooltip"><strong>{support:.2f}</strong>
        <span class="tooltiptext" style="width: 450px;">Calculated as: 
        <span class="fraction">
        <span class="numerator">Number of Transactions containing the Itemset ({num_transactions_containing_itemset})</span>
        <span class="denominator">Total number of Transactions ({total_transactions})</span>
        </span></span></span>
        """

    # Display the HTML with the specific content
    st.markdown(html_structure.format(content=content), unsafe_allow_html=True)

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
    min_support = st.slider('Set the minimum support threshold (&gt;= 0.01):', min_value=0.01, max_value=1.0, value=0.4, step=0.01)

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

with tab_associa:
    st.header("Association Rules")
    st.markdown("""
    An association rule is typically expressed in the form of **"X → Y"**. It can also be expressed in the form of 
    "<strong>If-Then</strong>" statements as If **X**, Then **Y**. Here the presence of one set of items (the 
    <strong>antecedent</strong> itemset, **X**) implies the presence of another set of items (the <strong>consequent</strong> 
    itemset, **Y**) with a certain level of <strong>confidence</strong> and <strong>support</strong>. We
    will examine confidence, support, and other metrics for association rules in the next section.


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

    # Select examples -- assumption: confidence < 1 is helpful for understanding
    single_item_rules = rules[(rules['antecedents'].apply(lambda x: len(x) == 1)) & 
                              (rules['consequents'].apply(lambda x: len(x) == 1)) & 
                         (rules['confidence'] < 1)].sort_values(by=['support', 'confidence'], ascending=False)


    # Check if there is at least one rule with confidence < 1. If not pick any rule with any confidence
    if single_item_rules.empty:
        single_item_rules = rules[(rules['antecedents'].apply(lambda x: len(x) == 1)) & 
                                  (rules['consequents'].apply(lambda x: len(x) == 1))].sort_values(by=['support', 'confidence'], ascending=False)

    # Check if there is at least one rule with any confidence
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
                         (rules['consequents'].apply(lambda x: len(x) > 1)) & 
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

    st.markdown("""
    Often, when analyzing transaction data, we are interested in identifying association rules that meet certain criteria. 
    Specifically, we look for rules whose support exceeds a specified threshold, indicating that the rule is applicable to a 
    significant portion of the transactions. Additionally, we seek rules with a confidence level above a certain threshold, 
    ensuring that the rule is reliable. Sometimes, we may also be interested in rules that contain specific items in the 
    antecedent (the "if" part) or the consequent (the "then" part). By applying these thresholds to our set of transactions, 
    we can filter and identify the most relevant association rules.
    """)


    # Selection widgets for filters
    support_threshold = st.slider('Set the minimum support threshold for rules (&gt;= 0.01):', min_value=0.01, max_value=1.0, value=1/len(transactions_encoded), step=0.01, key="support_threshold")
    confidence_threshold = st.slider('Set the minimum confidence threshold for rules:', min_value=0.0, max_value=1.0, value=1/len(transactions_encoded), step=0.01, key="confidence_threshold")
    antecedent_filter = st.multiselect("Filter Antecedent Items", items, key="antecedent_filter")
    consequent_filter = st.multiselect("Filter Consequent Items", items, key="consequent_filter")

    frequent_itemsets = apriori(transactions_encoded, min_support=support_threshold, use_colnames=True)

    # Calculate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_threshold)

    # Apply filters to rules
    if antecedent_filter:
        rules = rules[rules['antecedents'].apply(lambda x: all(item in x for item in antecedent_filter))]
    if consequent_filter:
        rules = rules[rules['consequents'].apply(lambda x: all(item in x for item in consequent_filter))]
    rules = rules[rules['support'] >= support_threshold]

    # Check if there are any rules after filtering
    if rules.empty:
        st.warning('No association rules found with the specified filter conditions.')
    else:
        # Sort rules by confidence in descending order
        # rules = rules.sort_values(by='confidence', ascending=False)
        rules = rules.sort_values(by=['support', 'confidence'], ascending=[False, False])


        # Form rules_disp for display purposes
        rules_disp = rules.copy()
        rules_disp['antecedents'] = rules_disp['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_disp['consequents'] = rules_disp['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_disp = rules_disp[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']]
        # Apply custom CSS class to support and confidence columns
        rules_disp['support'] = rules_disp['support'].apply(lambda x: f'<div class="left-align">{x:.2f}</div>')
        rules_disp['confidence'] = rules_disp['confidence'].apply(lambda x: f'<div class="left-align">{x:.2f}</div>')
        rules_disp['lift'] = rules_disp['lift'].apply(lambda x: f'<div class="left-align">{x:.2f}</div>')
        rules_disp['leverage'] = rules_disp['leverage'].apply(lambda x: f'<div class="left-align">{x:.2f}</div>')
        rules_disp['conviction'] = rules_disp['conviction'].apply(lambda x: f'<div class="left-align">{x:.2f}</div>')

        # Display rules in a table
        st.markdown(rules_disp.to_html(escape=False, index=False), unsafe_allow_html=True)


with tab_references:
    st.header("Further Reading")

    st.markdown("""
        - An article explaining Market Basket Analysis [https://www.geeksforgeeks.org/market-basket-analysis-in-data-mining/](https://www.geeksforgeeks.org/market-basket-analysis-in-data-mining/)
        - Coding with Python libraries for market basket analysis, and apriori algorithm [https://www.kaggle.com/code/burakbuyukyagmur/association-rules-with-apriori](https://www.kaggle.com/code/burakbuyukyagmur/association-rules-with-apriori)
    """)


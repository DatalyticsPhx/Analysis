import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analysis")

@st.cache()
def load_df():
    df = pd.read_csv('https://drive.google.com/uc?id=1WRRikGZ_DCgioxm1NGORp7xa4itHRVcc&confirm=t&uuid=50742ec4-25a5-4196-b54e-38ea28882b71&at=AHV7M3cQOatyBQ8YtVlVBcNOSBX2:1670427300799')
    return df

@st.cache()
def remove_outlier(df, mar,ed):
    mar_condition = df['mar'] == mar
    ed_condition = df['education'] == ed
    ret_df = df[mar_condition & ed_condition]
    return ret_df

@st.cache()
# Pipeline 
def pipeline():
    # default conditions
    df = load_df()
    condition1 = df['age'] == 35.0
    condition2 = df['sex'] == 1.0
    condition3 = df['st'] == 36.0
    pipe_df = df[condition1 & condition2 & condition3]
    
    # Remove outliers
    Q1 = pipe_df['spm_resources'].quantile(0.25)
    Q1
    # # # Calculate Q3
    Q3 = pipe_df['spm_resources'].quantile(0.75)
    Q3
    # # # Define the Inter Quartile Range (IQR)
    IQR = Q3 - Q1
    IQR
    # # # Make select condition for the values that fall below the Q1 - 1.5*IQR
    outliers_below = pipe_df['spm_resources'] < (Q1 - 1.5 * IQR)
    # # # Make select condition for the values that fall above the Q3 - 1.5*IQR
    outliers_above = pipe_df['spm_resources'] > (Q3 + 1.5 * IQR)
    # outliers_above
    # # # Select the INVERSE of the selection
    to_plot = pipe_df[ ~(outliers_above | outliers_below) ]
    return to_plot

@st.cache
def get_perc(x):
    df = load_df()
    condition = df['race'] == x
    temp = pd.DataFrame(df[condition])
    count = temp[temp.columns[0]].count()
    df_count = df[df.columns[0]].count()
    return (count/df_count) * 100

header = st.container()
personas = st.container()

with header:
    st.title("Data Visualization")
    st.markdown("## Disclaimer: White as more representation in our data set!")
    temp_count = {'Race': ['White', 'Black', 'Asian', 'Other'], 'Percentage': [get_perc(1.0), get_perc(2.0), get_perc(3.0), get_perc(4.0)]}
    temp_count_df = pd.DataFrame(data=temp_count)
    fig = plt.figure(figsize=(13,8))
    sns.barplot(data=temp_count_df, x='Race', y='Percentage')
    st.pyplot(fig)
    # maybe some text here

with personas:
    st.title("Personas")
    clean_df = pipeline()
    st.text("")
    st.markdown("### Persona #1")
    persona1, exp1 = st.columns([3,1])
    st.text("")
    st.markdown("### Persona #2")
    persona2, exp2 = st.columns([3,1])
    st.text("")
    st.markdown("### Persona #3")
    persona3, exp3 = st.columns([3,1])
    st.text("")
    st.markdown("### Persona #4")
    persona4, exp4 = st.columns([3,1])
    #persona1.pyplot(pipeline())

    # Persona number 1
    to_plot = remove_outlier(clean_df, 1.0, 4.0)
    a4_dims = (13,8)
    fig, ax = plt.subplots(figsize=a4_dims)
    temp = sns.boxplot(data=to_plot, x=to_plot['race'], y=to_plot['spm_resources'], showfliers = False)
    temp.set_xticks(range(4))
    temp.set_xticklabels(['White', 'Black', 'Asian', 'Other'])
    plt.title("35 years old, Married, Male with a College Education from New York")
    plt.xlabel("Race")
    plt.ylabel("Income")
    medians = to_plot.groupby(['race'])['spm_resources'].median().values
    vertical_offset = to_plot['spm_resources'].median() * 0.05

    for xtick in temp.get_xticks():
        temp.text(xtick,medians[xtick]+ vertical_offset,medians[xtick].astype(int),
                    horizontalalignment='center',size='x-large',color='black',weight='semibold')
    persona1.pyplot(fig)
    exp1.text("In persona 1...")

    # Persona number 2
    to_plot = remove_outlier(clean_df, 5.0, 4.0)
    a4_dims = (13,8)
    fig, ax = plt.subplots(figsize=a4_dims)
    temp = sns.boxplot(data=to_plot, x=to_plot['race'], y=to_plot['spm_resources'], showfliers = False)
    temp.set_xticks(range(4))
    temp.set_xticklabels(['White', 'Black', 'Asian', 'Other'])
    plt.title("35 years old, Single, Male with a College Degree from New York")
    plt.xlabel("Race")
    plt.ylabel("Income")
    medians = to_plot.groupby(['race'])['spm_resources'].median().values
    vertical_offset = to_plot['spm_resources'].median() * 0.05

    for xtick in temp.get_xticks():
        temp.text(xtick,medians[xtick]+ vertical_offset,medians[xtick].astype(int),
                    horizontalalignment='center',size='x-large',color='black',weight='semibold')
    persona2.pyplot(fig)
    exp2.text("In persona 2...")

    # Persona number 3
    to_plot = remove_outlier(clean_df, 1.0, 3.0)
    a4_dims = (13,8)
    fig, ax = plt.subplots(figsize=a4_dims)
    temp = sns.boxplot(data=to_plot, x=to_plot['race'], y=to_plot['spm_resources'], showfliers = False)
    temp.set_xticks(range(4))
    temp.set_xticklabels(['White', 'Black', 'Asian', 'Other'])
    plt.title("35 years old, Married, Male with some College Education from New York")
    plt.xlabel("Race")
    plt.ylabel("Income")
    medians = to_plot.groupby(['race'])['spm_resources'].median().values
    vertical_offset = to_plot['spm_resources'].median() * 0.05

    for xtick in temp.get_xticks():
        temp.text(xtick,medians[xtick]+ vertical_offset,medians[xtick].astype(int),
                    horizontalalignment='center',size='x-large',color='black',weight='semibold')
    persona3.pyplot(fig)
    exp3.text("In persona 3...")

    # Persona number 4
    to_plot = remove_outlier(clean_df, 5.0, 3.0)
    a4_dims = (13,8)
    fig, ax = plt.subplots(figsize=a4_dims)
    temp = sns.boxplot(data=to_plot, x=to_plot['race'], y=to_plot['spm_resources'], showfliers = False)
    temp.set_xticks(range(4))
    temp.set_xticklabels(['White', 'Black', 'Asian', 'Other'])
    plt.title("35 years old, Married, Male with some College Education from New York")
    plt.xlabel("Race")
    plt.ylabel("Income")
    medians = to_plot.groupby(['race'])['spm_resources'].median().values
    vertical_offset = to_plot['spm_resources'].median() * 0.05

    for xtick in temp.get_xticks():
        temp.text(xtick,medians[xtick]+ vertical_offset,medians[xtick].astype(int),
                    horizontalalignment='center',size='x-large',color='black',weight='semibold')
    persona4.pyplot(fig)
    exp4.text("In persona 4...")
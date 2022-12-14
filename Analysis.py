import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title="Analysis")

@st.cache()
def load_df():
    df = pd.read_csv('https://drive.google.com/uc?id=1WRRikGZ_DCgioxm1NGORp7xa4itHRVcc&confirm=t&uuid=50742ec4-25a5-4196-b54e-38ea28882b71&at=AHV7M3cQOatyBQ8YtVlVBcNOSBX2:1670427300799')
    return df

# @st.cache()
def boxplot_race(df, race, edu, mar):
    c1 = df['race'] == race
    c2 = df['education'] == edu
    c3 = df['mar'] == mar

    Q1 = df[c1&c2&c3]['spm_resources'].quantile(0.25)
    # Q1

    # # # Calculate Q3
    Q3 = df[c1&c2&c3]['spm_resources'].quantile(0.75)
    # Q3
    # # # Define the Inter Quartile Range (IQR)
    IQR = Q3 - Q1
    # IQR
    # # # Make select condition for the values that fall below the Q1 - 1.5*IQR
    outliers_below = df[c1&c2&c3]['spm_resources'] < (Q1 - 1.5 * IQR)

    # # # Make select condition for the values that fall above the Q3 - 1.5*IQR
    outliers_above = df[c1&c2&c3]['spm_resources'] > (Q3 + 1.5 * IQR)
    # outliers_above
    # # # Select the INVERSE of the selection
    df = df[c1&c2&c3][ ~(outliers_above | outliers_below) ]
    df1 = df.groupby("race")["spm_resources"].agg(Median='median')
    return df

header = st.container()
personas = st.container()

with header:
    st.title("Data Visualization")
    df = load_df()

    data1 = boxplot_race(df, 1.0, 4.0, 1.0)
    data1["location"] = 1

    data2 = boxplot_race(df, 2.0, 3.0, 5.0)
    data2["location"] = 2

    data3 = boxplot_race(df, 3.0, 4.0, 1.0)
    data3["location"] = 3

    data4 = boxplot_race(df, 4.0, 3.0, 5.0)
    data4["location"] = 4

    cdf = pd.concat([data1, data2, data3, data4])    

    a4_dims = (13,8)
    fig, ax = plt.subplots(figsize=a4_dims)
    temp = sns.boxplot(x="location", y="spm_resources", data=cdf, showfliers = False)    
    temp.set_xticks(range(4))
    temp.set_xticklabels(['White', 'Black', 'Asian', 'Other'])
    plt.xlabel("Race")
    plt.ylabel("Income")
    medians = cdf.groupby(['location'])['spm_resources'].median().values
    vertical_offset = cdf['spm_resources'].median() * 0.05

    for xtick in temp.get_xticks():
        temp.text(xtick,medians[xtick]+ vertical_offset,medians[xtick].astype(int),
                    horizontalalignment='center',size='x-large',color='black',weight='semibold')
    st.pyplot(fig)
    st.markdown("## This boxplot represents the range in salaries for the most common demographic occurances for each racial group.")
    st.text("The blue box plot is a White male aged 35 with a college degree who is married.")
    st.text("The orange box plot is a Black male aged 35 with some college education and never \nmarried.")
    st.text("The green box plot is an Asian male aged 35 with a college degree who is married.")
    st.text("The red box plot is a male who is categorized as \"Other\". He is 35 years old with \nsome college education who is single.")
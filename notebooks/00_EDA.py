# %% [markdown]
# 0.0 Imports 
# %%

import pandas as pd 
from pyspark.sql import SparkSession 
from tqdm import tqdm 
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt  

# %% [markdown]
    # # 0.0 Reading Dataset     
spark = (
    SparkSession
        .builder
        .appName("localCreditModelling")
        .master("local[*]")
        #.config('spark.driver.host', '127.0.0.1')
        .getOrCreate()
)

print("Spark Session created successfully.")



# %%
# Leitura do dataset 

df_application = (
    spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/application_train.csv")
)

df_bureau = (
    spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/bureau.csv")
)

df_bureau_balance = (
    spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/bureau_balance.csv")
)

df_credit_card_balance = (
    spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/credit_card_balance.csv")
)

df_installments_payments = (
    spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/installments_payments.csv")
)

df_pos_cash_balance = (
    spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/POS_CASH_balance.csv")
)

df_previous_applications = (
    spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/previous_application.csv")
)

df_sample_submission = (
    spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/sample_submission.csv")
)

# %%

datasets = [df_application, df_bureau, df_bureau_balance, df_credit_card_balance, df_installments_payments, df_pos_cash_balance, df_previous_applications, df_sample_submission]
datasets_idx = ['application', 'bureau', 'bureau_balance', 'credit_card_balance', 'installments_payments', 'pos_cash_balance', 'previous_applications', 'sample_submission']


# %%
for idx, dataset in enumerate(datasets):
    print(f"Dataset: {datasets_idx[idx]}")
    print(f"Number of rows: {dataset.count()}")
    print(f"Number of columns: {len(dataset.columns)}")
    print(f"Schema:")
    dataset.printSchema()
    print("\n")
# %%
df_application.show(5)
# %%
df_dict = pd.read_csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/HomeCredit_columns_description.csv" , 
                      sep = ',',
                      encoding = 'latin-1')
df_dict.head(5)
df_dict[df_dict.Table == 'application_{train|test}.csv']

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# %% [markdown]
## 1.0 EDA 

# ### 1.1 - Análise de contexto 
# **Objetivo:** Compreender o contexto de negocio dos dados da home Credit Risk Default.  

# %% [markdown]
#### 1.1.1 - Applications dataset 

# %%

df_application.columns 

# %% 

def get_dataset_sample_set(
        df ,
        n_samples : int = 5 ) -> dict:
    '''
    returns a dictionary with the column names as keys and a list 
    of n_samples distinct values for each column as values.
    '''
    df.show(5)
    samples = {}
    print('Recovering samples for each column...')
    for col in tqdm(df.columns): 
        samples[col] = [row[col] for row in df.select(col).distinct().limit(n_samples).collect()] 

    return samples 

# %% 
samples = get_dataset_sample_set(df_application, 
                                 n_samples = 5)

# %%
samples

# %%
df_application.describe().show()



# %% 
df_application.select('TARGET').groupBy('TARGET').count().show()

# ------------------------------------------------
# -----------------------------------------------
# %% [markdown]
#### 1.1 - Identificando nulos
#  

# %%
#df_application_describe.limit(1).select(c).collect()[0][c]

# %%
def getNotNullColumns(df):
    '''
    Summarizes the percentage of non-null values for each column in a Spark DataFrame.
    Returns a Dataframe with columns 'tx_notNulls' sorted in descending order, 
    filtered to include only columns with less than 100% non-null values.
    '''
    samples = df.count()
    df_describe = df.describe()
    df_nulls = {}

    for c in tqdm(df.columns): 
        notNulls = df_describe.limit(1).select(c).collect()[0][c]
        tx_notNulls = round(int(notNulls)/samples,2)*100 
        df_nulls[c] = tx_notNulls


    df_nulls = (
                            pd.DataFrame(df_nulls, 
                                    index = ['tx_notNulls']).
                                    T.
                                    sort_values(by = 'tx_notNulls', 
                                                ascending = False)
        )
    df_nulls = df_nulls[df_nulls['tx_notNulls'] < 100]
    return df_nulls


# %%
applications_nulls = getNotNullColumns(df_application)

# %%

applications_nulls = applications_nulls[applications_nulls['tx_notNulls'] < 100]
print(len(applications_nulls))
applications_nulls

# %% []
# %% [markdown]
# Ha uma grande ocorrencia de features relacionadas a appartments/buildings com mais de 50% de dados nulos. 
# Irei buscar a interpretacao de que se esses nulos nao foram consequencia do cliente nao possuir um imovel, caso 
# se confirme, podemos considerar a imputacao desses nulos por 0, ou seja, 0 metros, 0 media de apartments etc. 
# 

# %%
df_application.select('FLAG_OWN_REALTY').groupBy('FLAG_OWN_REALTY').count().show()

# %%
df_application_not_realty = df_application.filter("FLAG_OWN_REALTY == 'Y'")
df_application_not_realty_nulls = getNotNullColumns(df_application_not_realty)
df_application_not_realty_nulls

# %%

len(df_application.columns)
# %%

df_application_bureau = (df_application
    .join(
        df_bureau, 
        on = 'SK_ID_CURR', 
        how = 'outer',
    )
)

print(f'before the join: {df_application.count()} | {df_bureau.count()}')
print(f'after the join: {df_application_bureau.count()}')

df_application_bureau.groupby('SK_ID_CURR').count().show(5)

# %% [markdown]

# ## 1.2 - Analise univariada 

# %% [markdown]
# ### 1.2.1 - Distribuicoes das variaveis numericas
# Caso a variavel seja numerica e possua mais de 5 valores distintos, irei considerar a variavel como continua e analisar a distribuicao de seus valores.
# Caso contrario sera classificada como categorica. 

# %%
import numpy as np 

application_dtypes = dict(df_application.dtypes)
distinct_dtypes = []
for col in application_dtypes.keys(): 
    distinct_dtypes.append(application_dtypes[col])

np.unique(distinct_dtypes)


# %%

def get_column_types(df : pd.DataFrame, 
                     threshold : int = 5) -> dict:
    '''
    returns a dict of two lists: 
        "numerical"    :  with the names of continuous columns 
        "categorical"  :  with the names of categorical columns.
    '''

    continuous_columns = []
    categorical_columns = []

    numerical_columns = [x for x in df.columns if dict(df.dtypes)[x] in ['int', 'double']]
    print('Parsing dataset thats can leave some minutes...')
    for col in tqdm(numerical_columns):
         
        n_distinct = df.select(col).distinct().count()

        if n_distinct > threshold:
            continuous_columns.append(col) 
        else:
            categorical_columns.append(col)

    
    return {
        'numerical' : numerical_columns, 
        'categorical' : categorical_columns
    }  

# %%

applications_column_types = get_column_types(df_application)


# %%

print(f"Numerical   : {len(applications_column_types['numerical'])}")
print(f"Categorical : {len(applications_column_types['categorical'])}")

# %%
df_application.columns
# %%

df_application_sample = df_application.sampleBy('TARGET', fractions={0:0.3, 1:1} )
df_application_sample.count()
df_application_sample.groupBy('TARGET').agg({'SK_ID_CURR' : 'count'}).show()

# %%
df_application.groupBy('TARGET').agg({'SK_ID_CURR' : 'count'}).show()

# %% [markdown]
##### Pltoting the distribution of the continuous variables. 

def show_numerical_distribution(df, numerical_columns):
    for col in numerical_columns:
        # 1. Define the figure size
        plt.figure(figsize=(18, 4))
        
        # 2. BOXPLOT: Position 1 of 2
        plt.subplot(1, 2, 1) # <--- Corrected this to be 1, 2, 1
        sns.boxplot(x=df[col])
        plt.title('Box Plot') # Optional: Add a title to the subplot
        
        # 3. HISTOGRAM: Position 2 of 2
        plt.subplot(1, 2, 2) # <--- Corrected this to be 1, 2, 2
        sns.histplot(x=df[col], kde=True)
        plt.title('Histogram') # Optional: Add a title to the subplot
        
        # 4. Main title and display
        plt.suptitle(f'Distribution of {col}', fontsize=16) # Using f-string for clarity
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent suptitle overlap
        plt.show()

show_numerical_distribution(
    df = df_application, 
    numerical_columns=applications_column_types['numerical']
)
# %%

# %%

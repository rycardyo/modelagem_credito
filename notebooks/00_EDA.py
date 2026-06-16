# %%

import pandas as pd 
from pyspark.sql import SparkSession 
from tqdm import tqdm 

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


# %%markdown

## 2.0 - Exploratory Data Analysis (EDA)
### 2.1 - Hipoteses: 
#
#    1. Emprestimos feitos por pf , com justificativa comercial, possui uma mario expecativa deertorno. 
#    2. Emprestimos feitos por clientes que atuam em certas areas do mercado, como ti possuem maior 
#expectativa de retorno.
#    3. Clientes pontuais tendem a ser pagadores mais confiaveis.
#    4. Clientes com maior percentual de renda disponivel (50%) possuem maior chances de se formar
#uma inadimplencia.
#    5. Clientes que buscaram muitos emprestimos recentemnmente possuem menor chacenes de sere lresatados
#    6. Clientes com emprestimos ate 30% de sua renda possuem 90% de confiança
#    7. Clientes solteiros tendem a horar mais com seus emprestimos do que casados. 
#    8. Clientes com filhos tem maior tendencia a ser inadimplentes.
#    9. Clientes com mais de 3 emprestimos tem maior tendencia a ser inad
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

# %%
df_application_
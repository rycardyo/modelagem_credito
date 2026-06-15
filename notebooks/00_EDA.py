# %%

import pandas as pd 
from pyspark.sql import SparkSession 

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

dataset_path ''

df = (
    spark
        .read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/p123/Documents/DS/datasets/home-credit-default-risk/installments_payments.csv")
)
# %%
df.count()
# %%
df.limit(5).show()
# %%
df.printSchema()
# %%

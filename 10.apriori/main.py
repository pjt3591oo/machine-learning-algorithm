import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

dataset = [
  ['사과','치즈','생수'],
  ['생수','호두','치즈','고등어'],
  ['수박','사과','생수'],
  ['생수','호두','치즈','옥수수']
]

# 인코딩
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)

# 데이터 프레임 변경
df = pd.DataFrame(te_ary, columns=te.columns_) 
print(df)

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

print(frequent_itemsets)
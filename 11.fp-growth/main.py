import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

data = [
  ['사과','치즈','생수'],
  ['생수','호두','치즈','고등어'],
  ['수박','사과','생수'],
  ['생수','호두','치즈','옥수수']
]

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)

print(frequent_itemsets)
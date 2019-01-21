#%%
import pandas as pd
import umap
import matplotlib.pyplot as plt
 
 #dataの前処理
train=pd.read_csv('input/train.csv')
datas=pd.read_csv('input/merchants.csv')
print(datas.ix[[0,5],['most_recent_sales_range','most_recent_purchases_range']])
rank_list=pd.read_csv('input/features/rank.csv')

print(datas.head())
for i in range(datas.shape[0]):
    print(datas.iat['category_3',0])
    datas.at[i,'most_recent_sales_range']=rank_list.iat[datas.iat['most_recent_sales_range'],number]

drop_columns=['merchant_id']
datas.drop(drop_columns,axis=1)


#print(datas.loc(0))
'''
reduction=umap.UMAP(n_neighbors=50,metric='euclidean',n_components=2)
result_fit=reduction.fit_transform(datas)

plt.scatter(result_fit[:,0],result_fit[:,1],c='b',edgecolors='k')
plt.show()
'''
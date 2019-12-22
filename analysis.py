#!/usr/bin/env python
# coding: utf-8

# # 导入模块

# In[448]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from IPython.display import display
plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['simhei','Arial']})


# # 数据规整

# In[542]:


lf_df = pd.read_csv('./lianjia.csv')
lf_df.head()


# In[450]:


# 获取所有数据列信息
lf_df.info()


# In[660]:


# 拷贝一份数据，不对原始数据 进行 修改
df = lf_df.copy()
df.head()


# In[661]:


# 计算 每一平米的价格
# 总价 / 总面积
df['PerPrice'] = df['Price'] / df['Size']
df.head()


# In[662]:


# 重新摆放列的位置
columns = ['Region','District','Garden','Layout','Floor','Year','Size',
           'Elevator','Direction','PerPrice','Price']
# 第一种方法
df.loc[:,columns]
# 第二种方法
df = pd.DataFrame(df,columns=columns)
display(df.head(2))


# # 数据分析
# ## 地区特征分析

# ##### 对二手房区域分组，对比各个地区二手房数量和每平米房价

# #### 各个地区二手房数量

# In[663]:


# count()统计数量，
# sort_values(ascending=True)升序排列
# to_frame()将series转换为dataframe
df_house_count = df.groupby('Region')['Size'].count().sort_values(ascending=True).to_frame()


# In[664]:


# 修改列名称
df_house_count = df_house_count.reset_index().rename({'Region':'地区','Size':'数量'},axis=1).set_index('地区')
df_house_count


# ### 各地区 每平米房价对比

# In[665]:


df_mean_house = df.groupby('Region')['PerPrice'].mean().to_frame().reset_index().rename({'Region':'地区','PerPrice':'每平米房价'},axis=1).set_index('地区')
df_mean_house


# ### 各地区房价和数量信息可视化

# In[666]:


fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(5,1,1)
sns.barplot(x = df_mean_house.index,
            y = '每平米房价',
            palette = 'Blues_d',
            data = df_mean_house,
            ax = ax1)
# 修改 x轴的字体大小
ax1.set_xlabel('地区',{'size':14})
# 修改 y轴的字体大小
ax1.set_ylabel('每平方米房价',{'size':14})
# 设置刻度字体大小
ax1.tick_params(labelsize=10)
# 设置子图的标题
_ = ax1.set_title('北京各地区二手房每平米单价对比',fontsize=14)
# 第二个子图
ax2 = fig.add_subplot(5,1,3)

sns.barplot(x=df_house_count.index,
            y=df_house_count['数量'],
            data = df_house_count,
            palette='Greens_d',
            ax=ax2,
           )
# 修改 x轴的字体大小
ax2.set_xlabel('地区',{'size':14})
# 修改 y轴的字体大小
ax2.set_ylabel('数量',{'size':14})
# 设置刻度字体大小
ax2.tick_params(labelsize=10)
# 设置子图的标题
_ = ax2.set_title('北京各地区二手房数量',fontsize=14)

# 第三个子图，箱型图
ax3 = fig.add_subplot(5,1,5)
sns.boxplot(x='Region',y='Price',data=df,ax=ax3)
_ = ax3.set_title("北京各大区二手房房屋总价",fontsize=14)
ax3.set_xlabel("区域")
_ = ax.set_ylabel("房屋总价")


# # 房屋面积特征分析

# In[667]:


fig = plt.figure(figsize=(15,9))
ax1 = fig.add_subplot(1,2,1)
# 画出直方图,查看房屋面积特征分布
sns.distplot(df['Size'],bins=100,ax=ax1,color='r',kde=True)
ax2 = fig.add_subplot(1,2,2)
# 房屋面积和价格的关系
sns.regplot(x='Size',y='Price',data=df,ax=ax2)


# ## size分布
# 通过 distplot绘制的柱状图观察size特征的分布情况，属于长尾类型的分布
# 所谓的长尾分布，就是说明有一些面积很大超出正差范围的二手房

# # Size和Price的关系
# 
# 通过 regplot绘制的线性图，发现Size特征基本于Price呈线性关系。
# 符合常识，面积越大，价格越高。
# 但是有两组明显的异常点：
# 1. 面积很小，不到10平米，但价格却贵的离谱
# 2. 有一个面积超过了1000平米，价格却特别低

# In[668]:


# 筛选出面积小于10平米所有的房屋
condition = df['Size'] < 10
df[condition]


# 所有面积小于10的数据全部都是别墅，数据出现异常的原因是由于别墅的结构
# 比较特殊，字段定义与二手商品放不太一样导致爬虫爬取数据错位。
# 将这些别墅房移除。

# In[669]:


# 筛选出面积大于1000的房屋
condition = df['Size']>1000
df[condition]


# 结果发现，房屋面积大于1000的不是普通的民用房屋
# 而是商业用房，这个也需要移除

# In[670]:


condition = (df['Layout'] != '叠拼别墅') & ( df['Size'] <1000)
df = df[condition]


# # 重新可视化后，就没异常数据了

# In[671]:


fig = plt.figure(figsize=(15,9))
ax1 = fig.add_subplot(1,2,1)
# 画出直方图,查看房屋面积特征分布
sns.distplot(df['Size'],bins=100,ax=ax1,color='r',kde=True)
ax2 = fig.add_subplot(1,2,2)
# 房屋面积和价格的关系
sns.regplot(x='Size',y='Price',data=df,ax=ax2)


# # 户型特征分析

# In[672]:


plt.figure(figsize=(15,15))
sns.countplot(y='Layout',data=df)
_ = plt.title('房屋户型',fontsize=15)
plt.xlabel('数量')
plt.ylabel('户型')
plt.show()


# In[673]:


# 找出户型特征最多的五个
df['Layout'].value_counts().sort_values(ascending=False).head()


# 户型特征的厅室搭配花样太多了，各种奇怪的结构，需要进行特征处理
# 
# 可以看出，最多的是二室一厅

# # 现在假设，两室一厅、三室两厅为精装房
# # 三室一厅、二室二厅为简装房
# # 一室0厅、二室0厅为毛胚房
# # 剩余的划分为 其它

# In[674]:


# 条件
map_dict = {
    '2室1厅':'精装房',
    '2室2厅':'简装房',
    '3室1厅':'简装房',
    '1室0厅':'毛胚房',
    '2室0厅':'毛胚房',
    '3室1厅':'精装房',
}
# 增加一列，作为户型类别
df.loc[:,'Renovation'] = df.Layout.map(map_dict)
df.loc[:,'Renovation']=df['Renovation'].fillna('其它')
df.head()


# # 对 Renovation 特征分析

# In[675]:


df['Renovation'].value_counts()


# In[676]:


fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(1,3,1)
# 统计各类房的数量
sns.countplot(df['Renovation'],ax=ax1)

# 柱状图
ax2 = fig.add_subplot(1,3,2)
# 使用的是平均值
sns.barplot(x='Renovation',y='Price',data=df,ax=ax2)
# 箱型图
ax3 = fig.add_subplot(1,3,3)
sns.boxplot(x='Renovation',y='Price',data=df,ax=ax3)


# # 电梯特征分析

# 之前的处理阶段，我们能发现，Elevator特征是存在着大量的缺失值
# 所以需要进行一些处理

# In[677]:


# 查看缺失值数量
condition = df['Elevator'].isnull()
len(df[condition])


# 常见的处理缺失值方法有：平均数、中位数填补/删除缺失值
# 
# 拉格朗日中值法等等，但是有无电梯并不是具体的数据。根本不存在平均数、中位数这一说
# 
# 可以换一种思路，当楼层大于等于6层就要有电梯，小于六层就没有
# 
# 

# In[678]:


df.head()


# In[679]:


def ModifyElevator(x):
    # 由于 np.nan为flaot对象
    if isinstance(x['Elevator'],float):
        if x.Floor >6:
            return '有电梯'
        else:
            return '无电梯'
    return x['Elevator']
df['Elevator']= df.apply(ModifyElevator,axis=1)


# # 电梯特征可视化

# In[680]:


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)
sns.countplot(x='Elevator',data=df,ax=ax1)
ax2 = fig.add_subplot(1,2,2)
sns.barplot(x='Elevator',y='Price',data=df,ax=ax2)


# # 年份特征分析

# In[681]:


grid = sns.FacetGrid(
            df,
            row='Elevator',
            col='Renovation',
            palette='seismic',aspect=2)
grid.map(plt.scatter,'Year','Price')
grid.add_legend()


# 整个二手房房价趋势是随着时间增长而增长的
# 
# 毛胚房的价格稳定
# 
# 简装房主要是在 1980年之后
# 
# 1980年之前几乎不存在电梯二手房数据，说明1980年之前还没有大面积安装电梯
# 
# 

# # 楼层特征分析

# In[682]:


fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(1,1,1)
sns.countplot(x='Floor',data=df,ax=ax1)
_ = ax1.set_xlabel('楼层',fontsize=20)
_ =ax1.set_ylabel('数量',fontsize=20)


# 可以看到，6层二手房数量最多，但是单独的楼层特征没有什么意义，因为每个小区
# 
# 住房的总楼层都一样,我们需要知道楼层的相对意义。
# 
# 此外，楼层与文化也紧密联系在一次，中国人喜欢6不喜欢4
# 
# 喜欢七层不喜欢八层，也不喜欢十八层
# 
# 这些复杂的条件结合起来，就使得楼层变成一个非常复杂的特征。
# 

# # 特征工程

# 特征工程包括的内容很多，有特征的清洗，预处理、监控等，而预处理根据单一特征
# 
# 或多特征又分很多种方法，如归一化、降维，特征选择，特征筛选等等。
# 
# 这么多方法，为的是什么呢？其目的就是让这些特征更友好的作为模型输入
# 
# 处理数据的好坏会严重影响模型的性能，而好的特征工程有的时候甚至比建模调参更重要
# 
# 之前我们处理一些数据，就是我们常说的特征工程，如下几个例子：
# 
# 
# 

# In[683]:


'''
    特征工程
'''

# 1. 移除结构类型异常值和房屋大小异常值
# condition = (df['Layout'] != '叠拼别墅') & ( df['Size'] <1000)
# df = df[condition]

# 2. 填补 Elevator缺失值
# def ModifyElevator(x):
#     # 由于 np.nan为flaot对象
#     if isinstance(x['Elevator'],float):
#         if x.Floor >6:
#             return '有电梯'
#         else:
#             return '无电梯'
#     return x['Elevator']
# df['Elevator']= df.apply(ModifyElevator,axis=1)

# 3. 只考虑 “室” 和 “厅”，将其它少数“房间” 和 “卫” 移除
# 3室1厅
df = df[(df['Layout']         .str.extract('^\d(.*?)\d.*?') == '室')[0]]
df.head()


# In[687]:


# 4. 提取 “室” 和 “厅”创建新特征
df.loc[:,'Layout_roomNum'] = df['Layout'].str.extract(r'(\d)室')[0]
df.loc[:,'Layout_hallNum'] = df['Layout'].str.extract(r'\d.*(\d).*')[0]
df.head()


# In[690]:


# 5. 根据已有特征创造新特征

df.loc[:,'Layout_total_num'] = df['Layout_roomNum'].astype(np.int16) +     df['Layout_hallNum'].astype(np.int16)
df.head()


# In[703]:


# 6. 按中位数对 “year” 特征进行分箱
df.loc[:,'Year'] = pd.qcut(df['Year'],q=8,precision=0)
df.head()


# In[710]:


# 7. 删除无用特征
df.drop(['Garden','PerPrice'],axis=1,inplace=True)
df.head()


# # end ....

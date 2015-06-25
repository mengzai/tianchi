# project
ali megagame




阿里天池移动推荐算法大赛

报告


姓名：王立立
学校：西安电子科技大学















1赛题简介
阿里巴巴移动电商平台的真实用户-商品行为数据为基础，同时提供移动时代特有的位置
而参赛队伍则需要通过大数据和算法构面向建移动电子商务的商品推荐模型
原始数据1：tianchi_mobile_recommend_train_item ：445624条
 
           2：tianchi_mobile_recommend_train_user  数据为1048576条
    
2：赛制流程与解决方案：
	合并数据问题:源文件user_train.csv,item_train.csv  数据量较大所以应用mysql进行数据操作；但是数据的合并一直是我们头疼的问题，由于数据的多样性使合并会出现丢数据的情况：
解决方案：后期由于更多接触python的模块发现pandas的功能甚是强大；
可以应用merge直接更改：仅仅一句语言解决大量烦恼
pd.merge(file1,file2,how='outer').to_csv("D:\\tian\\process\\sample.csv")


程序：
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
file1 = pd.read_csv("D:\\tian\\process\\user_trainnew.csv")
file2 = pd.read_csv("D:\\tian\\process\\item_train.csv")
pd.merge(file1,file2,how='outer').to_csv("D:\\tian\\process\\local.csv")

合并结果：为loaca.csv
 
	时间问题格式问题：比赛给定时间格式2014-11-11 29格式不对所以投机取巧的用python将时间改为标准时间  如上图已改正确时间
 程序:
__author__ = 'meng'
raw_file=open("D:\\tian\\process\\user_train.csv","r")
f1=open("D:\\tian\\process\\user_trainnew.csv","w")
for line in raw_file.readlines():
    line=line.replace('\n','-00-00\n')
    f1.writelines(line)
f1.close()
raw_file.close()

	回归函数构造准备工作：python连接数据库
1按时间进行区分构建训练集train.csv（time<2014-12-18），验证集test.csv (time>2014-12-18)；
__author__ = 'meng'
import csv
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

f1 = pd.read_csv("D:\\tian\\process\\local.csv")
f1[f1.loc[:,'time']>=  '2014-12-18 00-00-00'].to_csv("D:\\tian\\process\\test.csv")


2
对train.csv取样1/4得到sample.csv（sample.py）
随机抽取数据防止数据具有不准确性：利用pandas模块直接完成
程序：
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

f1 = pd.read_csv("D:\\tian\\process\\train.csv")
f1.iloc[1:250000,3:].to_csv("D:\\tian\\process\\sample.csv")
结果为sample.csv如下图：
 

3：
构造admit 找到Logit 回归匹配项
匹配sample.csv与test.csv中的uaer_id,item_id，将匹配结果admit加入sample.csv：
程序：
#-*- coding: UTF-8 -*-
"""
程序整体概念：
1:选取train每行的五分之一数据中user ,item
2：选取18号数据的user item
3：在train 中按行读取如果数据在18号中则置admit=1
输出含有admit的文件
"""
import _collections,_csv,array,string
from collections import defaultdict
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
#读取18号的user，item
result1 = defaultdict(set)
f1= open("D:\\tian\\1\\user_item_18.csv",'r')
for line in f1.readlines():
    line = line.strip()
    result1[line]=line

#选取数据
admit = open("D:\\tian\\1\\teat_user_item.csv",'r')
#admit1=pd.read_csv("D:\\tian\\testfile\\buy1.csv",'a')
f1=open("D:\\tian\\1\\catage\\u_i_admit10.csv",'a')
for line1 in admit.readlines():
    line1 = line1.strip()
    if line1 in result1:
       f1.writelines('10\n')
    else:
       f1.writelines('0\n')

print f1
f1.close()

#admit1.to_csv('D:\\tian\\testfile\\admit1_later.csv')
得到结果到admit1_later.csv中
4 
对sample.csv中的geohash，hebavior_type，time分别量化到geohash.csv，hebavior_type.csv，time.csv，每个文件都包 含admit!
程序分别为：
Geohash。Csv
#-*- coding: UTF-8 -*-
import _collections,_csv
from collections import defaultdict
import pandas as pd
import statsmodels.api as sm                         #�����Ҫ��װ�������⣬import����
import pylab as pl
import numpy as np
"""
根据地理位置进行划分将地理位置前
7个置为4
5个置为3
3个置为2
1个置为1
选取地理位置的两列：
item_geohash,和user_geohash
比较进行赋值
SUBSTR(item_geohash,1,7）表示item_geohash前7个字符串
SUBSTR(user_geohash,1,7）表示user_geohash前7个字符
将
"""
#pd取数据
f1=open("d://tian//1//hash_admit.csv","w")
hasher = pd.read_csv("D:\\tian\\1\\test_u_i_u_ig_c_time-go $.csv")
result1 = defaultdict(list)
f = open("D:\\tian\\1\\test_u_i_u_ig_c_time-go $.csv")
for line in f.readlines():
    line = line.strip()
    user_geohash,item_geohash= line.split(",",1)
    if user_geohash[0:3]==item_geohash[0:3]:
        result1='4'
        f1.writelines(result1+'\n')
    elif user_geohash[0:2]==item_geohash[0:2]:
        result1='3'
        f1.writelines(result1+'\n')
    elif user_geohash[0:1]==item_geohash[0:1]:
        result1='2'
        f1.writelines(result1+'\n')
    else:
        result1='1'
        f1.writelines(result1+'\n')
f.close()
#hasher.to_csv('d:\\tian\\1\\hasher_admit.csv')
Time.csv
__author__ = 'meng'
import csv
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
f1 = open("D:\\AppServ\\MySQL\\data\\tianchi\\datasource\\time.csv",'r')
f2 = open("D:\\AppServ\\MySQL\\data\\tianchi\\datasource\\time_result.csv",'w')
for line in f1.readlines():
    line = line.strip()
    #print(line)
    if cmp('2014-11-25', line)+1:
        result1 = '1'
        print(result1)
        f2.writelines(result1+'\n')
    elif cmp('2014-12-02', line)+1:
        result1 = '2'
        print(result1)
        f2.writelines(result1+'\n')
    elif cmp('2014-12-09', line)+1:
        result1='3'
        f2.writelines(result1+'\n')
    else:
        result1='4'
        f2.writelines(result1+'\n')
f2.close()
f1.close()
	构建回归函数
这样admit都构造好之后构建对每个种类的admit构建虚拟变量：
pandas提供了一系列分类变量的控制。我们可以用get_dummies来将”admit”一列虚拟化。
get_dummies为每个指定的列创建了新的带二分类预测变量的DataFrame，在本例中，behavior_admit,behavior _type,time_type有四个级别：1，2，3以及4（1代表最有声望），prestige作为分类变量更加合适。当调用get_dummies时，会产生四列的dataframe，每一列表示四个级别中的一个。
这样便产生逻辑回归logit
#-*- coding: UTF-8 -*-
import _collections,_csv
from collections import defaultdict
import pandas as pd
import statsmodels.api as sm                         
import pylab as pl
import numpy as np
train = pd.read_csv("D:\\tian\\1\\catage\\u_th_tadmit.csv")                   
train_cols1 = train.loc[:,['admit','behavior_type']]   
#print(train_cols1)
dummy_1 = pd.get_dummies(train['behavior_type'], prefix='behavior_type')          
#print (dummy_1.head())

cols_to_keep = ['admit']
data1 = train[cols_to_keep].join(dummy_1.ix[:, 'behavior_type_2':])    #admit,behavior_type_2,behavior_type_3,behavior_type_4
#print (data1.head())
data1['intercept'] = 1.0
#print (data1.head())
train_cols1 = data1.columns[1:]
print(train_cols1)
logit = sm.Logit(data1['admit'],data1[train_cols1])

result = logit.fit()
产生的logit结果如下：
 
	结果分析
将这些采样的样本放入logistic regression 的计算模型中得到浏览和购买对用户下个月行为的系数关系，这一步可以用现成的库去实现
用logit产生的系数对整个train进行回归预测，用这些系数关系来预测那些没有标记的行为会不会产生购买行为
combos = pd.read_csv("D:\\tian\\1\\train.csv")
combos_cols1 = combos.loc[:,['behavior_type']]
dummy_2 = pd.get_dummies(combos['behavior_type'], prefix='behavior_type')
#print (dummy_2.head())       #behavior_type_2,behavior_type_3,behavior_type_4
data2 = dummy_2.ix[:, 'behavior_type_2':]
data2['intercept'] = 1.0
combos['prediction'] = result.predict(data2)
#print(combos.head())
#combos.to_csv('D:\\tian\\1\\catage\\prediction1.csv')
结果放到pre1中
 


获得的prediction进行取舍数据得到完美的预测结果
predicts = defaultdict(set)
for term in combos.values:
    user, item, prediction = str(int(term[0])), str(int(term[1])), term[4]
if prediction > 0.46:
    predicts[user].add(item)

#可以通过调节POINT的大小来控制最后结果的个数，当然你也可以取分数topN
    predicts[user].add(item)

	检验结果正确性
我们本地也要自己完成在验证集合上的测试，需要对比算法预测出来的结果和验证集上的结果：
__author__ = 'meng'
__author__ = 'meng'
import _collections,_csv,array,string
from collections import defaultdict    #去掉key的开发模块
#初始化
train_num=0       #训练集中多少数据
test_num=0        #测试集中多少数据
result3= defaultdict(set)
result2 = defaultdict(set)
hit_num=0
#返回结果
result1 = defaultdict(set)
f = open("D:\\tian\\测试文件\\buy1.csv")
for line1 in f.readlines():       #按行读取
        line1 = line1.strip()                #去掉空格
        result3[line1]=line1
        uidd,bidd = line1.split(',',1)       #将两个数据区分 user.item
        result1[uidd]=uidd   #将将item放到user中
        result2[bidd]=bidd                  #将user返回结果 2
        train_num += 1                    #多少个train数据

f.close()
result4 = defaultdict(set)
f = open("D:\\tian\\测试文件\\test.csv")
for line in f.readlines():
    line = line.strip()
    result4[line]=line
    (uid, bid)= line.split(",",1)
    test_num+=1                             #测试数据个数
    if uid not in result1:                 #判断测试集user是否在train中
     continue
    else:
       if result4[line] in result3:
       # for i in bidd:                #对item进行逐个读取
       # if   bid in result2:               #如果测试集item数据在训练集的item中
              hit_num += 1              #[测试集和检测集集合]
     #         print ("line is",result4[line] )
#print ("reslt13",result3)
#print ("reslt1",result1)
#print ("reslt２",result2)
print ("heji  is", hit_num)
print ("total train  is", train_num)
print ("test_num  is", test_num)
precision = float(hit_num)/train_num
callrate = float(hit_num)/test_num
print ("precision is ",precision)
print ("call rate is ", callrate)
print ("F1 is ", 2*precision*callrate/(precision+callrate))


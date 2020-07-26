import pandas as pd
import numpy as np
import string
import csv

from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 把单词添加到words_dict字典里，并计数
def add_words(word, word_dict):
    for d in word:
        if d in words_dict:
            words_dict[d]+=1
        else:
            words_dict[d]=1

# 处理文件每行数据
def isLineEmpty(line):
    return len(str(line))<1

def process_line(line,words_dict,newData):
    if not isLineEmpty(line):
        newData.append(line)
    return newData

val_key_list=[]
# 按格式输出words_dict中的数据
def print_result(words_dict):   
    for key,val in words_dict.items():
        if len(key)>3 and val>2:       #过滤结果，只输出词频大于2以及单词长度大于3的单词
            val_key_list.append((val,key))
    val_key_list.sort(reverse=True)  #对val值进行逆排序
    print ("%-10s%-10s" %("word","count"))
    print ("_"*25)
    for val,key in val_key_list:
        print ("%-12s   %3d" %(key,val))

# # 主函数
words_dict={}
df = pd.read_csv('tag_result.csv')
print(df.count())
newData = []
df = df.replace(np.nan,'',regex=True)
for tagline in df['tags'][:]:
    if len(str(tagline)) >1:
        newData.append(tagline)

for word in newData:
    # print(word)
    if(word!='nan'):
        word = word.lower()
        words_list=word.split()
        word=[d.strip(string.punctuation)for d in words_list] #删除进过分割的单词的尾部的一些符号
        add_words(word,words_dict)   #调用add_words函数，把单词插入到words_dict字典中
# 
print ("the length of the dictionary:",len(words_dict))
print_result(words_dict)

csv_tag = open('tag_value.csv', "w")
writer_label = csv.writer(csv_tag)
writer_label.writerow(['word','count'])
i = 1
while i<len(val_key_list):
    tmp1 = [val_key_list[i][0], val_key_list[i][1]]
    writer_label.writerow(tmp1)
    i += 1
csv_tag.close()

# generate word cloud
wc = WordCloud(background_color="white", max_words=1000)
wc.generate_from_frequencies(words_dict)
 # show
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
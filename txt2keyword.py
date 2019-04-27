import os
import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy
from nltk.stem.lancaster import LancasterStemmer
import pickle
lancaster_stemmer = LancasterStemmer()

def load_stopword(filename):
    '''
    加载停止词
    :return:
    '''
    with open(filename,'r',encoding='utf-8') as f:
        stopword = [line.strip() for line in f]
    return stopword

def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract(contents, stopword):
    '''
    用正则表达式去除无用内容以及停止词，保留有用的内容
    :param contents: list
    :param stopword: list
    :return:
    '''
    extracted_contents = []
    for content in contents:
        extracted_words = []
        content = str(content).strip()
        re_h = re.compile('</?\w+[^>]*>')
        re_nbsp = re.compile('&nbsp;')
        content = re_h.sub('', content)
        content = re_nbsp.sub('', content)
        words = jieba.lcut(content)
        for w in words:
            #
            w = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", w)
            w_new = lancaster_stemmer.stem(w)

            if len(w_new)>= 5 and w_new not in stopword and is_number(w_new) == False:
                extracted_words.append(w_new)
        #if len(extracted_words) > 0:
        extracted_contents.append(extracted_words)
    return extracted_contents

def get_Content(HTML_list):
    html_all = []
    for each in HTML_list:
        each = each.strip()
        each  = each.strip('\n')
        html_all.append(each)
    str_list = ''
    for each in html_all:
        str_list = str_list+each
        
    end_sign = []
    content = []
    content_length = []
    start_idx = 0
    end_idx = 0 
    peroid_idx = 0
    for i in range(len(str_list)-1):
        st = str_list[i]
        st_next = str_list[i+1]
        if st == '>':
            start_idx = i
        if st == '<' and st_next == '/':
            end_idx = i
        if start_idx > end_idx and end_idx>0:
            if start_idx > end_idx + 1:
                content.append(str_list[period_idx+1:end_idx])
            else:
                content.append([])

            content_length.append(len(content[-1]))
            end_sign.append(str_list[end_idx:start_idx+1])
        period_idx = start_idx
    
    new_content = []
    for i in range(len(content)):
        if content_length[i]>0:
            if end_sign[i] not in['</script>','</style>']:
                new_content.append(content[i])
    return new_content

# load stopword
stopword = load_stopword('stopword.txt')
extract_out = ['showDate']
stopword.extend(extract_out)


# get article set
print('loading articles...')
article_list = os.listdir('txts/')
print('length of the article set:', len(article_list))


# convert to bag of word
print('converting to bag of words...')
sentence_all = []
for page in range(len(article_list)):
    #if page==83:
    #    continue
    HTML_list = []
    #contain_index_list = []
    if page >= 0:
        #pageid_list.append(page)
        with open("txts/"+str(article_list[page]),'r') as f:
            for line in f:
                HTML_list.append(line.strip())


        content = HTML_list
        extract_content = extract(content,stopword)
        sentence = []
        for each in extract_content:
            if len(each) <= 3:
                continue
            sentence.extend(each)
        sentence_all.append(sentence)
        #contain_Index_list.append(contain_index_list)
        print(page)


# build vocab
print('building vocab...')
vocab = []
for sentence in sentence_all:
    for w in sentence:
        vocab.append(w)
vocab = list(set(vocab))
print('vocab length: '+str(len(vocab)))


# converting to bag of words with weights
print('converting to bag of words with weights...')
def convert2vec_with_weights(content, vocab, kw_weights = 1.0, not_kw_weights = 1.0, tag_weights = 1.0):
    '''
    针对关键词与非关键词采用不同权重的One-hot编码转换成vector
    '''
    vec = []
    for each in content:
        each_vec = [0] * len(vocab)
        for w in each:
            each_vec[vocab.index(w)] += not_kw_weights
        vec.append(each_vec)
    return vec
# the final bag of word with weights
dataX = np.array(convert2vec_with_weights(sentence_all, vocab, 1, 1, 1))



X = scipy.sparse.csr_matrix(dataX)
#类调用  
transformer = TfidfTransformer()
# print( transformer)  
#将词频矩阵X统计成TF-IDF值  
tfidf = transformer.fit_transform(X)
#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重  
# print( tfidf.toarray())


print('keyword for each article')
num_to_show = 50
for j in range(len(sentence_all)):
    sort_array = np.argsort(tfidf.toarray()[j])
    keywords_list = [vocab[sort_array[-(i+1)]] for i in range(num_to_show)]
    print('\n')
    with open('./freq_pickle/'+article_list[j]+'.pickle','wb') as f:
        pickle.dump(keywords_list,f)
    print(article_list[j])
    print(keywords_list)
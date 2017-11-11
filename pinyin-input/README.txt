实验目标：
本程序根据基于字的二元模型和三元模型实现简单的汉语拼音（全拼）输入法。

文件（夹）说明：
/src/    存放实验的源代码
/src/corpus_analysis.py    从给定语料中提取N-gram统计信息。
/src/pinyin_analysis.py    利用模型将输入的拼音转化为短语或者句子。
/data/    存放实验需要用到的数据
/data/input.txt    输入文件，每行为拼音串，音之间用空格分割，不包含标点符号。
/data/output.txt    输出文件，根据输入文件给出的拼音串，转换后的汉字串，每句占一行，无标点，无空格。
/data/一二级汉字表.txt    给出全部一二级汉字，共6763个，gbk编码方式
/data/拼音汉字表.txt    拼音-汉字对照表，每个音为一行，汉字之间用空格分割，gbk编码方式
/corpus/    存放语料文件，如果需要提取语料（执行corpus_analysis.py），需要在其中加入相应语料
/tmp/    存放从语料中提取的N-gram统计信息
/tmp/words_count.pkl    单个字,unigram
/tmp/gram_2_count.pkl    两个字,bigram
/tmp/gram_3_count.pkl    三个字，trigram

使用方法：
进入/src/目录，
corpus_analysis.py的用法：
python corpus_analysis.py file_name num
其中file_name是想提取的语料文件名称（包括路径），num是想保存的编号
pinyin_analysis.py的用法：终止
python pinyin_analysis.py [options]，options可选内容如下：
-i 表示input_file，输入文件名称（包括路径），默认为../data/input.txt。
-o 表示output_file，输出文件名称（包括路径），默认为../data/output.txt。
-n 表示ngram，表示使用哪种模型，ngram=2为二元模型，ngram=3为三元模型，默认为2。
-s 表示是否调用shell，默认为False，否则如果出现-s即为True，此时将不再考虑输入文件和输出文件。
pinyin_analysis.py的示例:
(1) python pinyin_analysis.py -s -n 3 
表示用三元模型，不进行文件读写，只在shell中实时输入拼音并翻译，直到输入exit终止。
(2) python pinyin_analysis.py -i ../data/input.txt -o ../data/output.txt -n 2 
表示用二元模型，对../data/input.txt中每一行的拼音，将它转化成短语或句子，输出到../data/output.txt中。

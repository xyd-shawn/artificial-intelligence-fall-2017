ʵ��Ŀ�꣺
��������ݻ����ֵĶ�Ԫģ�ͺ���Ԫģ��ʵ�ּ򵥵ĺ���ƴ����ȫƴ�����뷨��

�ļ����У�˵����
/src/    ���ʵ���Դ����
/src/corpus_analysis.py    �Ӹ�����������ȡN-gramͳ����Ϣ��
/src/pinyin_analysis.py    ����ģ�ͽ������ƴ��ת��Ϊ������߾��ӡ�
/data/    ���ʵ����Ҫ�õ�������
/data/input.txt    �����ļ���ÿ��Ϊƴ��������֮���ÿո�ָ�����������š�
/data/output.txt    ����ļ������������ļ�������ƴ������ת����ĺ��ִ���ÿ��ռһ�У��ޱ�㣬�޿ո�
/data/һ�������ֱ�.txt    ����ȫ��һ�������֣���6763����gbk���뷽ʽ
/data/ƴ�����ֱ�.txt    ƴ��-���ֶ��ձ�ÿ����Ϊһ�У�����֮���ÿո�ָgbk���뷽ʽ
/corpus/    ��������ļ��������Ҫ��ȡ���ϣ�ִ��corpus_analysis.py������Ҫ�����м�����Ӧ����
/tmp/    ��Ŵ���������ȡ��N-gramͳ����Ϣ
/tmp/words_count.pkl    ������,unigram
/tmp/gram_2_count.pkl    ������,bigram
/tmp/gram_3_count.pkl    �����֣�trigram

ʹ�÷�����
����/src/Ŀ¼��
corpus_analysis.py���÷���
python corpus_analysis.py file_name num
����file_name������ȡ�������ļ����ƣ�����·������num���뱣��ı��
pinyin_analysis.py���÷�����ֹ
python pinyin_analysis.py [options]��options��ѡ�������£�
-i ��ʾinput_file�������ļ����ƣ�����·������Ĭ��Ϊ../data/input.txt��
-o ��ʾoutput_file������ļ����ƣ�����·������Ĭ��Ϊ../data/output.txt��
-n ��ʾngram����ʾʹ������ģ�ͣ�ngram=2Ϊ��Ԫģ�ͣ�ngram=3Ϊ��Ԫģ�ͣ�Ĭ��Ϊ2��
-s ��ʾ�Ƿ����shell��Ĭ��ΪFalse�������������-s��ΪTrue����ʱ�����ٿ��������ļ�������ļ���
pinyin_analysis.py��ʾ��:
(1) python pinyin_analysis.py -s -n 3 
��ʾ����Ԫģ�ͣ��������ļ���д��ֻ��shell��ʵʱ����ƴ�������룬ֱ������exit��ֹ��
(2) python pinyin_analysis.py -i ../data/input.txt -o ../data/output.txt -n 2 
��ʾ�ö�Ԫģ�ͣ���../data/input.txt��ÿһ�е�ƴ��������ת���ɶ������ӣ������../data/output.txt�С�

使用说明：
1. 实验使用的是keras库，底层基于tensorflow
2. 运行程序前，需要创建四个文件夹：src/、data/、result/和tmp/，
   将models.py移至src/下，训练和测试数据train.csv和test.csv移至data/下
   result/用于存放预测结果，tmp/用来保存中间变量以及tensorboard
3. 移至src/目录下，models.py此时使用的是实验最佳结果对应CNN模型的网络和参数设置，
   如要更换模型，需要在models.py中切换，保存的文件名也使用的是测试时使用的版本，
   可以在models.py文件中进行更改。
4. 程序运行，在src/目录下，执行python models.py的操作。

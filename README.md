# examples/Chewie_CO_2016文件夹中：

AE_model是Chewie_CO_2016数据集source session训练的代码

Stage2_MMD是Chewie_CO_2016数据集target session的cross-session代码

to_Mihili_CO_2014/Stage2_MMD是以Mihili_CO_2014数据集作为target session的cross-subject代码


# data文件夹中：

解压文件unzip_this_file.zip,其中包含Chewie_CO_2016和Chewie_CO_2016的两个数据集用到的示例数据

数据格式：

data['spike'] [number of trials, trial_length, number of  channels] 数据的spike信息

data['behavior'] [number of trials, trial_length, number of  channels] 数据的行为信息(position)

data['label'] [number of trials,] 行为标签，数值是[0,1,2,...7]中的一个

# TCLA文件夹中：

data是数据加载方式

network是用到的网络

losses是损失函数

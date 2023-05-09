## 快速建立图片特征数据库并用文本描述进行搜索
>基于CLIP官方项目改编：https://github.com/openai/CLIP  
>采用国内配套中文模型：https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese  
>首次运行代码时会自动下载模型文件到本地clip(890M)和huggingface(1.4G)文件夹中(共约2.3G)  
>GPU运行时显存占用约2500M  
### CLIP介绍
>CLIP是2021年openai发布的基于对比学习的多模态模型(pytorch)，用于将文字描述和图片匹配  
>由一个图片编码模型和一个文本编码模型组成，一张图片经过图片模型得到的特征向量和这张图片的描述经过文本模型得到的特征向量会相近  
>通过计算余弦相似度和设定阈值可以通过文本找图片、通过图片找文本  
>原CLIP官方文本模型只支持英文，国内有人训练了中文的文本模型，但只支持ViT-L/14型号(890M)  
### 项目介绍
>本项目将建立图片特征数据库，然后用文本描述(75字以内，长的会被截断)去搜索符合描述的图片  
### 1，database_prepare.py
>将数据库图片放入文件夹image_database中  
>运行database_prepare.py即可生成特征数据库feature_database.csv  
### 2，predict.py
>在english_text、chinese_text中输入英文、中文文本，运行程序后可以搜到数据库中符合文本描述的图片  
>args中english_score_threshold、chinese_score_threshold为匹配的相似度筛选阈值，0.17、0.12为基准，可适当调整  
### 其他
>github链接：https://github.com/TWK2022/clip  
>学习笔记：https://github.com/TWK2022/notebook  
>邮箱：1024565378@qq.com  
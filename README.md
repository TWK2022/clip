## 快速建立图片特征数据库并用文本描述进行搜索
>基于CLIP官方项目整理：https://github.com/openai/CLIP  
>国内配套中文文本模型：https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese  
>首次运行代码时会自动下载模型文件到本地用户下(2.2G)  ：clip(890M)、中文文本模型(1.3G)  
>GPU运行时显存占用约2500M  
### CLIP介绍
>CLIP是2021年openai发布的基于对比学习的多模态模型(pytorch)，用于将文字描述和图片匹配  
>由一个图片编码模型和一个文本编码模型组成，一张图片经过图片模型得到的特征向量和这张图片的描述经过文本模型得到的特征向量会相近  
>通过计算余弦相似度可以通过文本找图片、通过图片找文本  
>原CLIP官方文本模型只支持英文，国内有人训练了中文的文本模型，但只支持ViT-L/14型号(890M)  
### 项目介绍
>本项目将建立图片特征数据库，然后用文本描述(75字以内，长的会被截断)去搜索符合描述的图片  
### 1，环境：linux
>torch：https://pytorch.org/get-started/previous-versions/
>```
>pip install ftfy regex tqdm transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
>pip install git+https://github.com/openai/CLIP.git
>```
### *，单独下载clip模型(890M)
>https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
### *，单独下载中文文本模型(1.3G)
>```
>sudo apt-get install git-lfs：安装git-lfs
>git lfs install：启用lfs。不使用lfs无法下载大文件
>git clone https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese：下载中文文本模型
>```
### 2，database_prepare.py
>将数据库图片放入文件夹image_database中  
>运行database_prepare.py即可生成特征数据库feature_database.csv  
### 3，predict.py
>在text中输入文本，运行程序后可以搜到数据库中符合文本描述的图片  
### 4，gradio_start.py
>用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
### 5，flask_start.py
>用flask将程序包装成一个服务，并在服务器上启动
### 6，flask_request.py
>以post请求传输数据调用服务
### 7，gunicorn_config.py
>用gunicorn多进程启动flask服务：gunicorn -c gunicorn_config.py flask_start:app
### 8，run.py
>微调模型，即使几张图片+描述也可以微调，在能达到满意效果的情况下训练次数越少越好
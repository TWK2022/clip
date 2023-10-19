# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
import gradio
import argparse
from predict import clip_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--database_path', default='feature_database.csv', type=str, help='|特征数据库位置|')
parser.add_argument('--model_name', default='ViT-L/14', type=str, help='|模型名称，中文文本模型只支持ViT-L/14(890M)|')
parser.add_argument('--chinese_cache', default='/root/.cache/huggingface/hub', type=str, help='|中文文本模型缓存/下载位置|')
parser.add_argument('--device', default='cuda', type=str, help='|运行设备|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def function(text, image):
    return text, image


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = clip_class(args)
    gradio_app = gradio.Interface(fn=model.predict, inputs=['text', 'image'], outputs=['text', 'image'])
    gradio_app.launch(share=False)

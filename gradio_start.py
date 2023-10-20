# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
# gradio_app=gradio.Interface(self,fn,inputs=None,outputs=None,examples=None)：配置。fn为传入inputs后执行的函数；inputs为输入的参数类型，单个参数直接传入，多个参数用列表对应传入，outputs为输出显示的类型，'text'为传入/显示字符串，'image'为传入/显示图片(RGB)
# gradio_app.launch(share=False)：启动界面，启动后默认可在http://127.0.0.1:7860访问。share=False时只能在本地访问，True时可在外部访问，但只有24小时的免费，超过的要在gradio官方购买云服务
import cv2
import gradio
import argparse
from predict import clip_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--database_path', default='feature_database.csv', type=str, help='|特征数据库位置|')
parser.add_argument('--model_name', default='ViT-L/14', type=str, help='|模型名称，中文文本模型只支持ViT-L/14(890M)|')
parser.add_argument('--chinese_model', default='clip_chinese_model', type=str,
                    help='|IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese或模型下载位置|')
parser.add_argument('--device', default='cpu', type=str, help='|运行设备|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def function(text, use_chinese=True):
    use_chinese = True if use_chinese == 'True' else False
    column, score = model.predict(text, use_chinese)
    image = cv2.imread(f'image_database/{column[0]}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return score[0], image


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = clip_class(args)
    gradio_app = gradio.Interface(fn=function, inputs=['text', 'text'], outputs=['text', 'image'],
                                  examples=[['一只白色的狗', 'True']])
    gradio_app.launch(share=False)

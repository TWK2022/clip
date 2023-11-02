# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
import cv2
import gradio
import argparse
from predict import clip_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--database_path', default='feature_database.csv', type=str, help='|特征数据库位置|')
parser.add_argument('--model_path', default='ViT-L/14', type=str, help='|模型名称或模型位置，中文文本模型只支持ViT-L/14(890M)|')
parser.add_argument('--chinese_model', default='IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese', type=str,
                    help='|模型名称或模型位置|')
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

# pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用flask将程序包装成一个服务，并在服务器上启动
import json
import flask
import argparse
from predict import clip_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动flask服务|')
parser.add_argument('--database_path', default='feature_database.csv', type=str, help='|特征数据库位置|')
parser.add_argument('--model_path', default='ViT-L/14', type=str, help='|模型名称或模型位置，中文文本模型只支持ViT-L/14(890M)|')
parser.add_argument('--chinese_model', default='chinese_model', type=str, help='|中文文本模型名称或模型位置|')
parser.add_argument('--device', default='cpu', type=str, help='|运行设备|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
app = flask.Flask(__name__)  # 创建一个服务框架


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
@app.route('/test/', methods=['POST'])  # 每当调用服务时会执行一次flask_app函数
def flask_app():
    request_json = flask.request.get_data()
    request_dict = json.loads(request_json)
    text = request_dict['text']
    use_chinese = request_dict['use_chinese']
    column, score = model.predict(text, use_chinese)
    result = {'column': column, 'score': score}
    return result


if __name__ == '__main__':
    print('| 使用flask启动服务 |')
    model = clip_class(args)
    app.run(host='0.0.0.0', port=9999, debug=False)  # 启动服务

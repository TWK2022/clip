# 启用flask_start的服务后，将数据以post的方式调用服务得到结果
import json
import requests

if __name__ == '__main__':
    url = 'http://0.0.0.0:9999/test/'  # 根据flask_start中的设置: http://host:port/name/
    text = ['Lipstick', 'Cat', 'Office']
    use_chinese = False
    request_dict = {'text': text, 'use_chinese': use_chinese}
    request = json.dumps(request_dict)
    response = requests.post(url, data=request)
    result = response.json()
    print(result)

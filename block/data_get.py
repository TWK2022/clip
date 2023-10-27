import numpy as np
import pandas as pd


def data_get(args):
    data_dict = data_prepare(args)._load()
    return data_dict


class data_prepare(object):
    def __init__(self, args):
        self.data_path = args.data_path

    def _load(self):
        # 读取数据
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
        except:
            df = pd.read_csv(self.data_path, encoding='gbk')
        input_data = df.columns
        output_data = df.values.T.astype(np.float32)
        data_dict = {'input_data': input_data, 'output_data': output_data}
        return data_dict

import numpy as np
import pandas as pd


def data_get(args):
    data_dict = data_prepare(args)._load()
    return data_dict


class data_prepare(object):
    def __init__(self, args):


    def _load(self):
        # 读取数据
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
        except:
            df = pd.read_csv(self.data_path, encoding='gbk')
        return data_dict
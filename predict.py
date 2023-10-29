import os
import clip
import torch
import argparse
import numpy as np
import transformers
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'  # 防止可能出现的libiomp5.dylib报错
parser = argparse.ArgumentParser(description='|clip文本搜图片|')
parser.add_argument('--database_path', default='feature_database.csv', type=str, help='|特征数据库位置|')
parser.add_argument('--model_path', default='ViT-L/14', type=str, help='|模型名称或模型位置，中文文本模型只支持ViT-L/14(890M)|')
parser.add_argument('--chinese_model', default='IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese', type=str,
                    help='|模型名称或模型位置|')
parser.add_argument('--device', default='cpu', type=str, help='|运行设备|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
class clip_class:
    def __init__(self, args):
        self.device = args.device
        self.model_path = args.model_path
        self.database_path = args.database_path
        # 模型
        clip_model, image_deal = clip.load(self.model_path, device=self.device)  # clip模型：图片模型+英文文本模型
        self.clip_model = clip_model.eval()
        chinese_tokenizer = transformers.BertTokenizer.from_pretrained(args.chinese_model)
        self.chinese_tokenizer = chinese_tokenizer
        chinese_model = transformers.BertForSequenceClassification.from_pretrained(
            args.chinese_model).eval().to(self.device)  # 中文文本模型，只支持ViT-L/14(890M)
        self.chinese_model = chinese_model
        # 数据
        df = pd.read_csv(self.database_path)
        column = df.columns
        image_feature = df.values
        self.column = column
        self.image_feature = image_feature
        print(f'| 初始化完成 |')

    def _deal(self, text_feature):  # 输入单个/多个文本，返回大于阈值的图片名和相似度
        text_feature /= torch.norm(text_feature, dim=1, keepdim=True)  # 归一化
        text_feature = text_feature.cpu().numpy()
        score = np.dot(text_feature, self.image_feature)
        index_list = [np.argmax(_) for _ in score]
        column = [self.column[_] for _ in index_list]
        score = [i[j] for i, j in zip(score, index_list)]
        return column, score

    def predict(self, text, use_chinese=True):  # 输入单个/多个文本，返回最匹配的图片和其相似度
        with torch.no_grad():
            # 英文
            if not use_chinese:
                english_sequence = clip.tokenize(text).to(self.device)  # 处理
                english_text_feature = self.clip_model.encode_text(english_sequence)  # 推理
                column, score = self._deal(english_text_feature)
            # 中文
            else:
                chinese_sequence = self.chinese_tokenizer(text, max_length=77, padding='max_length',
                                                          truncation=True, return_tensors='pt')['input_ids'].type(
                    torch.int32).to(self.device)  # 处理
                chinese_text_feature = self.chinese_model(chinese_sequence).logits  # 推理
                column, score = self._deal(chinese_text_feature)
        return column, score


if __name__ == '__main__':
    # 输入文本
    text = ['两只黑色的猫', '一只白色的狗']
    # text = ['Two Black cat', 'One White Dog']
    # 开始预测
    model = clip_class(args)
    column, score = model.predict(text, use_chinese=True)
    print(f'| 输入:{text} |')
    print(f'| 图片:{column} |')
    print(f'| 相似度:{[round(_, 3) for _ in score]} |')

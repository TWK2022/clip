import clip
import torch
import argparse
import numpy as np
import transformers
import pandas as pd

parser = argparse.ArgumentParser(description='')
parser.add_argument('--database_path', default='feature_database.csv', type=str, help='|特征数据库位置|')
parser.add_argument('--model_name', default='ViT-L/14', type=str, help='|模型名称，中文文本模型只支持ViT-L/14(890M)|')
parser.add_argument('--chinese_cache', default='/root/.cache/huggingface/hub', type=str, help='|中文文本模型缓存|')
parser.add_argument('--use_chinese', default=False, type=bool, help='|True时输入文本为中文，False时为英文|')
parser.add_argument('--device', default='cuda', type=str, help='|运行设备|')
args = parser.parse_args()


class clip_class:
    def __init__(self, args):
        self.device = args.device
        self.model_name = args.model_name
        self.database_path = args.database_path
        self.use_chinese = args.use_chinese
        # 模型
        model, image_deal = clip.load(self.model_name, device=self.device)  # clip模型：图片模型+英文文本模型
        self.model = model.eval()
        if self.use_chinese:
            chinese_tokenizer = transformers.BertTokenizer.from_pretrained(
                "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese", cache_dir=args.chinese_cache)
            self.chinese_tokenizer = chinese_tokenizer
            chinese_encode = transformers.BertForSequenceClassification.from_pretrained(
                "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese",
                cache_dir=args.chinese_cache).eval().half().to(self.device)  # 中文文本模型，只支持ViT-L/14(890M)
            self.chinese_encode = chinese_encode
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

    def predict(self, text):  # 输入单个/多个文本，返回图片名和概率值
        with torch.no_grad():
            # 英文
            if not self.use_chinese:
                english_sequence = clip.tokenize(text).to(self.device)  # 处理
                english_text_feature = self.model.encode_text(english_sequence)  # 推理
                column, score = self._deal(english_text_feature)
            # 中文
            else:
                chinese_sequence = self.chinese_tokenizer(text, max_length=77, padding='max_length',
                                                          truncation=True, return_tensors='pt')['input_ids'].type(
                    torch.int32).to(self.device)  # 处理
                chinese_text_feature = self.chinese_encode(chinese_sequence).logits  # 推理
                column, score = self._deal(chinese_text_feature)
        return column, score


if __name__ == '__main__':
    # 输入文本
    text = ['Lipstick', 'Cat', 'Office']
    # 开始预测
    model = clip_class(args)
    column, score = model.predict(text)  # 输入单个/多个文本，最匹配的图片和相似度
    print(f'| 输入:{text} |')
    print(f'| 图片:{column} |')
    print(f'| 相似度:{score} |')

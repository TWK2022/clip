import os
import PIL
import tqdm
import clip
import torch
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|建立图片数据库|')
parser.add_argument('--image_path', default='image_database', type=str, help='|图片文件夹位置|')
parser.add_argument('--save_path', default='feature_database.csv', type=str, help='|特征数据库保存位置(.csv)|')
parser.add_argument('--model_path', default='ViT-L/14', type=str, help='|模型名称或模型位置，中文文本模型只支持ViT-L/14(890M)|')
parser.add_argument('--batch', default=8, type=int, help='|模型预测的图片批量|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--num_worker', default=0, type=int, help='|数据处理cpu线程数|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, image_name, image_deal):
        self.image_name = image_name
        self.image_deal = image_deal

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        image = self.image_deal(PIL.Image.open(f'{args.image_path}/{self.image_name[index]}'))
        image = image.type(torch.float32)
        return image


if __name__ == '__main__':
    # 模型
    model, image_deal = clip.load(args.model_path, device=args.device)  # clip模型：图片模型+英文文本模型
    model = model.eval().float()
    print(f'| 加载模型成功:{args.model_path} |')
    # 图片处理
    image_name = sorted(os.listdir(args.image_path))
    dataset = torch_dataset(image_name, image_deal)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False,
                                             drop_last=False, num_workers=args.num_worker)
    with torch.no_grad():
        image_feature_list = []
        for image_batch in tqdm.tqdm(dataloader):
            image_batch = image_batch.to(args.device)
            image_feature = model.encode_image(image_batch)
            image_feature /= torch.norm(image_feature, dim=1, keepdim=True)
            image_feature_list.append(image_feature.cpu().numpy())
        image_feature = np.concatenate(image_feature_list, axis=0).T
    # 记录图片特征
    column = image_name
    df = pd.DataFrame(image_feature, columns=column, dtype=np.float32)
    df.to_csv(args.save_path, index=False, header=True)
    print(f'| 图片特征处理完毕:{len(image_name)}，保存在:{args.save_path} |')

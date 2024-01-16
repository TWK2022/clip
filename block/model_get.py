import transformers


def model_get(args):
    model = transformers.BertForSequenceClassification.from_pretrained(args.model)
    model_dict = {'model': model}
    model_dict['epoch'] = 0  # 已训练的轮次
    model_dict['optimizer_state_dict'] = None  # 学习率参数
    model_dict['lr_adjust_index'] = 0  # 学习率调整次数
    model_dict['ema_updates'] = 0  # ema参数
    return model_dict

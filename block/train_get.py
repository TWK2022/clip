import tqdm
import torch
import transformers
from block.ModelEMA import ModelEMA
from block.lr_get import adam, lr_adjust


def train_get(args, data_dict, model_dict, loss):
    # 加载模型
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    # 学习率
    optimizer = adam(args.regularization, args.r_value, model.parameters(), lr=args.lr_start, betas=(0.937, 0.999))
    optimizer.load_state_dict(model_dict['optimizer_state_dict']) if model_dict['optimizer_state_dict'] else None
    optimizer_adjust = lr_adjust(args, model_dict['lr_adjust_index'])  # 学习率调整函数
    optimizer = optimizer_adjust(optimizer, model_dict['epoch'] + 1, 0)  # 初始化学习率
    # 使用平均指数移动(EMA)调整参数(不能将ema放到args中，否则会导致模型保存出错)
    ema = ModelEMA(model) if args.ema else None
    if args.ema:
        ema.updates = model_dict['ema_updates']
    # 数据集
    train_dataset = torch_dataset(args, data_dict['train_input'], data_dict['train_output'])
    train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                   drop_last=True, pin_memory=args.latch, num_workers=args.num_worker,
                                                   sampler=train_sampler)
    # 分布式初始化
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank) if args.distributed else model
    epoch_base = model_dict['epoch'] + 1  # 新的一轮要+1
    for epoch in range(epoch_base, epoch_base + args.epoch):  # 训练
        print(f'\n-----------------------第{epoch}轮-----------------------') if args.local_rank == 0 else None
        model.train()
        train_loss = 0  # 记录训练损失
        if args.local_rank == 0:  # tqdm
            tqdm_len = len(data_dict['train_input']) // args.batch // args.device_number * args.device_number
            tqdm_show = tqdm.tqdm(total=tqdm_len, mininterval=0.2)
        for index, (input_ids, attention_mask, true_batch) in enumerate(train_dataloader):
            input_ids = input_ids.to(args.device, non_blocking=args.latch)
            attention_mask = attention_mask.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            if args.amp:
                with torch.cuda.amp.autocast():
                    pred_batch = model(input_ids, attention_mask).logits
                    loss_batch = loss(pred_batch, true_batch)
                args.amp.scale(loss_batch).backward()
                args.amp.step(optimizer)
                args.amp.update()
                optimizer.zero_grad()
            else:
                pred_batch = model(input_ids, attention_mask).logits
                loss_batch = loss(pred_batch, true_batch)
                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()
            # 调整参数，ema.updates会自动+1
            ema.update(model) if args.ema else None
            # 记录损失
            train_loss += loss_batch.item()
            # tqdm
            if args.local_rank == 0:
                tqdm_show.set_postfix({'train_loss': loss_batch.item()})  # 添加loss显示
                tqdm_show.update(args.device_number)  # 更新进度条
        # tqdm
        if args.local_rank == 0:
            tqdm_show.close()
        # 计算平均损失
        train_loss /= index + 1
        if args.local_rank == 0:
            print(f'\n| train_loss:{train_loss:.4f} | lr:{optimizer.param_groups[0]["lr"]:.6f} |\n')
        # 调整学习率
        optimizer = optimizer_adjust(optimizer, epoch + 1, train_loss)
        # 清理显存空间
        del input_ids, attention_mask, true_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 保存
        if args.local_rank == 0:  # 分布式时只保存一次
            if epoch % 5 == 0:
                model.save_pretrained(f'save_epoch_{epoch}')
        torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args.model)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        input_dict = self.tokenizer(self.input_data[index], max_length=77, padding='max_length', truncation=True,
                                    return_tensors='pt')
        input_ids = input_dict['input_ids'].type(torch.int32).squeeze(0)
        attention_mask = input_dict['attention_mask'].type(torch.int32).squeeze(0)
        label = torch.tensor(self.output_data[index], dtype=torch.float32)  # 转换为tensor
        return input_ids, attention_mask, label

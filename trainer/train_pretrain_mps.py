import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')

# 设置环境变量以优化MPS内存使用
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

def Logger(content):
    print(content)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, wandb):
    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)

        lr = get_lr(step, len(train_loader), args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播
        outputs = model(X)
        loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), 
                       Y.view(-1)).view(Y.size())
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # 更高效的内存释放

        # 日志记录
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{len(train_loader)}) '
                  f'loss:{loss.item():.3f} lr:{lr:.7f} '
                  f'time:{(spend_time/(step+1)*len(train_loader)//60):.0f}min')

            if wandb:
                wandb.log({
                    "loss": loss.item(),
                    "lr": lr,
                    "epoch": epoch,
                    "step": step
                })

        # 内存清理
        if args.device == "mps":
            torch.mps.empty_cache()

        # 模型保存
        if (step + 1) % args.save_interval == 0:
            save_model(epoch, step)

def save_model(epoch, step):
    model.eval()
    state_dict = model.state_dict()
    # 使用半精度保存以减小文件大小
    state_dict = {k: v.half() if v.dtype == torch.float32 else v 
                 for k, v in state_dict.items()}
    
    ckp_path = f"{args.save_dir}/checkpoint_epoch{epoch}_step{step}.pt"
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckp_path)
    model.train()

def init_model():
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe
    )
    model = MiniMindForCausalLM(config).to(args.device)
    
    # 打印模型大小
    num_params = sum(p.numel() for p in model.parameters())
    Logger(f"模型参数数量: {num_params/1e6:.2f}M")
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # 数据参数
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="../model/")
    
    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=384)  # 减小默认值
    parser.add_argument("--num_hidden_layers", type=int, default=6)  # 减小默认值
    parser.add_argument("--max_seq_len", type=int, default=256)  # 减小默认值
    parser.add_argument("--use_moe", action="store_true")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8)  # 减小默认值
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # 系统参数
    parser.add_argument("--device", type=str, 
                      default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="../out")
    
    # 日志参数
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()

    # 初始化
    os.makedirs(args.out_dir, exist_ok=True)
    args.save_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备设置
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.9)  # 设置内存限制
        
    # 初始化模型和数据
    model, tokenizer = init_model()
    train_ds = PretrainDataset(args.data_path, tokenizer, args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # WandB
    wandb = None
    if args.use_wandb:
        import wandb
        wandb.init(project="MiniMind-M4", config=vars(args))
    
    # 训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
    
    # 最终保存
    save_model(args.epochs, len(train_loader))
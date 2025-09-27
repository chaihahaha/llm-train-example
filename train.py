import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

def build_lm_dataset(pathtoparquet: str, tokenizer, block_size: int = 1024):
    """
    构建语言模型数据集：加载、分词并按固定长度块分组。
    """
    # 1. 加载原始数据
    raw = load_dataset(path="parquet")
    
    # 2. 分词函数（注意：tokenizer 必须返回 input_ids 等字段）
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # 我们后面会手动填充，所以这里设为 False
        )
    
    tokenized = raw.map(
        tokenize_function,
        batched=True,
        remove_columns=raw["train"].column_names,  # 移除原始列，只保留 tokenized 字段
        desc="Tokenizing dataset",
    )

    # 3. 按块分组函数：将所有文本拼接成一个长序列，再切成固定长度的块
    def group_texts(examples):
        # 将所有 input_ids 拼接成一个长列表
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        
        # 计算总长度
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # 如果长度不够一个 block_size，可以丢弃或填充（这里我们选择截断）
        total_length = (total_length // block_size) * block_size
        
        # 切分成固定长度的块
        result = {
            k: [concatenated_examples[k][i:i + block_size] 
                for i in range(0, total_length, block_size)]
            for k in concatenated_examples.keys()
        }
        
        # 为语言模型设置 labels（通常与 input_ids 相同）
        result["labels"] = result["input_ids"].copy()
        
        return result

    # 应用分组
    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into blocks of {block_size}",
    )

    return lm_dataset


def train(pathtotok: str, pathtoparquet: str, pathtosave: str):
    """
    训练 GPT-2 语言模型
    """
    # 1. 加载 tokenizer（确保使用正确的预训练词表）
    tokenizer = AutoTokenizer.from_pretrained(pathtotok)
    
    # 2. 检查是否已设置 pad_token，GPT-2 默认没有 padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 3. 构建数据集
    ds = build_lm_dataset(pathtoparquet, tokenizer, block_size=512)  # 可根据显存调整

    # 4. 加载模型配置，并实例化模型
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,       # 与 block_size 对齐
        n_ctx=512,
        n_embd=768,            # 可根据需求调整
        n_layer=12,
        n_head=12,
    )
    
    model = GPT2LMHeadModel(model_config)
    
    # 5. 初始化数据收集器（自动处理 padding 和 masking）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 语言模型不使用 masked language modeling
    )

    # 6. 设置训练参数（关键！）
    train_args = TrainingArguments(
        output_dir=pathtosave,               # 模型保存路径
        evaluation_strategy="epoch",         # 每个 epoch 评估一次
        learning_rate=2e-5,                  # 学习率
        per_device_train_batch_size=8,       # 根据显存调整（如 16/32）
        per_device_eval_batch_size=8,
        num_train_epochs=5,                  # 训练轮数
        weight_decay=0.01,
        logging_dir="./logs",                # 日志目录
        save_strategy="epoch",
        load_best_model_at_end=True,         # 保存最佳模型
        metric_for_best_model="eval_loss",   # 根据验证损失选择最优模型
        report_to="tensorboard",
        gradient_accumulation_steps=4,       # 梯度累积，节省显存
    )

    # 7. 创建 Trainer 实例（注意：参数顺序和名称要正确）
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),  # 确保有验证集，否则去掉
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. 开始训练
    trainer.train()

    # 9. 保存模型（推荐使用 save_pretrained，而不是 torch.save）
    model.save_pretrained(pathtosave)
    tokenizer.save_pretrained(pathtosave)

    print(f"✅ 模型已保存至: {pathtosave}")

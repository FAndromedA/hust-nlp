import torch
from transformers import BertForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


def remove_spaces(example):
    example['text'] = example['text'].replace(' ', '')
    return example


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, the_tokenizer):
        self.encodings = the_tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors='pt')

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('./bert')
    model = BertForMaskedLM.from_pretrained('./bert')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # print(len(tokenizer))
    # 拓展 tokenizer
    new_corpus = load_dataset('text', data_files={'train':'./data/train.txt'})

    new_corpus = new_corpus.map(remove_spaces)
    old_vocab = tokenizer.vocab
    old_vocab_size = len(tokenizer)
    new_tokens = set()
    sentences = []
    for dct in new_corpus['train']:
        sentences.append(dct['text'])
        for token in dct['text']:
            if token not in old_vocab:
                new_tokens.add(token)
    # print(list(new_tokens))

    tokenizer.add_tokens(list(new_tokens))
    tokenizer.save_pretrained('./myBert')
    # print(len(tokenizer))

    # 训练 bert 嵌入
    model.resize_token_embeddings(len(tokenizer))
    # for param in model.bert.embeddings.word_embeddings.parameters():
    #     param.requires_grad = False
    # model.bert.embeddings.word_embeddings.weight[old_vocab_size:].requires_grad = True

    # 冻结BERT的底层参数，只训练顶部层
    for name, param in model.named_parameters():
        if name.startswith("bert.encoder.layer.11") or name.startswith("bert.pooler") or name.startswith("classifier"):
            print(name)
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    training_args = TrainingArguments(
        output_dir='./myBert',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=10000,
        save_total_limit=2,
    )

    train_dataset = MyDataset(sentences, tokenizer)
    # 使用 DataCollatorForLanguageModeling 创建 MLM 数据
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    # 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    # 训练模型
    trainer.train()
    
    model.save_pretrained('./myBert')

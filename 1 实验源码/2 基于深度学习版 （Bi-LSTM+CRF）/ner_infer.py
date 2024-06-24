import torch
import pickle
from transformers import BertTokenizer

separator = ['、','，','。','！','？',',','.','!','?']

if __name__ == '__main__':
    model = torch.load('nerSave/dp_model_epoch29.pkl', map_location=torch.device('cuda'))
    tokenizer = BertTokenizer.from_pretrained('./myBert')
    output = open('myner_result.txt', 'w', encoding='utf-8')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('ner-data.pkl', 'rb') as inp:
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
    #id2tag = ['B', 'M', 'E', 'S']
    with open('data/test_data.txt', 'r', encoding='utf-8') as f:
        count = 0
        test_list = f.readlines()# for lines
        total = len(test_list)
        for test in test_list:
            count += 1
            if count % 100 == 0:
                print(f"processing {count}/{total}={count/total}")
            flag = False
            test = test.strip()
            
            sub_sentence = []
            if len(test) > 512 :
                for i in range(433, 512):
                    if test[i] in separator:
                        sub_sentence.append(test[:i+1])
                        sub_sentence.append(test[i+1:])
                        break
            else :
                sub_sentence.append(test)

            for sentence in sub_sentence:
                # print(sentence, end='', file=output)
                inputs = tokenizer.encode_plus(
                    sentence,
                    None,
                    add_special_tokens=False,
                    max_length=512,
                    pad_to_max_length=False, #不足部分填充
                    return_token_type_ids=None,
                    truncation=True #超过部分截断
                )
                ids = torch.tensor(inputs['input_ids'], dtype=torch.long).to(device, dtype=torch.long).unsqueeze(0)
                mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device, dtype=torch.bool).unsqueeze(0)
                predict = model.infer(ids, mask)[0]

                for i in range(len(ids[0])):
                    #print(, end='', file=output)
                    token = tokenizer.decode(ids[0][i]).replace('#','').replace(' ','')
                    for c in token:
                        print(c, end=' ', file=output)
                        print(id2tag[predict[i]], file=output)   

            
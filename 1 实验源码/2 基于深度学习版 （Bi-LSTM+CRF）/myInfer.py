import torch
import pickle
from transformers import BertTokenizer


if __name__ == '__main__':
    model = torch.load('mySave/model_epoch29.pkl', map_location=torch.device('cuda'))
    tokenizer = BertTokenizer.from_pretrained('./myBert')
    output = open('mycws_result.txt', 'w', encoding='utf-8')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('data/mydatasave.pkl', 'rb') as inp:
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
    
    with open('data/test.txt', 'r', encoding='utf-8') as f:
        count = 0
        test_list = f.readlines()
        total = len(test_list)
        for test in test_list:
            count += 1
            if count % 100 == 0:
                print(f"processing {count}/{total}={count/total}%")
            flag = False
            test = test.strip()

            #x = torch.LongTensor(1, len(test))
            #length = [len(test)]
            if len(test) == 1:
                print(test, file=output)
            else :
                inputs = tokenizer.encode_plus(
                    test,
                    None,
                    add_special_tokens=False,
                    max_length=512,
                    pad_to_max_length=False, #不足部分填充
                    return_token_type_ids=None,
                    truncation=True #超过部分截断
                )
                ids = torch.tensor(inputs['input_ids'], dtype=torch.long).to(device, dtype=torch.long).unsqueeze(0)
                mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device, dtype=torch.bool).unsqueeze(0)
                #print("1", tokenizer.decode(ids[0][4]), len(ids[0]))
                predict = model.infer(ids, mask)[0]
                
                #print("2", len(test), len(predict))
                for i in range(len(ids[0])):
                    print(tokenizer.decode(ids[0][i]).replace('#',''), end='', file=output)
                    if id2tag[predict[i]] in ['E', 'S']:
                        print(' ', end='', file=output)
            print(file=output)
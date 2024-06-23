import pickle
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

SAVE_PATH = './ner-data.pkl'

def process_ner_data(file_path):
    tag_num = 0
    id2tag = []
    tag2id = {}
    current_sentence = []
    current_labels = []
    x_data = []
    y_data = []
    
    tokenizer = BertTokenizer.from_pretrained('./myBert')
    max_len = 0

    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                char, label = line.split()
                if label not in tag2id:
                    tag2id[label] = tag_num
                    id2tag.append(label)
                    tag_num += 1
                current_sentence.append(char)
                current_labels.append(tag2id[label])
            else:
                if len(current_sentence) > 512:
                    current_sentence = current_sentence[:512]
                    current_labels = current_labels[:512]
                # tmp_sentence = []
                # for i in range(0, len(current_sentence), 512):
                #     tmp_sentence.extend(tokenizer.encode(current_sentence[i:i+512], add_special_tokens=False))
                current_sentence = tokenizer.encode(current_sentence, add_special_tokens=False)
                assert len(current_sentence) == len(current_labels)
                x_data.append(current_sentence)
                y_data.append(current_labels)
                max_len = max(max_len, len(current_sentence))
                current_sentence = []
                current_labels = []
        
        if current_sentence:
            if len(current_sentence) > 512:
                current_sentence = current_sentence[:512]
                current_labels = current_labels[:512]
            # tmp_sentence = []
            # for i in range(0, len(current_sentence), 512):
            #     tmp_sentence.extend(tokenizer.encode(current_sentence[i:i+512], add_special_tokens=False))
            current_sentence = tokenizer.encode(current_sentence, add_special_tokens=False)
            assert len(current_sentence) == len(current_labels)
            x_data.append(current_sentence)
            y_data.append(current_labels) 
            max_len = max(max_len, len(current_sentence))

    # print(x_data[0], y_data[0])
    # print(tokenizer.decode(x_data[0]))
    print(tag2id)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=43)
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
    print(f"max length {max_len}, len of sentences {len(x_data)}")
    print("data preprocess succeed")


if __name__ == '__main__':
    process_ner_data('./RMRB_NER_CORPUS.txt')
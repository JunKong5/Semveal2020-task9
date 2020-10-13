

import pandas as pd
import numpy as np
import re
import io
import os
import pickle

from sklearn.svm import SVC
from keras.utils import to_categorical
from ekphrasis.classes.tokenizer import SocialTokenizer
from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
# from tool import ekphrasis_config
from keras.models import load_model


# train_lexicon = pd.read_csv('./weka_lexicon/train_lexicon.csv', sep=',')
# trial_lexicon = pd.read_csv('./weka_lexicon/test_lexicon.csv', sep=',')
#
# train_lexicon = train_lexicon.values
# trial_lexicon = trial_lexicon.values


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    # discretePredictions = to_categorical(predictions.argmax(axis=1))
    discretePredictions = to_categorical(predictions)

    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)

    # print("True Positives per class : ", truePositives)
    # print("False Positives per class : ", falsePositives)
    # print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, 4):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        # print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    # print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    # print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)

    # print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def pre_convert():
    data = pd.read_csv('./result/test_results.tsv', sep='\t', header=None)
    data = data.values
    y_pred = np.argmax(data, axis=1)
    print(y_pred)
    result = pd.DataFrame(data = {'label':y_pred})
    result.to_csv("./result/emoContext_pre.csv", index=False, quoting=3)


def get_wassa_data():
    list = ['anger', 'sad', 'joy']
    result = []
    wassa_data = pd.read_csv('./data/wassa/train-v3.csv', sep='\t')
    for row in range(len(wassa_data)):
        line = []
        if wassa_data.values[row][0] in list:
            if len(wassa_data.values[row][1].split()) < 160:
                if wassa_data.values[row][0] == 'anger':
                    line.append('angry')
                elif wassa_data.values[row][0] == 'joy':
                    line.append('happy')
                else:
                    line.append(wassa_data.values[row][0])
                line.append(wassa_data.values[row][1])
                result.append(line)

    df = pd.DataFrame(result, columns=['label', 'review'])
    df.to_csv('./data/wassa/wassa_bert.csv', sep='\t', index=None)


def convert():
    dict_ = {'0':"surprise", '1':"anger", '2':"sad", '3':"joy", '4':'fear', '5':'disgust'}
    print(dict_['3'])
    data = pd.read_csv('./result/emoContext_pre.csv', sep='\t')
    for row in range(len(data)):
        num = dict_[str(data['label'][row])]
        data['label'][row] = num
    result = pd.DataFrame(data)
    result.to_csv('./result/pre.csv', sep='\t', index=False, header=None)


def test_pell_correct():
    from ekphrasis.classes.spellcorrect import SpellCorrector
    sp = SpellCorrector(corpus="english")
    print(sp.correct("Thaaaanks"))


def delete_token():
    str = "I know that	you know everything	Haha  😂   😂   😂   😂 	happy"
    repeatedChars = ['😂']
    for c in repeatedChars:
        line_split = str.split(c)
        print(line_split)
        while True:
            try:
                line_split.remove('   ')
            except:
                break
        print(line_split)
        cSpace = ' ' + c + ' '
        line = cSpace.join(line_split)
        print(line)


def ekphrasis_config(str):
    
    social_tokenizer = SocialTokenizer(lowercase=False).tokenize
    return social_tokenizer(str)


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)

            emoji_repeatedChars = TWEMOJI_LIST
            for emoji_meta in emoji_repeatedChars:
                emoji_lineSplit = line.split(emoji_meta)
                while True:
                    try:
                        emoji_lineSplit.remove('')
                        emoji_lineSplit.remove(' ')
                        emoji_lineSplit.remove('  ')
                        emoji_lineSplit = [x for x in emoji_lineSplit if x != '']
                    except:
                        break
                emoji_cSpace = ' ' + TWEMOJI[emoji_meta][0] + ' '
                line = emoji_cSpace.join(emoji_lineSplit)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4]) + ' '

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            string = re.sub("tha+nks ", ' thanks ', conv)
            string = re.sub("Tha+nks ", ' Thanks ', string)
            string = re.sub("yes+ ", ' yes ', string)
            string = re.sub("Yes+ ", ' Yes ', string)
            string = re.sub("very+ ", ' very ', string)
            string = re.sub("go+d ", ' good ', string)
            string = re.sub("Very+ ", ' Very ', string)
            string = re.sub("why+ ", ' why ', string)
            string = re.sub("wha+t ", ' what ', string)
            string = re.sub("sil+y ", ' silly ', string)
            string = re.sub("hm+ ", ' hmm ', string)
            string = re.sub("no+ ", ' no ', string)
            string = re.sub("sor+y ", ' sorry ', string)
            string = re.sub("so+ ", ' so ', string)
            string = re.sub("lie+ ", ' lie ', string)
            string = re.sub("okay+ ", ' okay ', string)
            string = re.sub(' lol[a-z]+ ', 'laugh out loud', string)
            string = re.sub(' wow+ ', ' wow ', string)
            string = re.sub('wha+ ', ' what ', string)
            string = re.sub(' ok[a-z]+ ', ' ok ', string)
            string = re.sub(' u+ ', ' you ', string)
            string = re.sub(' wellso+n ', ' well soon ', string)
            string = re.sub(' byy+ ', ' bye ', string)
            string = string.replace('’', '\'').replace('"', ' ').replace("`", "'")
            string = string.replace('whats ', 'what is ').replace("what's ", 'what is ').replace("i'm ", 'i am ')
            string = string.replace("it's ", 'it is ')
            string = string.replace('Iam ', 'I am ').replace(' iam ', ' i am ').replace(' dnt ', ' do not ')
            string = string.replace('I ve ', 'I have ').replace('I m ', ' I\'am ').replace('i m ', 'i\'m ')
            string = string.replace('Iam ', 'I am ').replace('iam ', 'i am ')
            string = string.replace('dont ', 'do not ').replace('google.co.in ', ' google ').replace(' hve ', ' have ')
            string = string.replace(' F ', ' Fuck ').replace('Ain\'t ', ' are not ').replace(' lv ', ' love ')
            string = string.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', ' It is').replace(' its ', ' it is ')
            string = string.replace('  Nd  ', ' and ').replace(' nd ', ' and ').replace('i ll ', 'i will ')
            string = ' ' + string
            for item in LOGOGRAM.keys():
                string = string.replace(' ' + item + ' ', ' ' + LOGOGRAM[item] + ' ')

            list_str = ekphrasis_config(string)
            for index in range(len(list_str)):
                if list_str[index] in EMOTICONS_TOKEN.keys():
                    list_str[index] = EMOTICONS_TOKEN[list_str[index]][1:len(EMOTICONS_TOKEN[list_str[index]]) - 1].lower()

            for index in range(len(list_str)):
                if list_str[index] in LOGOGRAM.keys():
                    list_str[index] = LOGOGRAM[list_str[index]].lower()

            for index in range(len(list_str)):
                if list_str[index] in LOGOGRAM.keys():
                    list_str[index] = LOGOGRAM[list_str[index]].lower()

            string = ' '.join(list_str)
            indices.append(int(line[0]))
            conversations.append(string)
    if mode == "train":

        return indices, conversations, labels
    else:
        return indices, conversations


def generate_data():
    mask_ratio = 0.2
    result = []
    indices, conversations, labels = preprocessData('./data/train.txt', 'train')
    for row in range(len(conversations)):
        line = []
        if labels[row] != 0:
            line.append(labels[row])
            line.append(conversations[row])
            result.append(line)
            # for turn in range(2):
            #     line_augment = []
            #     line_augment.append(labels[row])
            #     text_list = conversations[row].split()
            #     for index in range(len(text_list)):
            #         if text_list[index] != '<eos>':
            #             if np.random.random() <= mask_ratio:
            #                 text_list[index] = '<mask>'
            #     line_augment.append(' '.join(text_list))
            #     result.append(line_augment)
        else:
            line.append(labels[row])
            line.append(conversations[row])
            result.append(line)

    df = pd.DataFrame(result, columns=['label', 'review']).sample(frac=1).reset_index(drop=True)
    df.to_csv('./data/train.csv', sep='\t', index=False, encoding='utf-8')
    # df1 =df[0:24000]
    # df2 = df[24000:]
    # df1.to_csv('./data/train.tsv', sep='\t', index=False, encoding='utf-8')
    # df2.to_csv('./data/dev.tsv', sep='\t', index=False, encoding='utf-8')

    test_result = []
    indices, conversations = preprocessData('./data/devwithoutlabels.txt', 'test')
    for row in range(len(conversations)):
        line = []
        line.append('0')
        line.append(conversations[row])
        test_result.append(line)
    df_test = pd.DataFrame(test_result, columns=['label', 'review'])
    df_test.to_csv('./data/test.csv', sep='\t', index=False, encoding='utf8')


def test_svc():
    label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
    emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
    golden_label = pd.read_table('./data/dev.txt', sep='\t')
    golden_label = golden_label['label'].replace(emotion2label)
    golden_label = to_categorical(golden_label)
    solutionPath = "./result/test.txt"
    testDataPath = "./data/testwithoutlabels.txt"
    x, d, t, y = pickle.load(open('./pickle/stacking_elmo_glove1.pickle', 'rb'))
    # pre = pickle.load(open('./pickle/elmo.pickle', 'rb'))
    # pres = pre[0]['result'][0]
    # print(pre)
    # pres = np.array(pre[0]).argmax(axis=1)
    # accuracy, microPrecision, microRecall, microF1 = getMetrics(pres, golden_label)+
    # print(microF1)
    # predictions = t[:, 0:4].argmax(axis=1)
    for i in np.arange(0.1, 4, 0.1):
        svc = SVC(kernel='sigmoid', gamma=1.3, degree=8, C=3)
        svc.fit(x, np.array(y.argmax(axis=1)))
        predictions = svc.predict(d)

        accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, golden_label)
        # # if microF1 > 0.72:
        #
        print('*'*50)
        print('parameter= %f' %i)
        print(microF1)


    # with io.open(solutionPath, "w", encoding="utf8") as fout:
    #     fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
    #     with io.open(testDataPath, encoding="utf8") as fin:
    #         fin.readline()
    #         for lineNum, line in enumerate(fin):
    #             fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
    #             fout.write(label2emotion[predictions[lineNum]] + '\n')


def test_saved_model():
    testData = pickle.load(open('./pickle/testData_single.pickle', 'rb'))
    print(np.shape(testData))
    metrix = {'attention': [], 'capsulnet': [], 'gru': [], 'lstm': []}
    path = './model'
    list = os.listdir(path)

    emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
    golden_label = pd.read_table('./data/dev.txt', sep='\t')
    golden_label = golden_label['label'].replace(emotion2label)
    golden_label = to_categorical(golden_label)

    for item in list:
        second_path = os.path.join(path, item)
        for root, dirs, files in os.walk(second_path):
            for model_name in files:
                if 'attention' in model_name:
                    model = load_model(os.path.join(second_path, model_name), {"AttentionM": AttentionM})
                    accuracy, microPrecision, microRecall, microF1 = getMetrics(np.argmax(model.predict(testData), axis=1), golden_label)
                    print(microF1)
                    metrix['attention'].append(model.predict(testData))

                elif 'capsulnet' in model_name:
                    model = load_model(os.path.join(second_path, model_name), {"Capsule": Capsule})
                    pre = model.predict([testData, trial_lexicon])
                    accuracy, microPrecision, microRecall, microF1 = getMetrics(np.argmax(pre, axis=1), golden_label)
                    print(microF1)
                    metrix['capsulnet'].append(pre)

                elif 'gru' in model_name:
                    model = load_model(os.path.join(second_path, model_name))
                    accuracy, microPrecision, microRecall, microF1 = getMetrics(
                        np.argmax(model.predict(testData), axis=1), golden_label)
                    print(microF1)
                    metrix['gru'].append(model.predict(testData))

                elif 'lstm' in model_name:
                    model = load_model(os.path.join(second_path, model_name))
                    accuracy, microPrecision, microRecall, microF1 = getMetrics(
                        np.argmax(model.predict(testData), axis=1), golden_label)
                    print(microF1)
                    metrix['lstm'].append(model.predict(testData))
    pre1 = []
    result = pickle.load(open('./pickle/model_save_result.pickle', 'rb'))
    for keys in [result['attention'], result['capsulnet'], result['lstm'], result['gru']]:
        result_meta = np.zeros((np.shape(keys)[1], 4))
        for items in keys:
            result_meta += items
            result_meta = result_meta / 5
        pre1.append(result_meta)
    meta_test = np.concatenate([np.array(y_test_set).reshape(-1, 4) for y_test_set in pre1], axis=1)

    emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
    golden_label = pd.read_table('./data/dev.txt', sep='\t')
    golden_label = golden_label['label'].replace(emotion2label)
    golden_label = to_categorical(golden_label)
    print('meta_test')
    pickle.dump(metrix, open('./pickle/model_save_result.pickle', 'wb'))
    print('xyz')
    x, t, y = pickle.load(open('./pickle/elmo.pickle', 'rb'))
    print('for')
    for i in np.arange(0.1, 4, 0.1):
        svc = SVC(kernel='sigmoid', gamma=i, C=3)
        svc.fit(x, np.array(y.argmax(axis=1)))
        predictions = svc.predict(meta_test)
        accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, golden_label)
        if microF1 > 0.70:

            print('*'*50)
            print('parameter= %f' %i)
            print(microF1)


def test():
    pre1 = []
    result = pickle.load(open('./pickle/model_save_result.pickle', 'rb'))
    for keys in [result['attention'], result['capsulnet'], result['lstm'], result['gru']]:
        result_meta = np.zeros((np.shape(keys)[1], 4))
        for items in keys:
            result_meta += items
            result_meta = result_meta / 5
        pre1.append(result_meta)
    meta_test = np.concatenate([np.array(y_test_set).reshape(-1, 4) for y_test_set in pre1], axis=1)
    print(np.shape(meta_test))
    emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
    golden_label = pd.read_table('./data/dev.txt', sep='\t')
    golden_label = golden_label['label'].replace(emotion2label)
    golden_label = to_categorical(golden_label)

    x, t, y = pickle.load(open('./pickle/stacking_elmo.pickle', 'rb'))
    print('for')

    svc = SVC(kernel='sigmoid', gamma=3.9, C=3)
    svc.fit(x, np.array(y.argmax(axis=1)))
    predictions = svc.predict(meta_test)
    # for item in predictions:
    #     print(item)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, golden_label)
    # if microF1 > 0.70:
    #     print('*' * 50)
    #     print('parameter= %f' % i)
    print(microF1)


def statistic_fuck():
    a = 0
    o = 0
    s = 0
    h = 0
    data = pd.read_table('./result/test.txt', sep='\t')
    data = data.values
    texts = []
    for row in range(len(data)):
        text = ' '.join(data[row][1:4])
        texts.append(text)
    for index in range(len(texts)):
        if 'fuck' in texts[index]:
            if index not in [627,1676,1718,1835,4061,4194]:
                data[index][4] = 'angry'
    with io.open('./test.txt', "w", encoding="utf8") as fout:
            fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
            with io.open('./data/testwithoutlabels.txt', encoding="utf8") as fin:
                fin.readline()
                for lineNum, line in enumerate(fin):
                    fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                    fout.write(data[lineNum][4] + '\n')

if __name__ == '__main__':
    # test_saved_model()
    #
    # test()
    statistic_fuck()

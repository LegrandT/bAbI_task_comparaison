import os

import pprint
import tensorflow as tf

from mmn2nData import read_data, pad_data, depad_data
from mmn2nmodel import MemN2N

import numpy as np

from sklearn.metrics import accuracy_score

scores = []
scoresAv = []

iterations = 6

for t in range(1,21):
    av = []
    for it in range(iterations):
        param = {}

        param["embDim"] = 20
        param["nhop"] = 6
        param["mem_size"] = 50
        param["batch_size"] = 64
        param["nepoch"] = 100
        param["anneal_epoch"] = 25
        param["init_lr"] = 0.01
        param["anneal_rate"] = 0.5
        param["init_mean"] = 0.
        param["init_std"] = 0.1
        param["max_grad_norm"] = 40
        param["checkpoint_dir"] = "./checkpoints"
        param["lin_start"] = False
        param["is_test"] = False

        print('it ' + str(it) + ' file ' + str(t))

        word2idx = {}
        max_words = 0
        max_sentences = 0

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')

        train_stories, train_questions, max_words, max_sentences = read_data('./bAbI/en-valid/qa{}_train.txt'.format(t), word2idx, max_words, max_sentences)
        valid_stories, valid_questions, max_words, max_sentences = read_data('./bAbI/en-valid/qa{}_valid.txt'.format(t), word2idx, max_words, max_sentences)
        test_stories, test_questions, max_words, max_sentences = read_data('./bAbI/en-valid/qa{}_test.txt'.format(t), word2idx, max_words, max_sentences)

        pad_data(train_stories, train_questions, max_words, max_sentences)
        pad_data(valid_stories, valid_questions, max_words, max_sentences)
        pad_data(test_stories, test_questions, max_words, max_sentences)

        idx2word = dict(zip(word2idx.values(), word2idx.keys()))
        param["nwords"] = len(word2idx)
        param["max_words"] = max_words
        param["max_sentences"] = max_sentences

        print(param)

        with tf.Session() as sess:
            model = MemN2N(param, sess)
            model.build_model()

            if param["is_test"]:
                model.run(valid_stories, valid_questions, test_stories, test_questions)
            else:
                model.run(train_stories, train_questions, valid_stories, valid_questions)

            predictions, target = model.predict(train_stories, train_questions)


        pred = []
        targ = []
        for i in range(len(predictions)):
            pred.append(np.argmax(predictions[i]))
            targ.append(train_questions[i]['answer'][0])

        accuracy = accuracy_score(targ, pred)
        print(accuracy)
        av.append(accuracy)

        index = 0

        depad_data(train_stories, train_questions)

        question = train_questions[index]['question']
        answer = train_questions[index]['answer']
        story_index = train_questions[index]['story_index']
        sentence_index = train_questions[index]['sentence_index']

        story = train_stories[story_index][:sentence_index + 1]

        # story = [list(map(idx2word.get, sentence)) for sentence in story]
        # question = list(map(idx2word.get, question))
        # prediction = [idx2word[np.argmax(predictions[index])]]
        # answer = list(map(idx2word.get, answer))
        #
        # print('Story:')
        # print(story)
        # print('\nQuestion:')
        # print(question)
        # print('\nPrediction:')
        # print(prediction)
        # print('\nAnswer:')
        # print(answer)
        # print('\nCorrect:')
        # print(prediction == answer)
    scores.append(av)
    scoresAv.append(sum(av) / iterations)
print(scoresAv)
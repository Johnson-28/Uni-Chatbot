import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras import Input, Model
from keras.preprocessing.sequence import pad_sequences
import re
import pickle

from keras.models import load_model

model = load_model('C:/Users/Johnson/chatbot_model_tt_sgd.h5')
model1 = load_model('C:/Users/Johnson/chatbot_model_rms.h5')
enc_model = load_model('C:/Users/Johnson/chatbot_model_enc.h5')
dec_model = load_model('C:/Users/Johnson/chatbot_model_dec.h5')
# loading
with open('C:/Users/Johnson/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
import json
import random

intents = json.loads(open('intents.json').read(), strict=False)
words = pickle.load(open('C:/Users/Johnson/words.pkl', 'rb'))
classes = pickle.load(open('C:/Users/Johnson/classes.pkl', 'rb'))


def clean_text(text_to_clean):
    res = text_to_clean.lower()
    res = re.sub(r"i'm", "i am", res)
    res = re.sub(r"he's", "he is", res)
    res = re.sub(r"she's", "she is", res)
    res = re.sub(r"it's", "it is", res)
    res = re.sub(r"that's", "that is", res)
    res = re.sub(r"what's", "what is", res)
    res = re.sub(r"where's", "where is", res)
    res = re.sub(r"how's", "how is", res)
    res = re.sub(r"\'ll", " will", res)
    res = re.sub(r"\'ve", " have", res)
    res = re.sub(r"\'re", " are", res)
    res = re.sub(r"\'d", " would", res)
    res = re.sub(r"\'re", " are", res)
    res = re.sub(r"won't", "will not", res)
    res = re.sub(r"can't", "cannot", res)
    res = re.sub(r"n't", " not", res)
    res = re.sub(r"n'", "ng", res)
    res = re.sub(r"'bout", "about", res)
    res = re.sub(r"'til", "until", res)
    res = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", res)
    res = re.sub(r"[^\w\s]", "", res)
    return res


# def make_inference_models():
#     dec_state_input_h = Input(shape=(200,))
#     dec_state_input_c = Input(shape=(200,))
#     dec_states_inputs = [dec_state_input_h, dec_state_input_c]
#     dec_outputs, state_h, state_c = dec_lstm(dec_embedding,
#                                              initial_state=dec_states_inputs)
#     dec_states = [state_h, state_c]
#     dec_outputs = dec_dense(dec_outputs)
#     dec_model = Model(
#         inputs=[dec_inputs] + dec_states_inputs,
#         outputs=[dec_outputs] + dec_states)
#     print('Inference decoder:')
#     dec_model.summary()
#     print('Inference encoder:')
#     enc_model = Model(inputs=enc_inputs, outputs=enc_states)
#     enc_model.summary()
#     return enc_model, dec_model
#
#
# enc_model, dec_model = make_inference_models()


def str_to_tokens(sentence: str):
    # convert input string to lowercase,
    # then split it by whitespaces
    words = sentence.lower().split()
    # and then convert to a sequence
    # of integers padded with zeros
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')
        if result != '':
            tokens_list.append(result)
    return pad_sequences([tokens_list],
                         maxlen=26,
                         padding='post')


def outres(msg):
    prepro1 = msg
    ## prepro1 = "Hello"

    prepro1 = clean_text(prepro1)
    ## prepro1 = "hello"
    txt = str_to_tokens(prepro1)

    ## txt = [[454,0,0,0,.........13]]

    stat = enc_model.predict(txt)

    empty_target_seq = np.zeros((1, 1))
    ##   empty_target_seq = [0]

    empty_target_seq[0, 0] = tokenizer.word_index['start']
    ##    empty_target_seq = [255]

    stop_condition = False
    decoded_translation = ''

    while not stop_condition:

        dec_outputs, h, c = dec_model.predict([empty_target_seq] + stat)
        # decoder_concat_input = dense(dec_outputs)
        ## decoder_concat_input = [0.1, 0.2, .4, .0, ...............]

        # sampled_word_index = np.argmax( decoder_concat_input[0, -1, :] )
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        ## sampled_word_index = [2]

        sampled_word = None
        # append the sampled word to the target sequence
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != 'end':
                    decoded_translation += ' {}'.format(word)
                sampled_word = word
        # repeat until we generate the end-of-sequence word 'end'
        # or we hit the length of answer limit
        if sampled_word == 'end' \
                or len(decoded_translation.split()) \
                > 52:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        ## <SOS> - > hi
        ## hi --> <EOS>
        stat = [h, c]

    return decoded_translation


# --------------------------------------------------------------------------------


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    res_sorted = sorted(res, reverse=True)
    print(res_sorted[:4])
    ERROR_THRESHOLD = 0.30
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        print(r[0], r[1])
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    print(tag)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    if ints == []:
        res = outres(msg)
        return res
    else:
        res = getResponse(ints, intents)
        return res


# Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("ProBot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="48", font="Arial", )

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()

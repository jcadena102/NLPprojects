# encoding: utf-8

from keras.engine.saving import load_model
import pandas as pd
import numpy as np
import argparse
import datetime
import logging
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from nltk.stem import SnowballStemmer


class TextClassifier:
    def __init__(self):
        vToday = datetime.datetime.now()
        vFileName = 'textclassifier_' + datetime.datetime.now().strftime('%Y%m%d %H%M%S') + '.log'
        logging.basicConfig(filename=vFileName,
                            filemode='w', format= ' [%(asctime)s] [%(levelname)s] %(message)s',
                            level=logging.INFO)
        
        
        parser = argparse.ArgumentParser(description='Process the inputs')
        parser.add_argument('--max_len', help='maximun length of text sequences', default=150)
        parser.add_argument('--max_words', help='vocabulary length', default=1000)
        parser.add_argument('--output_dim', help='maximun length of embedding vector', default=150)
        parser.add_argument('--neurons_lstm', help='number of neurons in LSTM layer', default=80)
        parser.add_argument('--neurons_linear', help='number of neurons in Dense layer', default=1)
        parser.add_argument('--activation_function', help='hiden activation function', default='relu')
        parser.add_argument('--droput_rate', help='rate for Droput layers', default=0.5)
        parser.add_argument('--output_function', help='output function in final Dense layer', default='sigmoid')
        parser.add_argument('--data_path', help='complete path to data csv file')
        parser.add_argument('--output_path', help='complete path to save classification result', default='classification.csv')
        parser.add_argument('--target_column', help='name of the target column in data file', required=True)
        parser.add_argument('--text_column', help='name of the column with text in data file', required=True)
        parser.add_argument('--language', help='Language of text', default='english')
        parser.add_argument('--mode', help='use of the classifier train or inference', default='train')
        parser.add_argument('--path_model', help='complete path to save/load model', default='model')
        
        args = parser.parse_args()
        
        
        self.max_len = int(args.max_len)
        self.max_words = int(args.max_words)
        self.output_dim = int(args.output_dim)
        self.neurons_lstm = int(args.neurons_lstm)
        self.neurons_linear = int(args.neurons_linear)
        self.activation_function = args.activation_function
        self.droput_rate = float(args.droput_rate)
        self.output_function = args.output_function
        self.data_path = args.data_path
        self.target_column = str(args.target_column)
        self.text_column = str(args.text_column)
        self.language = str(args.language)
        self.mode = str(args.mode)
        self.path_model = args.path_model
        self.output_path = args.output_path
        
        
        

    def __matthews_correlation(self, y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = K.sum(y_pos * y_pred_pos)
        tn = K.sum(y_neg * y_pred_neg)

        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)

        numerator = (tp * tn - fp * fn)
        denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / (denominator + K.epsilon())


    def define_model(self):
        inputs = Input(name= 'inputs',shape=[self.max_len])
        layer = Embedding(self.max_words, self.output_dim, input_length=[self.max_len])(inputs)
        layer = LSTM(self.neurons_lstm)(layer)
        layer = Dense(self.neurons_linear, name='FC1')(layer)
        layer = Activation(self.activation_function)(layer)
        layer = Dropout(self.droput_rate)(layer)
        layer = Dense(self.neurons_linear ,name='out_layer')(layer)
        layer = Activation(self.output_function)(layer)
        model = Model(inputs=inputs,outputs=layer)
        return model
    
    def __get_data_train(self, return_x: bool = True):
        
        df = pd.read_csv(self.data_path, encoding='latin-1')
        
        if return_x is True:
            return df[self.text_column]
        else:
            y = df[self.target_column]
            return to_categorical(y)
        
    def __get_data_inference(self):
        df = pd.read_csv(self.data_path, encoding='latin-1')
        return df[self.text_column]
    
        
            
    def preparate_data_train(self, return_train: bool = True):
        stemmer = SnowballStemmer(self.language)
        X = self.__get_data_train()
        Y = self.__get_data_train(return_x=False,)
        
        for i, _ in enumerate(X):
            X[i] = stemmer.stem(X[i])
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
        
        tok = Tokenizer(num_words=self.max_words)
        tok.fit_on_texts(X_train)
        if return_train is True:
            sequences = tok.texts_to_sequences(X_train)
            sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
            return (Y_train, sequences_matrix)
        
        else:
            test_sequences = tok.texts_to_sequences(X_test)
            test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=self.max_len)
            return (Y_test, test_sequences_matrix)
    
    def preparate_data_inferece(self):
        stemmer = SnowballStemmer(self.language)
        X = self.__get_data_inference()
        self.X_data = X
        for i, _ in enumerate(X):
            X[i] = stemmer.stem(X[i])
        tok = Tokenizer(num_words=self.max_words)
        tok.fit_on_texts(X)
        sequences = tok.texts_to_sequences(X)
        return sequence.pad_sequences(sequences, maxlen=self.max_len)
        
            
        
        
        
    
       
    def create_model(self):
        model = self.define_model()
        model.compile(loss=self.__matthews_correlation, optimizer=RMSprop(), metrics=['accuracy'])
        return model
    
    def train_model(self):
        model = self.create_model()
        train_data = self.preparate_data_train()
        model.fit(train_data[1], train_data[0], batch_size=128,epochs=10,
                  validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
        return model
    
    def main(self):
        if self.mode == 'train':
            
            model = self.train_model()
            print("Model creation completed")
            logging.info("Model creation completed")
            test_data = self.preparate_data_train(return_train=False)
            accr = model.evaluate(test_data[1], test_data[0])
            print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
            logging.info('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
            model.save(self.path_model)
            print('The model was saved as {}'.format(self.path_model))
            logging.info('The model was saved as {}'.format(self.path_model))
            
        if self.mode = 'inference':
            model = load_model(self.path_model)
            print('The model was loaded')
            logging.info('The model was loaded')
            input_data = self.preparate_data_inferece()
            print('The data was loaded')
            logging.info('The data was loaded')
            result = model.predict(input_data)
            print('Prediction was made it')
            logging.info('Prediction was made it')
            export = pd.DataFrame() 
            export['Original Text'] = self.X_data
            export['Classification'] = result
            export.to_csv(self.output_path, sep=';', decimal=',')
            print('Results was saved as {}'.format(self.output_path))
            logging.info('Results was saved as {}'.format(self.output_path))
            
            
        else:
            logging.error('{} is not a valid value'.format(self.mode))
            raise ValueError('{} is not a valid value'.format(self.mode))
            
                        
if __name__ == '__main__':
    print('The process began at {}'.format(str(datetime.datetime.now())))
    model = TextClassifier()
    model.main()
    print('The process ended at {}'.format(str(datetime.datetime.now())))
    



    
    
        
    
        
        
        
            
            
            
        
        
        
        
        


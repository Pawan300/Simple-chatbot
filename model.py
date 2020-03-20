from preprocessing import Preprocessing
import numpy as np
import tensorflow as tf, argparse
import tflearn, os, nltk
import pickle as pkl
import random


class Model:
    def __init__(self):
        self.training_data = None
        self.output_data = None
        self.model = None
        self.dictionary = None
        self.labels = None
        self.data = None

    def define_model(self):
        tf.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(self.training_data[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.output_data[0]), activation="softmax")
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)

    def training(self, model_path):

        self.model.fit(self.training_data, self.output_data, n_epoch=1000, batch_size=8, show_metric=True)
        self.model.save(model_path)

    def prepare_data(self, data_path, parameter_path):
        p = Preprocessing()
        p.loading_data(data_path)
        self.data = p.data
        try:
            with open(parameter_path, 'rb') as file:
                self.training_data, self.output_data, self.dictionary, self.labels = pkl.load(file)
        except:
            tags, patterns, words, labels = p.extract_data()
            self.dictionary = p.create_dictionary(words)
            self.training_data, self.output_data = p.create_bow(patterns, self.dictionary, labels, tags)

            with open(parameter_path, 'wb') as file:
                pkl.dump((self.training_data, self.output_data, self.dictionary, labels), file)

    def chat_with_bot(self, model_path):
        self.model.load(model_path)

        def bag_of_words(input, words):
            ps = nltk.stem.PorterStemmer()
            tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            input = tokenizer.tokenize(input)
            bow = [0] * len(words)
            for i in input:
                bow[words.index(ps.stem(i.lower()))] = 1

            return bow

        while True:

            try:
                user_input = input("User :")
                if user_input == 'quit':
                    break

                prediction = self.model.predict([bag_of_words(user_input, self.dictionary)])
                result = np.argmax(prediction)
                tag = self.labels[result]

                for i in self.data['intents']:
                    if i["tag"] == tag:
                        print(random.choice(i["responses"]))
            except:
                print("I am still learning, Sorry to disappoint you")


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d", "--data_path", dest="data_path", help="path to data file", default="data/intents.json")
    parser.add_argument("--o", "--model_path", dest="model_path", default="model/")
    parser.add_argument("--sp", "--save_parameter", dest="save_parameter", default="parameter/parameter.pkl")

    args = parser.parse_args()
    return args


def main():
    m = Model()
    args = argument_parser()
    m.prepare_data(args.data_path, args.save_parameter)
    m.define_model()
    if os.path.isfile(args.model_path + "model_chatbot.data-00000-of-00001"):
        print('*' * 20, "Lets Chat with bot", '*' * 20)
        m.chat_with_bot(args.model_path + "model_chatbot")
    else:
        print('*' * 20, "Training Starting", '*' * 20)
        m.training(args.model_path + "model_chatbot")
        print('*' * 20, "Training Complete", '*' * 20)


if __name__ == '__main__':
    main()

import nltk, json, argparse


class Preprocessing:

    def __init__(self):
        self.data = None
        self.ps = nltk.stem.PorterStemmer()

    def loading_data(self, path):
        with open(path, 'r') as file:
            self.data = json.load(file)

    def extract_data(self):
        tags = []
        patterns = []
        words = []
        labels = []
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

        for intent in self.data["intents"]:
            for pattern in intent["patterns"]:
                word = tokenizer.tokenize(pattern)
                words.extend(word)
                patterns.append(word)
                tags.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])
        labels = sorted(labels)

        return tags, patterns, words, labels

    def create_dictionary(self, words):
        words = sorted(list(set([self.ps.stem(i.lower()) for i in words])))
        return words

    def create_bow(self, words, dictionary, labels, tags):
        bow = []
        for word in words:
            temp = [0] * len(dictionary)
            for w in word:
                index = dictionary.index(self.ps.stem(w.lower()))
                temp[index] = 1
            bow.append(temp)

        bow_label = []
        for t in tags:
            temp = [0] * len(labels)
            temp[labels.index(t)] = 1
            bow_label.append(temp)

        return bow, bow_label


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d", "--data_path", dest="data_path", help="path to data file", default="data/intents.json")

    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    p = Preprocessing()
    p.loading_data(args.data_path)
    tags, patterns, words, labels = p.extract_data()
    dictionary = p.create_dictionary(words)
    bow, bow_label = p.create_bow(patterns, dictionary, labels, tags)
    print("dictionary : ", dictionary)
    print("tags : ", tags)
    print("patterns : ", patterns)
    print("words : ", words)
    print("labels : ", labels)
    print("bow :", bow)
    print("bow label :", bow_label)


if __name__ == "__main__":
    main()

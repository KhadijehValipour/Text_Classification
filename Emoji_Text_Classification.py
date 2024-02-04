
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from argparse import ArgumentParser

class EmojiTextClassifier:

    def __init__(self, dim):
        self.dim = dim

    def load_dataset(self, file_path):
        df = pd.read_csv(file_path)
        X = np.array(df["sentence"])
        Y = np.array(df["label"], dtype=int)
        return X, Y

    def label_to_emoji(self, label):
        emojis = ["💖", "⚾️", "😄", "😔", "🍴"]
        return emojis[label]

    def load_feature_vectors(self, vectors_file):
        self.word_vector = {}
        for line in vectors_file:
            line = line.strip().split(" ")
            word = line[0]
            vector = np.array(line[1:], dtype=np.float64)
            self.word_vector[word] = vector

    def sentence_to_feature_vectors_avg(self, sentence):
        try:
            sentence = sentence.lower()
            words = sentence.strip().split(" ")
            sum_vectors = np.zeros((self.dim,))
            for word in words:
                sum_vectors += self.word_vector[word]

            avg_vectors = sum_vectors / len(words)
            return avg_vectors
        except Exception as e:
            print(sentence)
            print(e)

    def load_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(5, input_shape=(self.dim,), activation="softmax")
        ])

    def train(self, X_train, Y_train):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        output = self.model.fit(X_train, Y_train, epochs=280)
        return output

    def test(self, X_test, Y_test):
        result = self.model.evaluate(X_test, Y_test)
        return result

    def inference(self, test_sentence):
        sentence = self.sentence_to_feature_vectors_avg(test_sentence)
        sentence = np.array([sentence])
        result = self.model.predict(sentence)
        y_pred = np.argmax(result)
        label = self.label_to_emoji(y_pred)
        return label

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dimension", type=int,
                        help="Dimensionality of word vectors (50/100/200/300):",
                        default=100)
    parser.add_argument("--sentence", type=str,
                        help="The sentence that is in the class category",
                        default="I am very well today")
    args = parser.parse_args()

    X_train_avg = []
    X_test_avg = []

    text_classifier = EmojiTextClassifier(args.dimension)
    X_train, Y_train = text_classifier.load_dataset("dataset/train.csv")
    X_test, Y_test = text_classifier.load_dataset("dataset/test.csv")
    text_classifier.load_feature_vectors(open(f"glove.6B/glove.6B.{args.dimension}d.txt", encoding="utf-8"))


    for x_train in X_train:
        X_train_avg.append(text_classifier.sentence_to_feature_vectors_avg(x_train))
    X_train_avg = np.array(X_train_avg)

    for x_test in X_test:
        X_test_avg.append(text_classifier.sentence_to_feature_vectors_avg(x_test))
    X_test_avg = np.array(X_test_avg)

    Y_train_one_hot = tf.keras.utils.to_categorical(Y_train, num_classes=5)
    Y_test_one_hot = tf.keras.utils.to_categorical(Y_test, num_classes=5)

    text_classifier.load_model()
    output = text_classifier.train(X_train_avg, Y_train_one_hot)

    plt.plot(output.history['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"Train accuracy {args.dimension}d-Dropout")
    plt.savefig(f"assents/accuracy-{args.dimension}-Dropout.png")
    plt.show()

    plt.plot(output.history["loss"])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Train loss {args.dimension}d-Dropout")
    plt.savefig(f"assents/loss-{args.dimension}-Dropout.png")
    plt.show()


    result = text_classifier.test(X_test_avg, Y_test_one_hot)
    print("test loss : ", result[0], "test accuracy : ", result[1])

    # Inference
    start = time.time()
    for i in range(100):
        result = text_classifier.inference(args.sentence)
    inference_time = time.time() - start
    inference_avg_time = inference_time / 100
    print("result:", result, "Inference average time:", inference_avg_time)

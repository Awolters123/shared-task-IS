# Importing libraries
import pandas as pd
import random as python_random
import numpy as np
import emoji
import argparse
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  classification_report
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import plot_confusion_matrix
import tensorflow
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")
# Make reproducible as much as possible
np.random.seed(1234)
tensorflow.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    """This function creates all command arguments, for data input,
    model selection, and custom parameters, please see the help sections
    for a detailed description."""
    parser = argparse.ArgumentParser()
    # Data input arguments
    parser.add_argument("-d1", "--data_file1", default="train_all_tasks.csv",
                        type=str, help="Dataset to train model with, default "
                                       "is the SemEval 2022 sexism dataset")
    parser.add_argument("-d2", "--data_file2", default="EXIST2021_merged.csv",
                        type=str, help="Extra dataset to train the model "
                                       "with, has to have compatible labels "
                                       "with the first dataset")
    parser.add_argument("-d_text", "--dev_text", type=str,
                        help="Codalab dev set text")
    parser.add_argument("-d_label", "--dev_label", type=str,
                        help="Codalab dev set labels")
    parser.add_argument("-t", "--test_file", type=str,
                        help="Codalab test set (only the text and no labels)")
    parser.add_argument("-m", "--mode", default="concat", type=str,
                        help="This argument sets the data merge option, "
                             "you can choose between concatenating (concat) "
                             "or shuffling (shuffle), default is concat")
    # Model arguments
    parser.add_argument("-tf", "--transformer", default="GroNLP/hateBERT",
                        type=str,
                        help="this argument takes the pretrained language "
                             "model URL from HuggingFace default is HateBERT, "
                             "please visit HuggingFace for full URL")
    # Parameter arguments
    parser.add_argument("-lr", "--learn_rate", default=5e-5, type=float,
                        help="Set a custom learn rate for "
                             "the pretrained language model, default is 5e-5")
    parser.add_argument("-bs", "--batch_size", default=8, type=int,
                        help="Set a custom batch size for "
                             "the pretrained language model, default is 8")
    parser.add_argument("-sl", "--sequence_length", default=100, type=int,
                        help="Set a custom maximum sequence length for "
                             "the pretrained language model, default is 100")
    parser.add_argument("-epoch", "--epochs", default=1, type=int,
                        help="This argument selects the amount of epochs "
                             "to run the model with, default is 1 epoch")

    args = parser.parse_args()
    return args


def read_data(d1, d2):
    """Reading in both datasets and returning them as pandas dataframes
    with only the text and labels."""
    # read in data to pandas
    df1 = pd.read_csv(d1)
    df2 = pd.read_csv(d2)

    # drop columns we don't use
    df1 = df1.drop(columns=['rewire_id', 'label_category', 'label_vector'])
    df2 = df2.drop(columns=['test_case', 'id', 'source', 'language',
                            'category', 'Unnamed: 0'])

    # convert labels to numerical value (non sexist = 0, sexist = 1 )
    df1.loc[df1.label_sexist == 'not sexist', 'label_sexist'] = 0
    df1.loc[df1.label_sexist == 'sexist', 'label_sexist'] = 1

    df1.columns = ['text', 'label']
    df2.columns = ['text', 'label']
    return df1, df2


def preprocess(text):
    """Removes hashtags and converts links to [URL] and usernames starting
    with @ to [USER], it also converts emojis to their textual form."""
    documents = []
    for instance in text:
        instance = re.sub(r'@([^ ]*)', '[USER]', instance)
        instance = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.'
                          r'([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
                          '[URL]', instance)
        instance = emoji.demojize(instance)
        instance = instance.replace('#', '')
        documents.append(instance)
    return documents


def train_transformer(lm, epoch, bs, lr, sl, X_train, Y_train, X_dev, Y_dev):
    """This function takes as input the train file, dev file, model name,
    and parameters. It trains the model with the specified parameters and
    returns the trained model."""
    print("Training model: {}\nWith parameters:\nLearn rate: {}, "
          "Batch size: {}\nEpochs: {}, Sequence length: {}"
          .format(lm, lr, bs, epoch, sl))

    # Selecting the correct tokenizer for the model, and selecting the model
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm,
                                                                 num_labels=2,
                                                                 from_pt=True)

    # Tokenzing the train and dev texts
    tokens_train = tokenizer(X_train, padding=True, max_length=sl,
                             truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=sl,
                           truncation=True, return_tensors="np").data

    # Setting the loss function for binary task and optimization function
    loss_function = BinaryCrossentropy(from_logits=True)
    optim = Adam(learning_rate=lr)

    # Early stopping
    early_stopper = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True,
        mode="auto")
    # Encoding the labels with sklearns LabelBinazrizer
    encoder = LabelBinarizer()
    Y_train = encoder.fit_transform(Y_train)
    Y_dev = encoder.fit_transform(Y_dev)
    Y_train_bin = np.hstack((1 - Y_train, Y_train))
    Y_dev_bin = np.hstack((1 - Y_dev, Y_dev))

    # Compiling the model and training it with the given parameter settings
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=epoch,
              batch_size=bs, validation_data=(tokens_dev, Y_dev_bin),
              callbacks=[early_stopper])
    return model


def test_transformer(lm, epoch, bs, lr, sl, model, X_test, Y_test, ident):
    """This function takes as input the trained transformer model, name of
    the model, parameters, and the test files, and predicts the labels
    for the test set and returns the accuracy score with a summarization
    of the model."""
    print(
        "Testing model: {} on {} set\nWith parameters:\nLearn rate: {}, "
        "Batch size: {}\nEpochs: {}, Sequence length: {}"
        .format(lm, ident, lr, bs, epoch, sl))

    # Selecting the correct tokenizer for the model,
    # and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predicitions on the test set and converting
    # the logits to sigmoid probabilities (binary)
    Y_pred = model.predict(tokens_test)["logits"]
    prob = tensorflow.nn.softmax(Y_pred)

    # converting gold labels with LabelBinarizer
    encoder = LabelBinarizer()
    Y_test = encoder.fit_transform(Y_test)
    Y_test_bin = np.hstack((1 - Y_test, Y_test))

    # Converting the predicitions and gold set
    # to their original numerical label value
    pred = np.argmax(prob, axis=1)
    gold = np.argmax(Y_test_bin, axis=1)

    # Printing classification report (rounding on 3 decimals)
    print("Classification Report on {} set:\n{}"
          .format(ident, classification_report(gold, pred, digits=3)))
    return gold, pred


def predict_dev(lm, sl, model, df_text, df_label):
    """This function takes as input the transformer model,
    maximum sequence length, Codalab dev set text and labels, and predicts
    and returns the classification report."""
    print('Running model on Codalab dev set')
    # pre-processing text
    X_test = preprocess(df_text['text'].tolist())
    Y_test = df_label['label'].tolist()
    # Selecting the correct tokenizer for the model,
    # and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predictions on the test set and converting the logits
    Y_pred = model.predict(tokens_test)["logits"]
    prob = tensorflow.nn.softmax(Y_pred)
    pred = np.argmax(prob, axis=1)

    predictions = []
    for n in pred:
        if n == 0:
            predictions.append('not sexist')
        elif n == 1:
            predictions.append('sexist')

    return classification_report(Y_test, predictions, digits=4)


def predict_test(lm, sl, model, df_test):
    """This function takes as input an unseen test file without labels,
    and predict the labels and returns the predicted labels as a .csv file in
    the correct Codalab format."""
    # pre-processing text
    print('Running model on Codalab test set (without labels)')
    X_test = preprocess(df_test['text'].tolist())

    # Selecting the correct tokenizer for the model,
    # and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predictions on the test set
    # and converting the logits to softmax probabilities
    Y_pred = model.predict(tokens_test)["logits"]
    prob = tensorflow.nn.softmax(Y_pred)

    # Converting the predictions and exporting it
    pred = []
    for n in np.argmax(prob, axis=1):
        if n == 0:
            pred.append('not sexist')
        elif n == 1:
            pred.append('sexist')

    # Save to csv file
    results = pd.DataFrame()
    results['rewire_id'] = df_test['rewire_id']
    results['label_pred'] = pred
    results.to_csv("test_task_a.csv", index=False)


def main():
    """Main function to train and finetune pretrained language models"""
    # Create the command arguments for the script
    args = create_arg_parser()

    # Creating parameter variables
    lr = args.learn_rate
    bs = args.batch_size
    sl = args.sequence_length
    ep = args.epochs

    # Reading data
    ori, d2 = read_data(args.data_file1, args.data_file2)
    ori_train, ori_dev = train_test_split(ori, test_size=0.2,
                                          random_state=1234)

    X_dev = preprocess(ori_dev['text'].tolist())
    Y_dev = ori_dev['label'].tolist()

    # Concat two datasets, with 80% of original and 100% of Exist2021
    ori_concat = pd.concat([ori_train, d2], axis=0)
    ori_concat_shuffled = ori_concat.sample(frac=1)

    # Checking if datasets are concatenated or shuffled
    if args.mode == "concat":
        X_train = preprocess(ori_concat['text'].tolist())
        Y_train = ori_concat['label'].tolist()

    elif args.mode == "shuffle":
        X_train = preprocess(ori_concat_shuffled['text'].tolist())
        Y_train = ori_concat_shuffled['label'].tolist()

    # Running model
    print('Running HateBERT for task A:')
    HateBERT = train_transformer(args.transformer, ep, bs, lr, sl,
                                 X_train, Y_train, X_dev, Y_dev)
    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_dev, Y_dev, "dev")
    if args.dev_text and args.dev_label:
        # Reading in and predicting on test set
        df_dev_text = pd.read_csv(args.dev_text)
        df_dev_label = pd.read_csv(args.dev_label)
        print(predict_dev(args.transformer, sl, HateBERT,
                          df_dev_text, df_dev_label))

    if args.test_file:
        test_set = pd.read_csv(args.test_file)
        predict_test(args.transformer, sl, HateBERT, test_set)


if __name__ == '__main__':
    main()

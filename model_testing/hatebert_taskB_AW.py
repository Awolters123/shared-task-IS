# Importing libraries
import pandas as pd
import random as python_random
import numpy as np
import emoji
import argparse
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
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
    '''This function creates all command arguments, for data input, model selection, and custom parameters,
    please see the help section for a detailed description.'''
    parser = argparse.ArgumentParser()
    # Data input arguments
    parser.add_argument("-d1", "--data_file1", default="train_all_tasks.csv", type=str,
                        help="Dataset to train the model with, default is the SemEval 2022 sexism dataset")
    parser.add_argument("-d2", "--data_file2", type=str,
                        help="Extra dataset to augment the training set (optional)")
    parser.add_argument("-d_text", "--dev_file_text", type=str,
                        help="Test file text, which will be used for prediction")
    parser.add_argument("-d_label", "--dev_file_label", type=str,
                        help="Test file labels, which will be used to evaluate the model")
    parser.add_argument("-T", "--test_file_codalab", type=str,
                        help="Test file without labels")

    # Model arguments
    parser.add_argument("-tf", "--transformer", default="GroNLP/hateBERT", type=str,
                        help="this argument takes the pretrained language model link from HuggingFace, "
                             "default is HateBERT")
    parser.add_argument("-m", "--mode", default="concat", type=str,
                        help="This argument sets the data merge option, you can choose between concatenating (concat) "
                             "or shuffling (shuffle), default is concat")
    # Parameter arguments
    parser.add_argument("-lr", "--learn_rate", default=5e-5, type=float,
                        help="Set a custom learn rate for the pretrained language model, default is 5e-5")
    parser.add_argument("-bs", "--batch_size", default=8, type=int,
                        help="Set a custom batch size for the pretrained language model, default is 8")
    parser.add_argument("-sl", "--sequence_length", default=100, type=int,
                        help="Set a custom maximum sequence length for the pretrained language model, default is 100")
    parser.add_argument("-epoch", "--epochs", default=1, type=int,
                        help="This argument selects the amount of epochs to run the model with, default is 1 epoch")

    args = parser.parse_args()
    return args


def read_data(dataset, type):
    '''Reading in the dataset and returning it as pandas dataframes
    with only the text and label.'''
    # read in data to pandas
    df = pd.read_csv(dataset)

    # drop columns we don't use
    if type == 'codalab':
        df = df.loc[df['label_sexist'] == 'sexist']
    elif type == 'exist2021':
        df = df.loc[df['sexist'] == 1]

    # convert labels to numerical values
    df.loc[df.label_category == '1. threats, plans to harm and incitement', 'label_category'] = 0
    df.loc[df.label_category == '2. derogation', 'label_category'] = 1
    df.loc[df.label_category == '3. animosity', 'label_category'] = 2
    df.loc[df.label_category == '4. prejudiced discussions', 'label_category'] = 3
    # converting column names
    df_sexist = df[['text', 'label_category']].copy()
    df_sexist.columns = ['text', 'label']
    return df_sexist


def preprocess(text):
    '''Removes hashtags and converts links to [URL] and usernames starting with @ to [USER],
    it also converts emojis to their textual form.'''
    documents = []
    for instance in text:
        instance = re.sub(r'@([^ ]*)', '[USER]', instance)
        instance = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '[URL]', instance)
        instance = emoji.demojize(instance)
        instance = instance.replace('#', '')
        documents.append(instance)
    return documents


def train_transformer(lm, epoch, bs, lr, sl, X_train, Y_train, X_dev, Y_dev):
    """This function takes as input the train file, dev file, transformer model name, and parameters.
    It trains the model with the specified parameters and returns the trained model."""
    print("Training model: {}\nWith parameters:\nLearn rate: {}, Batch size: {}\nEpochs: {}, Sequence length: {}"
          .format(lm, lr, bs, epoch, sl))

    # Selecting the correct tokenizer for the model, and selecting the model
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=4, from_pt=True)

    # Tokenzing the train and dev texts
    tokens_train = tokenizer(X_train, padding=True, max_length=sl,
                             truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=sl,
                           truncation=True, return_tensors="np").data

    # Setting the loss function for binary task and optimization function
    loss_function = CategoricalCrossentropy(from_logits=True)
    optim = Adam(learning_rate=lr)

    # Early stopping
    early_stopper = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True,
                                                             mode="auto")
    # Encoding the labels with sklearns LabelBinazrizer
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Compiling the model and training it with the given parameter settings
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=epoch,
              batch_size=bs, validation_data=(tokens_dev, Y_dev_bin), callbacks=[early_stopper])
    return model


def test_transformer(lm, epoch, bs, lr, sl, model, X_test, Y_test, ident):
    """This function takes as input the trained transformer model, name of the model, parameters, and the test files,
    and predicts the labels for the test set and returns the accuracy score with a summarization of the model."""
    print(
        "Testing model: {} on {} set\nWith parameters:\nLearn rate: {}, Batch size: {}\nEpochs: {}, Sequence length: {}"
        .format(lm, ident, lr, bs, epoch, sl))

    # Selecting the correct tokenizer for the model, and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predicitions on the test set and converting the logits to softmax probabilities (multi-class)
    Y_pred = model.predict(tokens_test)["logits"]
    prob = tensorflow.nn.softmax(Y_pred)

    # Converting the predicitions to the original numerical label value
    pred = np.argmax(prob, axis=1)

    # Printing classification report (rounding on 3 decimals)
    print("Classification Report on {} set:\n{}".format(ident, classification_report(Y_test, pred, digits=4)))
    return Y_test, pred


def predict_dev(lm, sl, model, df_dev, dev_labels):
    """This function takes as input an unseen test file without labels, and predict the labels and returns the
    predicted labels as a .csv file in the correct Codalab format."""
    # pre-processing text
    print('Running model on Codalab dev set')
    X_dev = preprocess(df_dev['text'].tolist())
    Y_dev = dev_labels['label']

    # Selecting the correct tokenizer for the model, and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_dev, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predictions on the test set and converting the logits to softmax probabilities
    Y_pred = model.predict(tokens_test)["logits"]
    prob = tensorflow.nn.softmax(Y_pred)

    # Converting the predictions and exporting it
    pred = []
    for n in np.argmax(prob, axis=1):
        if n == 0:
            pred.append('1. threats, plans to harm and incitement')
        elif n == 1:
            pred.append('2. derogation')
        elif n == 2:
            pred.append('3. animosity')
        elif n == 3:
            pred.append('4. prejudiced discussions')

    # Save to csv file
    results = pd.DataFrame()
    results['rewire_id'] = df_dev['rewire_id']
    results['label_pred'] = pred
    results.to_csv("EXAMPLE_SUBMISSION_dev_task_b.csv", index=False)
    
    return classification_report(Y_dev, pred, digits=4)


def predict_test(lm, sl, model, df_test):
    """This function takes as input an unseen test file without labels, and predict the labels and returns the
    predicted labels as a .csv file in the correct Codalab format."""
    # pre-processing text
    print('Running model on Codalab test set (without labels)')
    X_test = preprocess(df_test['text'].tolist())

    # Selecting the correct tokenizer for the model, and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predictions on the test set and converting the logits to softmax probabilities
    Y_pred = model.predict(tokens_test)["logits"]
    prob = tensorflow.nn.softmax(Y_pred)

    # Converting the predictions and exporting it
    pred = []
    for n in np.argmax(prob, axis=1):
        if n == 0:
            pred.append('1. threats, plans to harm and incitement')
        elif n == 1:
            pred.append('2. derogation')
        elif n == 2:
            pred.append('3. animosity')
        elif n == 3:
            pred.append('4. prejudiced discussions')

    # Save to csv file
    results = pd.DataFrame()
    results['rewire_id'] = df_test['rewire_id']
    results['label_pred'] = pred
    results.to_csv("test_task_b.csv", index=False)


def main():
    '''Main function to train and finetune pretrained language models'''
    # Create the command arguments for the script
    args = create_arg_parser()

    # Creating parameter variables
    lr = args.learn_rate
    bs = args.batch_size
    sl = args.sequence_length
    ep = args.epochs

    # Reading data
    data = read_data(args.data_file1, 'codalab')
    df_train, df_dev = train_test_split(data, test_size=0.2, random_state=1234)

    X_dev = preprocess(df_dev['text'].tolist())
    Y_dev = df_dev['label'].tolist()

    # Running model
    if args.data_file2:
        # Reading data
        data2 = read_data(args.data_file2, 'exist2021')
        ori_concat = pd.concat([df_train, data2], axis=0)
        ori_concat_shuffled = ori_concat.sample(frac=1, random_state=1234)

        if args.mode == "concat":
            X_train2 = preprocess(ori_concat['text'].tolist())
            Y_train2 = ori_concat['label'].tolist()

        else:
            X_train2 = preprocess(ori_concat_shuffled['text'].tolist())
            Y_train2 = ori_concat_shuffled['label'].tolist()

        print('Running HateBERT for task B with extra Exist2021 data in {} mode'.format(args.mode))
        HateBERT = train_transformer(args.transformer, ep, bs, lr, sl, X_train2, Y_train2, X_dev, Y_dev)
        test_transformer(args.transformer, ep, bs, lr, sl, HateBERT, X_dev, Y_dev, "dev")
        if args.dev_file_text and args.dev_file_label:
            # Reading in unlabeled test set
            df_dev_text = pd.read_csv(args.dev_file_text)
            df_dev_labels = pd.read_csv(args.dev_file_label)
            print(predict_dev(args.transformer, sl, HateBERT, df_dev_text, df_dev_labels))

        if args.test_file_codalab:
            df_test_codalab = pd.read_csv(args.test_file_codalab)
            predict_test(args.transformer, sl, HateBERT, df_test_codalab)

    else:
        #Reading data
        X_train1 = preprocess(df_train['text'].tolist())
        Y_train1 = df_train['label'].tolist()

        print('Running HateBERT for task B')
        HateBERT = train_transformer(args.transformer, ep, bs, lr, sl, X_train1, Y_train1, X_dev, Y_dev)
        test_transformer(args.transformer, ep, bs, lr, sl, HateBERT, X_dev, Y_dev, "dev")
        if args.dev_file_text and args.dev_file_label:
            # Reading in unlabeled test set
            df_dev_text = pd.read_csv(args.dev_file_text)
            df_dev_labels = pd.read_csv(args.dev_file_label)
            print(predict_dev(args.transformer, sl, HateBERT, df_dev_text, df_dev_labels))

        if args.test_file_codalab:
            df_test_codalab = pd.read_csv(args.test_file_codalab)
            predict_test(args.transformer, sl, HateBERT, df_test_codalab)


if __name__ == '__main__':
    main()

# Importing libraries
import pandas as pd
import random as python_random
import numpy as np
import emoji
import argparse
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.keras.optimizers import Adam
import tensorflow
import re
from pathlib import Path as path
# from collections import Counter
import warnings
# import matplotlib.pyplot as plt


RANDOM_STATE = 1234

warnings.filterwarnings("ignore")
# Make reproducible as much as possible
np.random.seed(RANDOM_STATE)
tensorflow.random.set_seed(RANDOM_STATE)
python_random.seed(RANDOM_STATE)


def create_arg_parser():
    '''This function creates all command arguments, for data input, model selection, and custom parameters,
    please see the help section for a detailed description.'''
    parser = argparse.ArgumentParser()
    # Data input arguments
    parser.add_argument("-d1", "--data_file1", default="data/train_all_tasks.csv", type=str,
                        help="Dataset to train the model with, default is the SemEval 2022 sexism dataset")
    parser.add_argument("-t", "--test_file", type=str,
                        help="Test file, which will be used to evaluate the model")
    parser.add_argument("-tn", "--test_name", default="no_name", type=str,
                        help="Test file name, which will be used to store multiple test files.")

    # Easy-to-learn data
    parser.add_argument("--easy", action="store_true",
                        help="Run the model using easy-to-learn data.")
    parser.add_argument("-d3", "--data_file3", default="data/train.tsv", type=str,
                        help="Easy-to-learn train data from data cartography")
    parser.add_argument("-d4", "--data_file4", default="data/dev.tsv", type=str,
                        help="Easy-to-learn dev data from data cartography")

    # Semeval data only
    parser.add_argument("--semeval", action="store_true",
                        help="Run the model for task A, using the semeval dataset.")

    # Padding max length
    parser.add_argument("--max_len", action="store_true",
                        help="Run the model using max padding length based on input.")

    # Task A
    parser.add_argument("-d2", "--data_file2", default="data/EXIST2021_merged.csv", type=str,
                        help="Extra dataset to train the model with, has to have compatible labels with "
                             "the first dataset")
    parser.add_argument("-m", "--mode", default="concat", type=str,
                        help="This argument sets the data merge option, you can choose between concatenating (concat) "
                             "or shuffling (shuffle), default is concat")
    parser.add_argument("--gab_only", action="store_true",
                        help="Run the model using only gab added data.")

    # Task B
    parser.add_argument("--task_b", action="store_true",
                        help="Run the model for task B")

    # Model arguments
    parser.add_argument("-tf", "--transformer", default="GroNLP/hateBERT", type=str,
                        help="this argument takes the pretrained language model link from HuggingFace, "
                             "default is HateBERT")
    # Parameter arguments
    parser.add_argument("-lr", "--learn_rate", default=5e-5, type=float,
                        help="Set a custom learn rate for the pretrained language model, default is 5e-5")
    parser.add_argument("-bs", "--batch_size", default=8, type=int,
                        help="Set a custom batch size for the pretrained language model, default is 8")
    parser.add_argument("-sl", "--sequence_length", default=100, type=int,
                        help="Set a custom maximum sequence length for the pretrained language model, default is 100")
    parser.add_argument("-epoch", "--epochs", default=1, type=int,
                        help="This argument selects the amount of epochs to run the model with, default is 1 epoch")
    parser.add_argument("--no_weight_restore", action="store_false",
                        help="Run the model without weight restore")

    # Save model
    parser.add_argument("--save_model", type=str,
                        help="Save the current model to a file for later use on a test file, "
                        "requires a name to be specified")

    args = parser.parse_args()
    return args


def read_data(d1, task_b, d2=None, gab_only=False):
    '''Reading in the dataset and returning it as pandas dataframes
    with only the text and label.'''
    # read in data to pandas
    df1 = pd.read_csv(d1)

    if task_b:
        # remove non-sexist instances, as these are only relevant for task A
        df1 = df1[df1['label_sexist'] == 'sexist']
        # drop columns we don't use
        df1 = df1.drop(columns=['rewire_id', 'label_sexist', 'label_vector'])

        df1.loc[df1.label_category == '1. threats, plans to harm and incitement', 'label_category'] = 0
        df1.loc[df1.label_category == '2. derogation', 'label_category'] = 1
        df1.loc[df1.label_category == '3. animosity', 'label_category'] = 2
        df1.loc[df1.label_category == '4. prejudiced discussions', 'label_category'] = 3

        # converting column names
        df1.columns = ['text', 'label']

        if d2:
            df2 = pd.read_csv(d2)

            df2 = df2[df2['sexist'] == 1]
            df2.loc[df2.label_category == '1. threats, plans to harm and incitement', 'label_category'] = 0
            df2.loc[df2.label_category == '2. derogation', 'label_category'] = 1
            df2.loc[df2.label_category == '3. animosity', 'label_category'] = 2
            df2.loc[df2.label_category == '4. prejudiced discussions', 'label_category'] = 3

            df2 = df2.drop(columns=['test_case', 'id', 'source', 'language', 'sexist', 'Unnamed: 0'])

            df2.columns = ['text', 'exist_label', 'label']

            return df1, df2
        else:
            return df1

    else:
        # read in data to pandas
        df2 = pd.read_csv(d2)

        if gab_only:
            df2 = df2[df2['source'] == 'gab']

        # drop columns we don't use
        df1 = df1.drop(columns=['rewire_id', 'label_category', 'label_vector'])
        df2 = df2.drop(columns=['test_case', 'id', 'source', 'language', 'category', 'Unnamed: 0'])

        # convert labels to numerical value (non sexist = 0, sexist = 1 )
        df1.loc[df1.label_sexist == 'not sexist', 'label_sexist'] = 0
        df1.loc[df1.label_sexist == 'sexist', 'label_sexist'] = 1

        df1.columns = ['text', 'label']
        df2.columns = ['text', 'label']
        return df1, df2


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


def train_transformer(lm, epoch, bs, lr, sl_train, sl_dev, X_train, Y_train, X_dev, Y_dev, nws, task_b):
    """This function takes as input the train file, dev file, transformer model name, and parameters.
    It trains the model with the specified parameters and returns the trained model."""
    print("\n\nTraining model: {}\nWith parameters:\nLearn rate: {}, Batch size: {}\nEpochs: {}, Sequence length (train; dev): {}; {}\n\n"
          .format(lm, lr, bs, epoch, sl_train, sl_dev))
    pt_state = False

    if lm == "GroNLP/hateBERT":
        pt_state = True
    # Selecting the correct tokenizer for the model, and selecting the model
    tokenizer = AutoTokenizer.from_pretrained(lm)

    num_labels = 1
    if task_b:
        num_labels = 4
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=num_labels, from_pt=pt_state)

    # Tokenzing the train and dev texts
    tokens_train = tokenizer(X_train, padding=True, max_length=sl_train,
                             truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=sl_dev,
                           truncation=True, return_tensors="np").data

    # Setting the loss function for binary task and optimization function
    if task_b:
        loss_function = CategoricalCrossentropy(from_logits=True)
    else:
        loss_function = BinaryCrossentropy(from_logits=True)

    optim = Adam(learning_rate=lr)

    # Early stopping
    early_stopper = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=nws,
                                                             mode="auto")
    if task_b:
        # Encoding the labels with sklearns LabelBinazrizer
        encoder = LabelBinarizer()
        Y_train_bin = encoder.fit_transform(Y_train)
        Y_dev_bin = encoder.fit_transform(Y_dev)
    else:
        # Encoding the labels with sklearns LabelBinazrizer
        encoder = LabelBinarizer()
        # Y_train = encoder.fit_transform(Y_train)
        Y_train_bin = encoder.fit_transform(Y_train)
        # Y_dev = encoder.fit_transform(Y_dev)
        Y_dev_bin = encoder.fit_transform(Y_dev)
        # Y_train_bin = np.hstack((1 - Y_train, Y_train))
        # Y_dev_bin = np.hstack((1 - Y_dev, Y_dev))

    # Compiling the model and training it with the given parameter settings
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=epoch,
              batch_size=bs, validation_data=(tokens_dev, Y_dev_bin), callbacks=[early_stopper])
    return model


# def confmatrix_display(gold, pred, task_b):
#     # plt.rcParams.update({'font size': 12})
#     plt.figure(dpi=1200)

#     cm_display = ConfusionMatrixDisplay(confusion_matrix(gold, pred))
#     cm_display.plot()
#     plt.show()

#     if task_b:
#         plt.xticks(rotation=45, ha='right')


def test_transformer(lm, epoch, bs, lr, sl, model, X_test, Y_test, ident, task_b):
    """This function takes as input the trained transformer model, name of the model, parameters, and the test files,
    and predicts the labels for the test set and returns the accuracy score with a summarization of the model."""
    print(
        "\n\nTesting model: {} on {} set\nWith parameters:\nLearn rate: {}, Batch size: {}\nEpochs: {}, Sequence length: {}\n\n"
        .format(lm, ident, lr, bs, epoch, sl))

    # Selecting the correct tokenizer for the model, and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predicitions on the test set and converting the logits to sigmoid probabilities (binary)
    Y_pred = model.predict(tokens_test)["logits"]
    if task_b:
        prob = tensorflow.nn.softmax(Y_pred)
    else:
        prob = tensorflow.round(tensorflow.nn.sigmoid(Y_pred))

    # converting gold labels with LabelBinarizer
    encoder = LabelBinarizer()
    Y_test_bin = encoder.fit_transform(Y_test)
    # if not task_b:
    #     Y_test_bin = np.hstack((1 - Y_test, Y_test))

    # Converting the predicitions and gold set to their original numerical label value
    if task_b:
        # pred = np.argmax(prob, axis=1)
        # gold = np.argmax(Y_test_bin, axis=1)

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

        gold = []
        for n in np.argmax(Y_test_bin, axis=1):
            if n == 0:
                gold.append('1. threats, plans to harm and incitement')
            elif n == 1:
                gold.append('2. derogation')
            elif n == 2:
                gold.append('3. animosity')
            elif n == 3:
                gold.append('4. prejudiced discussions')

    else:
        # pred = prob
        # gold = Y_test_bin

        pred = []
        for n in prob:
            if n == 0:
                pred.append('not sexist')
            elif n == 1:
                pred.append('sexist')

        gold = []
        for n in Y_test_bin:
            if n == 0:
                gold.append('not sexist')
            elif n == 1:
                gold.append('sexist')

    # Printing classification report (rounding on 3 decimals)
    print("")
    print("Classification Report on {} set:\n{}".format(ident, classification_report(gold, pred, digits=3)))
    # confmatrix_display(gold, pred, task_b)
    print("")
    print(confusion_matrix(gold, pred))
    # return gold, pred


def predict(lm, sl, model, df_test, task_b, test_name):
    """This function takes as input an unseen test file without labels, and predict the labels and returns the
    predicted labels as a .csv file in the correct Codalab format."""
    # pre-processing text
    X_test = preprocess(df_test['text'].tolist())

    # Selecting the correct tokenizer for the model, and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predictions on the test set and converting the logits to sigmoid probabilities (binary)
    Y_pred = model.predict(tokens_test)["logits"]

    if task_b:
        prob = tensorflow.nn.softmax(Y_pred)
        # Converting the predictions
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

        task_name = "task_b"

    else:
        prob = tensorflow.round(tensorflow.nn.sigmoid(Y_pred))
        # Converting the predictions
        pred = []
        for n in prob:
            if n == 0:
                pred.append('not sexist')
            elif n == 1:
                pred.append('sexist')

        task_name = "task_a"

    # Exporting the predictions
    results = pd.DataFrame()
    results['rewire_id'] = df_test['rewire_id']
    results['label_pred'] = pred
    results.to_csv(f"{test_name}-EXAMPLE_SUBMISSION_dev_{task_name}.csv", index=False)


def main():
    '''Main function to train and finetune pretrained language models'''
    # Create the command arguments for the script
    args = create_arg_parser()

    # Creating parameter variables
    lr = args.learn_rate
    bs = args.batch_size
    sl = args.sequence_length
    ep = args.epochs
    nws = args.no_weight_restore

    test_name = args.test_name

    # Reading data
    if args.task_b:
        if args.semeval:
            data = read_data(args.data_file1, args.task_b, gab_only=args.gab_only)
            df_train, df_dev = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)

            X_train = preprocess(df_train['text'].tolist())
            Y_train = df_train['label'].tolist()

            X_dev = preprocess(df_dev['text'].tolist())
            Y_dev = df_dev['label'].tolist()

        else:
            ori, d2 = read_data(args.data_file1, args.task_b, args.data_file2, args.gab_only)
            ori_train, ori_dev = train_test_split(ori, test_size=0.2, random_state=RANDOM_STATE)

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

    elif args.semeval:
        if args.easy:
            df_train = pd.read_csv(args.data_file3, sep='\t')
            df_dev = pd.read_csv(args.data_file4, sep='\t')
        else:
            data, d2 = read_data(args.data_file1, args.task_b, args.data_file2, args.gab_only)
            df_train, df_dev = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)

        X_train = preprocess(df_train['text'].tolist())
        Y_train = df_train['label'].tolist()

        X_dev = preprocess(df_dev['text'].tolist())
        Y_dev = df_dev['label'].tolist()

    else:
        if args.easy:
            print('Easy-to-learn should have been trained with shuffle.')
            df_train = pd.read_csv(args.data_file3, sep='\t')
            df_dev = pd.read_csv(args.data_file4, sep='\t')

            X_dev = preprocess(df_dev['text'].to_list())
            Y_dev = df_dev['label'].to_list()

            X_train = preprocess(df_train['text'].to_list())
            Y_train = df_train['label'].to_list()

        else:
            ori, d2 = read_data(args.data_file1, args.task_b, args.data_file2, args.gab_only)
            ori_train, ori_dev = train_test_split(ori, test_size=0.2, random_state=RANDOM_STATE)

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

    model_name = args.transformer.split("/")[-1]

    if args.task_b:
        print(f'Running {model_name} for task B')
    else:
        print(f'Running {model_name} for task A')

    if args.max_len:
        sl_train = len(max(X_train))
        sl_dev = len(max(X_dev))
    else:
        sl_train = sl
        sl_dev = sl

    model = train_transformer(args.transformer, ep, bs, lr, sl_train, sl_dev, X_train, Y_train, X_dev, Y_dev, nws, args.task_b)
    test_transformer(args.transformer, ep, bs, lr, sl_dev, model, X_dev, Y_dev, "dev", args.task_b)

    if args.save_model:
        # Save the model in ./saved_models
        path('./saved_models').mkdir(parents=True, exist_ok=True)
        model.save_pretrained('saved_models/' + args.save_model)

    if args.test_file:
        # Reading in unlabeled test set
        df_test = pd.read_csv(args.test_file)

        if args.max_len:
            sl = df_test.text.str.len().max()

        predict(args.transformer, sl, model, df_test, args.task_b, test_name)


if __name__ == '__main__':
    main()

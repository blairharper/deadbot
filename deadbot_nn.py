# Neural network for deadbot
import pickle
import numpy as np
import re
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.contrib import seq2seq

def load_data():
    # loads pre-processed data from pickle file
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    # write params to pickle file
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    # load params from pickle file
    return pickle.load(open('params.p', mode='rb'))


def get_available_gpus():
    # checks GPUs available to tensorflow
    local_device_protos = device_lib.list_local_devices()
    return (device.name for device in local_device_protos if device.device_type == 'GPU')


def version_check():
    print(get_available_gpus())
    print(tf.__version__)


def get_inputs():
    # sets placeholders for input tensors & learning rate
    inputs = tf.placeholder(tf.int32, shape=[None, None], name='input')
    targets = tf.placeholder(tf.int32, shape=[None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return inputs, targets, learning_rate


def get_init_cell(batch_size, rnn_size):
    # sets initial lstm cell
    lstm = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=rnn_size)
    drop = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(
            lstm,
            output_keep_prob=0.8)
        ])
    cell = tf.contrib.rnn.MultiRNNCell([drop])
    ini = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(ini, name='initial_state')
    return cell, initial_state


def get_embed(input_data, vocab_size, embed_dim):
    # create embedding layer
    embed = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embeded_input = tf.nn.embedding_lookup(embed, input_data)
    return embeded_input


def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    return outputs, final_state


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    embed = get_embed(input_data, vocab_size, embed_dim)
    output, final_state = build_rnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(output, vocab_size, activation_fn=None)

    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    # split input data into batches based on batch size and sequence length hyper params
    nb = int(len(int_text) / (batch_size * seq_length))
    nb_k = (nb * batch_size * seq_length)

    x_data = np.array(int_text[: nb_k])
    y_data = np.array(int_text[1 : nb_k] + [int_text[0]])

    x = x_data.reshape(batch_size, -1)
    y = y_data.reshape(batch_size, -1)

    x = np.split(x, nb, 1)
    y = np.split(y, nb, 1)

    batches = np.array(list(zip(x, y)))
    batches = batches.reshape(nb, 2, batch_size, seq_length)

    return batches


def train():
    version_check()
    # load word2vec data from preprocessed pickle file
    int_text, vocab_to_int, int_to_vocab, token_dict = load_data()

    # hyperparams
    num_epochs = 100
    batch_size = 100
    rnn_size = 1024
    embed_dim = 128
    seq_length = 19
    learning_rate = 0.002
    show_every_n_batches = 10
    save_dir = './save'

    train_graph = tf.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text, targets, lr = get_inputs()
        input_data_shape = tf.shape(input_text)
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
        logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

        # Probabilities for generating words
        probs = tf.nn.softmax(logits, name='probs')

        # Loss function
        cost = seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([input_data_shape[0], input_data_shape[1]]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # Train
    batches = get_batches(int_text, batch_size, seq_length)

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})

            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

                # Show every <show_every_n_batches> batches
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_dir)
        params = (seq_length, save_dir)
        save_params(params)
        print('Model Trained and Saved')


def get_tensors(loaded_graph):
    x_tensor = loaded_graph.get_tensor_by_name('input:0')
    ini_tensor = loaded_graph.get_tensor_by_name('initial_state:0')
    fin_tensor = loaded_graph.get_tensor_by_name('final_state:0')
    p_tensor = loaded_graph.get_tensor_by_name('probs:0')
    return x_tensor, ini_tensor, fin_tensor, p_tensor


def pick_word(probabilities, int_to_vocab):
    csum = np.cumsum(probabilities)
    c = np.sum(probabilities) * np.random.rand(1)
    word = int_to_vocab[int(np.searchsorted(csum, c))]
    return word


def split_into_sentences(text):
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def generate_question(prime_word):
    int_text, vocab_to_int, int_to_vocab, token_dict = load_data()
    seq_length, load_dir = load_params()
    gen_length = 1000

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)
        gen_question = [prime_word]

        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

        for n in range(gen_length):
            dyn_input = [[vocab_to_int[word] for word in gen_question[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})

            pred_word = pick_word(probabilities[0][dyn_seq_length - 1], int_to_vocab)

            gen_question.append(pred_word)

    ouija_question = ' '.join(gen_question)

    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        ouija_question = ouija_question.replace(' ' + token.lower(), key)

    ouija_question = ouija_question.replace('\n ', '\n')
    ouija_question = ouija_question.replace('( ', '(')
    ouija_question = split_into_sentences(ouija_question)

    for question in ouija_question:
        print(question)
        post_to_reddit = input("Would you like to post this question to reddit? [N/y] ")

        if post_to_reddit.lower() == 'y':
            return question.capitalize()
        print('\n')

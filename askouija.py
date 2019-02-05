import praw
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Base, Question
from tqdm import tqdm
from os import system
import sys
import deadbot_nn
import pickle
from flask import Flask, render_template, url_for, Response, flash
import time

app = Flask(__name__, template_folder="html")

print("deadbot 0.2.1 - reddit.com/AskOuija AI bot e: b@blairdev.com\n")

# Get secret login info etc, use rstrip to remove any whitespace
with open(".secrets", 'r') as secrets:
    secrets = secrets.readlines()
    client_id = secrets[0].rstrip()
    client_secret = secrets[1].rstrip()
    client_username = secrets[2].rstrip()
    client_password = secrets[3].rstrip()
    db_user = secrets[4].rstrip()
    db_pass = secrets[5].rstrip()

# Connect to DB and start session
db_login = "postgresql://{0}:{1}@localhost/deadbot".format(db_user, db_pass)
engine = create_engine(db_login)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

ua = "deadbot/0.1 by blairdev.com"

# Use PRAW to authenticate reddit session
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=ua,
                     username=client_username,
                     password=client_password)

ouija = reddit.subreddit('askouija')

### Flask ###

@app.route('/')
def get_home():
    return render_template('home.html')


@app.route('/getposts')
def get_posts():
    return render_template('getposts.html')


@app.route('/getposts/progress')
def get_posts_progress():
    def seed_db():
        """
        Gets "hot" posts from r/AskOuija from last 24hours and
        adds them to database if they have answer flairs

        :param askreddit: 0 - will search r/AskOuija, 1 will search r/AskReddit

        """
        post_counter = 0
        limit = 1000
        print("\n")
        askreddit = 0
        if askreddit == 1:
            ouija = reddit.subreddit('askreddit')
        else:
            ouija = reddit.subreddit('askouija')
        pbar = 0
        x = 0

        while x <= 100:
            for submission in ouija.new(limit=limit):
                if session.query(Question.id).filter_by(submission_id=submission.id).scalar() is None:
                    new_question = Question(submission_id=submission.id,
                                            title=submission.title,
                                            answer="||")
                    session.add(new_question)
                    session.commit()
                    post_counter += 1
                yield "data:" + str(x) + "\n\n"

                pbar += 1
                if pbar == 100:
                    pbar = 0
                    x += 10

                if x == 100:
                    print("Done! {0} new posts added to the database.\n".format(post_counter))
                    print("Preprocessing data ready for training...")
                    preprocess_data()
                    yield "data:Done! {0} new posts added to the database.\n\n".format(post_counter)
                    # return Response(seed_db(), mimetype='text/event-stream')
                if askreddit == 1:
                    flairexists = True
                    answered = True

    return Response(seed_db(), mimetype='text/event-stream')

@app.route('/train')
def get_train():
    return render_template('train.html')


@app.route('/train/progress')
def get_train_progress():
    ''' Called by event listener to train model and update progress bar'''
    def train_model():
        pass

@app.route('/generator')
def get_questions():
    return render_template('generatequestions.html')
### END ###

def get_hot(askreddit=0):
    """
    Gets "hot" posts from r/AskOuija from last 24hours and
    adds them to database if they have answer flairs

    :param askreddit: 0 - will search r/AskOuija, 1 will search r/AskReddit
    """
    post_counter = 0
    limit = int(input("How many posts to scan?"))
    print("\n")
    if askreddit == 1:
        ouija = reddit.subreddit('askreddit')
    else:
        ouija = reddit.subreddit('askouija')

    with tqdm(total=limit) as pbar:
        for submission in ouija.top(limit=limit, time_filter='day'):
            flairexists = submission.link_flair_text is not None
            answered = submission.link_flair_text != 'unanswered'

            if askreddit == 1:
                flairexists = True
                answered = True

            if flairexists and answered:
                # print(submission.link_flair_text[12:])

                if session.query(Question.id).filter_by(submission_id=submission.id).scalar() is None:

                    if submission.link_flair_text is not None:
                        answer = submission.link_flair_text[12:]
                    else:
                        answer = "||"
                    new_question = Question(submission_id=submission.id,
                                            title=submission.title,
                                            answer=answer)
                    session.add(new_question)
                    session.commit()
                    post_counter += 1
            pbar.update(1)
    print("Done! {0} new posts added to the database.\n".format(post_counter))
    print("Preprocessing data ready for training...")
    preprocess_data()


def get_new():
    """
    Gets "new" posts from r/AskOuija and adds them to database
    This feature is used to bolster data resources - many more posts are found this way
    """
    post_counter = 0
    limit = int(input("How many posts to scan? "))
    print("\n")
    with tqdm(total=limit) as pbar:
        for submission in ouija.new(limit=limit):
            if session.query(Question.id).filter_by(submission_id=submission.id).scalar() is None:

                new_question = Question(submission_id=submission.id,
                                        title=submission.title,
                                        answer="||")
                session.add(new_question)
                session.commit()
                post_counter += 1
            pbar.update(1)
    print("Done! {0} new posts added to the database.\n".format(post_counter))
    print("Preprocessing data ready for training...")
    preprocess_data()



def cls():
    system('clear')


def query_db():
    """
    Prints all database entries to terminal
    """
    q = session.query(Question).all()

    for x in q:
        print("ID: {0}\n"
              "Submission ID: {1}\n"
              "Title: {2}\n"
              "Answer: {3}\n\n\n".format(x.id, x.submission_id, x.title, x.answer))


def get_stats():
    """
    Prints statistics from database data to terminal
    """
    total = session.query(Question).count()
    unique = session.query(Question).distinct(Question.answer).count()
    print("\nThere are {0} answers from spirits in the database, {1} of them are unique.\n".format(total, unique))


def preprocess_data():
    """
    Prepares data for deep learning algorithm by converting words to integers (word2vec)
    and replacing punctuation with token keys
    Saves output to pickle file (preprocess.p)
    """
    questions = []
    q = session.query(Question).all()

    for post in q:
        questions.append(post.title)

    # convert DB query results to string
    text = ''.join(questions)

    token_dict = token_lookup()

    # replace punctuation characters with token
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    # convert text to integer IDs
    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))
    print("Data preprocessed and saved.")


def create_lookup_tables(text):
    """
    Creates lookup table for word2vec
    """
    int_to_vocab, vocab_to_int = {}, {}

    for x, y in enumerate(text):
        int_to_vocab[x] = y
        vocab_to_int[y] = x

    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    d_punctuation = {
        '.': '||Period||',
        ',': '||Comma||',
        '"': '||QuotationMark||',
        ';': '||Semicolon||',
        '!': '||ExclamationMark||',
        '?': '||QuestionMark||',
        '(': '||LeftParentheses||',
        ')': '||RightParentheses||',
        '--': '||Dash||',
        '_': '||Underscore||',
        '\n': '||Return||',
    }
    return d_punctuation


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.debug = True
    app.run()

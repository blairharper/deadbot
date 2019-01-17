import praw
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Base, Question
from tqdm import tqdm
from os import system
import sys
print("deadbot 0.1 - reddit.com/AskOuija AI bot e: b@blairdev.com\n")

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

ans = True


def get_hot():
    post_counter = 0
    limit = int(input("How many posts to scan? "))
    print("\n")
    with tqdm(total=limit) as pbar:
        for submission in ouija.top(limit=limit, time_filter='day'):
            flairexists = submission.link_flair_text is not None
            answered = submission.link_flair_text != 'unanswered'

            if flairexists and answered:
                # print(submission.link_flair_text[12:])

                if session.query(Question.id).filter_by(submission_id=submission.id).scalar() is None:

                    new_question = Question(submission_id=submission.id,
                                            title=submission.title,
                                            answer=submission.link_flair_text[12:])
                    session.add(new_question)
                    session.commit()
                    post_counter += 1
            pbar.update(1)
    print("Done! {0} new posts added to the database.\n".format(post_counter))


def cls():
    system('clear')


def query_db():
    q = session.query(Question).all()

    for x in q:
        print("ID: {0}\n"
              "Submission ID: {1}\n"
              "Title: {2}\n"
              "Answer: {3}\n\n\n".format(x.id, x.submission_id, x.title, x.answer))


def get_stats():
    total = session.query(Question).count()
    unique = session.query(Question).distinct(Question.answer).count()
    print("\nThere are {0} answers from spirits in the database, {1} of them are unique.\n".format(total, unique))


def preprocess_data():
    # TODO:
    # 1. Get every entry in DB
    # 2. Give each unique word an ID number
    # 3. replace every word with an ID number / create lookup table
    #       (dictionaries -> word to ID and ID to word)
    # 4. Tokenise punctuation
    pass


while ans:
    print("""
        Deadbot Menu:
        1. Scan for hot posts
        2. Show me what you've got
        3. Stats
        4. Preprocess data
        5. Exit\n
        """)
    ans = input("What would you like to do? ")
    if ans == "1":
        cls()
        get_hot()
    elif ans == "2":
        cls()
        query_db()
    elif ans == "3":
        cls()
        get_stats()
    elif ans == "4":
        cls()
        preprocess_data()
    elif ans == "5":
        print("\nGoodbye")
        sys.exit()

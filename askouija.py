import praw
from sqlalchemy import create_engine, asc, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from database_setup import Base, Question, Answer
import sys
from time import sleep
print("IamAdeadbot 0.1 - Welcome to the spirit world. e: b@blairdev.com\n")

# Connect to DB and start session
engine = create_engine('postgresql://deadbot:PASSWORD@localhost/deadbot')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

# Get secret login info etc, use rstrip to remove any whitespace
secrets = open(".secrets", 'r').readlines()
client_id = secrets[0].rstrip()
client_secret = secrets[1].rstrip()
client_username = secrets[2].rstrip()
client_password = secrets[3].rstrip()

ua = "deadbot/0.1 by blairdev.com"

# Use PRAW to authenticate reddit session
reddit = praw.Reddit(client_id=client_id,
    client_secret=client_secret,
    user_agent=ua,
    username=client_username,
    password=client_password)

ouija = reddit.subreddit('askouija')

ans = True


def getRising():
 
    for submission in ouija.rising(limit=5):
        if submission.id != "673qgu": # ignore the 'please read the rules blah blah blah' post
            comment_ids = []
            scores = []
            total_score = 0
            
            submission.comments.replace_more(limit=None)
            submission.comment_sort = 'top'
            
            top_level_comments = list(submission.comments)        
            top_word = "{0}".format(top_level_comments[0].body)
            comment_ids.append(top_level_comments[0].id)
            scores.append(top_level_comments[0].score)

            for reply in top_level_comments[0].replies.list():
                if (len(reply.body) == 1) and (reply.score > 0):
                    top_word = top_word + "{0}".format(reply.body)
                    comment_ids.append(reply.id)
                    scores.append(reply.score)

            for s in scores:
                total_score += int(s)
            
            if session.query(Question.id).filter_by(submission_id=submission.id).scalar() is None:

                newQuestion = Question(submission_id=submission.id,
                                       title=submission.title)
                session.add(newQuestion)
                session.commit()

                newAnswer = Answer(comment_ids=comment_ids,
                                   scores=scores,
                                   body=top_word,
                                   total_score=total_score,
                                   question_id=newQuestion.id)
                session.add(newAnswer)

                session.commit()
                print("{0}\n".format(top_word))

def queryDB():
    q = session.query(Question).all()

    for x in q:
        print("ID: {0} Title: {1}\n".format(x.id, x.title))


    a = session.query(Answer).all()

    for x in a:
        print("ID: {0} QuestionID: {1} Answer: {2}".format(x.id, x.question_id, x.body))


while ans:
    print("""
        1. Get rising
        2. Query database
        3. Exit
        """)
    ans = input("What would you like to do? ")
    if ans == "1":
        getRising()
    elif ans == "2":
        queryDB()
    elif ans == "3":
        print("\nGoodbye")
        sys.exit()


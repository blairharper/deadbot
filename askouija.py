import praw

print("IamAdeadbot 0.1 - Welcome to the spirit world. e: b@blairdev.com\n")

secrets = open(".secrets", 'r').readlines()

client_id = secrets[0].rstrip()
client_secret = secrets[1].rstrip()
client_username = secrets[2].rstrip()
client_password = secrets[3].rstrip()
ua = "deadbot/0.1 by blairdev.com"

reddit = praw.Reddit(client_id=client_id,
    client_secret=client_secret,
    user_agent=ua,
    username=client_username,
    password=client_password)

ouija = reddit.subreddit('askouija')

top_word = ""

for submission in ouija.hot(limit=5):
    if submission.id != "673qgu": # ignore the 'please read the rules blah blah blah' post
        submission.comments.replace_more(limit=None)
        submission.comment_sort = 'top'
        print(submission.title)
        top_level_comments = list(submission.comments)        
        top_word = "{0}".format(top_level_comments[0].body)
        for reply in top_level_comments[0].replies.list():
            if (len(reply.body) == 1) and (reply.score > 0):
                top_word = top_word + "{0}".format(reply.body)
        print("{0}\n".format(top_word))

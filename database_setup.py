from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

Base = declarative_base()

with open(".secrets", 'r') as secrets:
    secrets = secrets.readlines()
    db_user = secrets[4].rstrip()
    db_pass = secrets[5].rstrip()

class Question(Base):
    """ Database table structure for storing questions """
    __tablename__ = 'question'

    id = Column(Integer, primary_key=True)
    submission_id = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)
    answer = Column(String(500), nullable=False)


# Set engine as postgresql and define db file name
db_login = "postgresql://{0}:{1}@localhost/deadbot".format(db_user, db_pass)
engine = create_engine(db_login)


# Create db
Base.metadata.create_all(engine)

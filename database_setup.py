from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()


class Question(Base):
    """ Database table structure for storing questions """
    __tablename__ = 'question'

    id = Column(Integer, primary_key=True)
    submission_id = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)


class Answer(Base):
    """  Database table structure for storing answers """
    __tablename__ = 'answer'

    id = Column(Integer, primary_key=True)
    comment_ids = Column(String(250), nullable=False)
    scores = Column(String(250), nullable=False)
    body = Column(String(250), nullable=False)
    total_score = Column(Integer)
    question_id = Column(Integer, ForeignKey('question.id'))
    question = relationship(Question)

    @property
    def serialise(self):
        """ Return object data in serialisable format """
        return {
            'body': self.body,
            'id': self.id,
        }


# Set engine as postgresql and define db file name
engine = create_engine('postgresql://deadbot:PASSWORD@localhost/deadbot')

# Create db
Base.metadata.create_all(engine)
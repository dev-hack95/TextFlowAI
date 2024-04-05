import os
import warnings
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer

# Config
load_dotenv(".env")
warnings.filterwarnings("ignore")
Base = declarative_base()

class DBLocalSession:
    def __init__(self) -> None:
        self.db_user = os.environ.get("db_user")
        self.db_password = os.environ.get("db_password")
        self.db_host = os.environ.get("db_host")
        self.db_name = os.environ.get("db_name")

    def LocalSession(self) -> sessionmaker:
        engine = create_engine(f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")
        SessionLocal = sessionmaker(autoflush=True, bind=engine)
        db = SessionLocal()
        return db
    
class SignInSchema(Base):

    __tablename__ = "user_service"
    email = Column(String)
    password = Column(String)
from flask import Flask
from config import Config
from flask_bootstrap import Bootstrap
import logging

# print errors/debug log in terminal
import sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


app = Flask(__name__, static_url_path='/static')
Bootstrap(app)
app.config.from_object(Config)

from app import routes
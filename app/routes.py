from app import app
from flask import render_template, flash, redirect
from app.forms import inputForm
from Search import WWUTransphoner
import SentenceGen

wordMatches = None
sentenceMatches = None

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = inputForm()
    global wordMatches
    global sentenceMatches
    matchesReady = False

    if form.validate_on_submit():
        matchesReady = getResults(form)
        return render_template("home.html",form=form, matchesReady=matchesReady, words=wordMatches, sentences=sentenceMatches)
    return render_template("home.html",form=form, matchesReady=matchesReady)

def getResults(form):
    global wordMatches
    global sentenceMatches

    wwut = WWUTransphoner(form.inputLang.data)
    wordMatches = wwut.get_mnemonics(form.inputWord.data, form.translation.data, form.numMatches.data)
    sentenceMatches = SentenceGen.gen_sentence(wordMatches)

    ##for testing
    # wordMatches = ['this', 'is', 'a', 'test']
    # sentenceMatches = ['sentence 1', 'sentence 2', 'sentence 3', 'sentence 4']

    return True


@app.route('/about')
def about():
    return render_template("about.html")



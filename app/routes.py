from app import app
from flask import render_template, flash, redirect
from app.forms import inputForm
from WWUTransphoner import WWUTransphoner

wordMatches = None
phoneMatches = None
sentenceMatches = None
inputWordPhones = None

wwut = None

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = inputForm()
    global wordMatches
    global phoneMatches
    global sentenceMatches
    global inputWordPhones
    matchesReady = False

    if form.validate_on_submit():
        matchesReady = getResults(form)
        if matchesReady:
            return render_template(
                "home.html",
                form=form, 
                matchesReady=matchesReady, 
                wordMatches=wordMatches, 
                wordPhoneMatches=zip(wordMatches,phoneMatches), 
                inputWordPhones=inputWordPhones,
                sentences=sentenceMatches)
        else:
            flash("Server doesn't enough data for: " + form.inputWord.data + ", sorry about that.")
    return render_template("home.html",form=form, matchesReady=matchesReady)

def getResults(form):
    global wordMatches
    global phoneMatches
    global sentenceMatches
    global inputWordPhones
    global wwut

    if not wwut or form.inputLang.data != wwut.input_language:
        flash("Setting up server for '" + form.inputLang.data + "', may take a moment.")
        wwut = WWUTransphoner(form.inputLang.data)

    wordMatches, phoneMatches, inputWordPhones = wwut.get_mnemonics(form.inputWord.data, form.translation.data, int(form.numMatches.data), include_phones=True)
    if not wordMatches:
        return False

    sentenceMatches = wwut.gen_sentence(wordMatches)

    ##for testing
    #wordMatches = ['this', 'is', 'a', 'test']
    #sentenceMatches = ['sentence 1', 'sentence 2', 'sentence 3', 'sentence 4']

    return True


@app.route('/about')
def about():
    return render_template("about.html")



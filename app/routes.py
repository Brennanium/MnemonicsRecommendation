from app import app
from flask import render_template, flash, redirect
from app.forms import inputForm
from WWUTransphoner import WWUTransphoner
from typing import List

wordMatches: List[str] = None
phoneMatches: List[str] = None
sentenceMatches: List[str] = None
inputWordPhones: str = None

wwut: WWUTransphoner = None

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
                phoneMatches=phoneMatches, 
                inputWordPhones=inputWordPhones,
                sentences=sentenceMatches)
        else:
            flash("Server doesn't enough data for: " + form.inputWord.data + ", sorry about that.")
    return render_template("home.html",form=form, matchesReady=matchesReady)

def getResults(form: inputForm):
    global wordMatches
    global phoneMatches
    global sentenceMatches
    global inputWordPhones
    global wwut

    if not wwut or (form.inputLang.data != wwut.input_language and form.outputLang.data != wwut.output_language):
        flash("Setting up server for '" + form.inputLang.data + "' and '" + form.outputLang.data  + "', may take a moment.")
        wwut = WWUTransphoner(form.inputLang.data, form.outputLang.data)
    elif form.inputLang.data != wwut.input_language:
        flash("Setting up server for '" + form.inputLang.data + "', may take a moment.")
        wwut = WWUTransphoner(form.inputLang.data, form.outputLang.data)
    elif form.outputLang.data != wwut.output_language:
        flash("Setting up server for '" + form.outputLang.data + "', may take a moment.")
        wwut = WWUTransphoner(form.inputLang.data, form.outputLang.data)


    try:
        wordMatches, phoneMatches, inputWordPhones = wwut.get_mnemonics(form.inputWord.data, form.translation.data, int(form.numMatches.data), include_phones=True)
        if not wordMatches:
            return False
    except KeyError as error:
        flash(str(error))
        return False

    if form.outputLang.data == 'en':
        sentenceMatches = wwut.gen_sentences(wordMatches)
    else:
        sentenceMatches = []

    ##for testing
    #wordMatches = ['this', 'is', 'a', 'test']
    #sentenceMatches = ['sentence 1', 'sentence 2', 'sentence 3', 'sentence 4']

    return True


@app.route('/about')
def about():
    return render_template("about.html")



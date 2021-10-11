# GaidhligAutofill

https://github.com/Brennanium/GaidhligAutofill

## Setting up GaidhligAutofill in pipenv
1) Install pipenv (run `brew install pipenv` on MacOS)
2) Navigate to folder where repo was cloned
3) Run `pipenv install` to initialize virtual environment and install dependancies


## Setting up ARCOSG with nltk

1) Add the `'gd-parole.map'` file to `/~/nltk_data/taggers/universal_tagset` (run `nltk.download('universal')` if not present)
2) Find the file `'__init__.py'` in `/~/Library/Python/3.7/lib/python/site-packages/nltk/corpus/__init__.py`
3) Open the file with a text editor
4) Add the following lines to the file, under the code for 'Brown':

```
arcosg = LazyCorpusLoader(
	'arcosg',
	CategorizedTaggedCorpusReader,
	r'.\*\.txt',
	cat_file='cats.prn',
	tagset='-gd-parole',
	encoding='utf-8',
)
```

5) Save the file

Initiate ARCOSG with the following code in Python:


	>>> import nltk
	>>> from nltk.corpus import arcosg
	>>> arcosg.tagged_words()
	[('[3]', 'XSC'), ('ach', 'CC'), ('bha', 'V-S'), ...]
	>>> arcosg.tagged_words(tagset='universal')
	[('[3]', 'X'), ('ach', 'CONJ'), ('bha', 'VERB'), ...]


If this fails, try restarting Python and reimporting nltk.

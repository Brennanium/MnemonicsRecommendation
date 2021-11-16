from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, SelectField
from wtforms.validators import DataRequired, NumberRange

class inputForm(FlaskForm):
    inputLang = SelectField('Input Language', choices=[('en', 'English'), ('ja', 'Japanese'), ('da', 'German')])
    outputLang = SelectField('Output Language', choices=[('en','English')])
    inputWord = StringField('Input Word', validators=[DataRequired()])
    translation = StringField('Input Word Translation', )
    numMatches = SelectField('Number of Matches', choices=[(1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6') ])
    submit = SubmitField('Generate')
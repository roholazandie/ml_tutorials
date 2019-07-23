from flair.data import Sentence
from flair.models import SequenceTagger



sentence = Sentence("I was in New York.")

tagger = SequenceTagger.load("ner")

result = tagger.predict(sentence)

print(result)
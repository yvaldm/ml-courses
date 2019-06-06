from classes.sentence import Sentence
from classes.tag import Tag

tag1 = Tag(1, 6, "mylabel")
tag2 = Tag(10, 16, "something")
tag3 = Tag(25, 30, "whatever")

sentence = Sentence("black", [tag1, tag2, tag3])

print(sentence.sentence)
print(sentence.tags[0].start)

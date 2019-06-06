"""

Build NER model for electronics entities

Valery Yakovlev
"""

import spacy

prdnlp = spacy.load("vd.products.ner.model")

# Test your text
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

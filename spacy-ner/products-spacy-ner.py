"""

Build NER model for electronics entities

Valery Yakovlev
"""

import random

import spacy

TRAIN_DATA = [('сколько стоит polo?', {'entities': [(21, 25, 'PrdName')]}),
              ('сколько стоит ball?', {'entities': [(21, 25, 'PrdName')]}),
              ('сколько стоит jegging?', {'entities': [(21, 28, 'PrdName')]}),
              ('сколько стоит t-shirt?', {'entities': [(21, 28, 'PrdName')]}),
              ('сколько стоит jeans?', {'entities': [(21, 26, 'PrdName')]}),
              ('сколько стоит bat?', {'entities': [(21, 24, 'PrdName')]}),
              ('сколько стоит shirt?', {'entities': [(21, 26, 'PrdName')]}),
              ('сколько стоит bag?', {'entities': [(21, 24, 'PrdName')]}),
              ('сколько стоит cup?', {'entities': [(21, 24, 'PrdName')]}),
              ('сколько стоит jug?', {'entities': [(21, 24, 'PrdName')]}),
              ('сколько стоит plate?', {'entities': [(21, 26, 'PrdName')]}),
              ('сколько стоит glass?', {'entities': [(21, 26, 'PrdName')]}),
              ('сколько стоит moniter?', {'entities': [(21, 28, 'PrdName')]}),
              ('сколько стоит desktop?', {'entities': [(21, 28, 'PrdName')]}),
              ('сколько стоит bottle?', {'entities': [(21, 27, 'PrdName')]}),
              ('сколько стоит mouse?', {'entities': [(21, 26, 'PrdName')]}),
              ('сколько стоит keyboad?', {'entities': [(21, 28, 'PrdName')]}),
              ('сколько стоит chair?', {'entities': [(21, 26, 'PrdName')]}),
              ('сколько стоит table?', {'entities': [(21, 26, 'PrdName')]}),
              ('сколько стоит watch?', {'entities': [(21, 26, 'PrdName')]})]


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(TRAIN_DATA, 20)

# Save our trained Model
prdnlp.to_disk("vd.products.ner.model.rus")

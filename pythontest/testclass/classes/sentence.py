class Sentence(object):
    """docstring"""

    def __init__(self, sentence, tags):
        """Constructor"""
        self.sentence = sentence
        self.tags = tags

    def brake(self):
        """
        Stop the car
        """
        return "Braking"

    def drive(self):
        """
        Drive the car
        """
        return "I'm driving!"

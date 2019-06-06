class Tag(object):
    """docstring"""

    def __init__(self, start, end, label):
        """Constructor"""
        self.start = start
        self.end = end
        self.tag = label

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

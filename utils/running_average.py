class RunningAverage():
    """ Calculated at every iteration the average of a value
    """
    def __init__(self) -> None:
        self.steps = 0
        self.total = 0

    def update(self, value):
        self.total += value
        self.steps += 1

    def __call__(self):
        return self.total / self.steps

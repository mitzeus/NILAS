# TODO:
# Refactor add, get_best, and clear cause of Copilot generation
# Make sure it picks top branches based on allowed vocab
# Allow some slack to make sure new words are used
# Measure naturalness of generated sequences
# Have some kind of check to see if the final output actually makes sense to the input given.


class BeamSearch:
    def __init__(self, beam_size):
        self.beam_size = beam_size
        self.beams = [[]]

    def add(self, sequence, score):
        self.beams.append((sequence, score))
        self.beams.sort(key=lambda x: x[1], reverse=True)
        self.beams = self.beams[: self.beam_size]

    def get_best(self):
        return self.beams[0] if self.beams else None

    def clear(self):
        self.beams = []

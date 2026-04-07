# TODO:
# Refactor add, get_best, and clear cause of Copilot generation
# Make sure it picks top branches based on allowed vocab
# Allow some slack to make sure new words are used
# Measure naturalness of generated sequences
# Have some kind of check to see if the final output actually makes sense to the input given.

import graphviz
import os


class Node:
    def __init__(self, me: str, parent: object | None, probability: float):
        self.me = me
        self.parent = parent
        self.probability = probability
        self.children = []


class BeamSearch:
    """
    The `BeamSearch` class contains attributes and methods for doing beam search on a sequence.
    Each object supports one sequence and does operations on that sequence.

    Args:
        beam_size: Size of beams to expands for each iteration
    """

    def __init__(self, beam_size: int, allowed_words: list[str]):
        self.sequence = ""

        self.beam_size = beam_size
        self.allowed_words = allowed_words

        self.tree = []
        self.beams = []  # Holds beam strings
        self.beam_obj = []  # Holds object for last token in beams

    def update(self, layer: list[dict[str, float]] | dict[str, float]):
        """
        Adds layer to beam tree, updates beams to be the best ones based on metrics of
        naturalness and lexical constraints.

        Args:
            layer: list of list of strings where each string must be the updated part of the beam i.
                    Each index i corresponds to the promposed updated sequences for beam i. Takes only
                    a single list of strings if it's the first token in the sequence (tree root). #TODO

        Returns:
            list[str]: Updated beams
        """
        # TODO:
        # Add new layer to beam tree
        # Measure lexical constraints and naturalness of all the new paths and rank them
        # Update beams

        if len(self.tree) == 0 and type(layer) is dict:  # Check if root
            # root

            proposed_objs = []
            for key, prob in layer.items():
                root = Node(key, None, prob)
                self.tree.append(root)
                proposed_objs.append(root)

            # rank the best
            top_results = self.pick_best_paths(proposed_objs)

            for item in top_results:
                self.beams.append(item.me)
                self.beam_obj.append(item)

            return self.beams

        # normal sequence generation

        proposed_objs = []
        for i, beam_i_proposal in enumerate(layer):
            for key, prob in beam_i_proposal.items():
                node = Node(key, self.beam_obj[i], prob)
                self.beam_obj[i].children.append(node)
                self.tree.append(node)
                proposed_objs.append(node)

        top_results = self.pick_best_paths(proposed_objs)

        self.beams = []
        self.beam_obj = []
        for item in top_results:
            self.beams.append(self.build_sequence_from_obj(item))
            self.beam_obj.append(item)

        return self.beams

        # make objects of them and append to children of the beam objs

        # ...

    def build_sequence_from_obj(self, obj: object):
        sequence_build = ""

        def build_sequence(part: object):
            if part.parent == None:
                return part.me

            return build_sequence(part.parent) + " " + part.me

        sequence_build = build_sequence(obj)

        return sequence_build

    def pick_best_paths(self, proposed: list[object]):
        """
        Ranks all newly generated paths by total sequence probability. Supports only
        picking best paths for one beam at a time.

        Args:
            proposed: List of objects where each object is a proposed token continuation of a sequence.

        Returns:
            list[object]: Top-N (based on beam size) results with highest sequence probability.
        """
        # TODO add limitations using flashcard list

        def calculate_sequence_prob(obj: object):
            if obj.parent == None:  # base case
                return obj.probability

            # has parents
            return obj.probability * calculate_sequence_prob(obj.parent)

        probabilities = []

        for proposal in proposed:
            probabilities.append(calculate_sequence_prob(proposal))

        # rank and keep top beam size
        top_indices = sorted(
            range(len(probabilities)), key=lambda i: probabilities[i], reverse=True
        )[: self.beam_size]
        top_n_items = [proposed[i] for i in top_indices]

        return top_n_items

    def reset(self):
        self.sequence = ""
        self.tree = []
        self.beams = []

    def visualize_tree(self, filename: str):
        u = graphviz.Digraph(
            "Beam Tree",
            node_attr={"color": "lightblue2", "style": "filled"},
        )

        for item in self.tree:
            if item.parent != None:
                u.edge(
                    f"({item.parent.probability}) {item.parent.me}",
                    f"({item.probability}) {item.me}",
                )

        u.save(f"figures/{filename}.gv")
        u.attr(size="8,8", dpi="500")

        # u.format = "png"
        u.render(f"figures/{filename}", view=False, format="png")

        os.remove(f"figures/{filename}")  # removes extra no codex .dot file

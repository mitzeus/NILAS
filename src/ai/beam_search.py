# TODO:
# Refactor add, get_best, and clear cause of Copilot generation
# Make sure it picks top branches based on allowed vocab
# Allow some slack to make sure new words are used
# Measure naturalness of generated sequences
# Have some kind of check to see if the final output actually makes sense to the input given.

import graphviz
import os


class Node:
    """
    Node object for use in `BeamSearch` and creating a beam tree.
    """

    def __init__(
        self,
        token_id: int,
        token_string: str,
        parent: object | None,
        probability: float,
    ):
        self.id = token_id
        self.token = token_string
        self.parent = parent
        self.probability = probability
        self.children = []


# TODO remake for token ids and not raw strings (until return)
class BeamSearch:
    """
    The `BeamSearch` class contains attributes and methods for doing beam search on a sequence.
    Each object supports one sequence and does operations on that sequence.

    Args:
        beam_size: Size of beams to expands for each iteration
    """

    def __init__(self, beam_size: int, allowed_words: list[str]):
        self.sequence = ""
        self.initialized = False  # to check if root has been added to tree yet

        self.beam_size = beam_size
        self.allowed_words = allowed_words

        self.tree = []
        self.beams = []  # Holds beam strings
        self.beam_ids = []  # Holds beam token ids
        self.beam_obj = []  # Holds object for last token in beams

        self.best_beam_probability = float("-inf")
        self.best_beam_ids = []

    def update(self, layer: list[dict[int, any]] | dict[int, any]):
        """
        Adds layer to beam tree, updates beams to be the best ones based on metrics of
        naturalness and lexical constraints.

        # TODO redo documentation for the new input format
        # TODO add a checker where len logprobs must be bigger than beam size
        Args:
            layer: list of list of integers where each integer must be the updated part of the beam i.
                    Each index i corresponds to the promosed updated sequences for beam i. Takes only
                    a single list of integers if it's the first token in the sequence (tree root). #TODO

        Returns:
            list[str]: Updated beams
        """

        if len(self.tree) == 0 and type(layer) is dict:  # Check if root
            # root

            self.initialized = True

            proposed_objs = []
            for token_id, token_info in layer.items():
                root = Node(token_id, token_info["word"], None, token_info["logprob"])
                proposed_objs.append(root)

            # rank the best
            top_results = self.pick_best_paths(proposed_objs)

            for item in top_results:
                self.tree.append(item)  # Only add selected nodes to tree
                self.beams.append(item.token)
                self.beam_ids.append([item.id])
                self.beam_obj.append(item)

        else:
            # normal sequence generation

            proposed_objs = []
            for i, beam_i_proposal in enumerate(layer):
                for token_id, token_info in beam_i_proposal.items():
                    node = Node(
                        token_id,
                        token_info["word"],
                        self.beam_obj[i],
                        token_info["logprob"],
                    )
                    self.beam_obj[i].children.append(node)
                    proposed_objs.append(node)

            top_results = self.pick_best_paths(proposed_objs)

            # Only add selected nodes to tree
            for item in top_results:
                self.tree.append(item)

            self.beams = []
            self.beam_ids = []
            self.beam_obj = []
            for item in top_results:
                string_sequence, id_sequence = self.build_sequence_from_obj(item)
                self.beams.append(string_sequence)
                self.beam_ids.append(id_sequence)
                self.beam_obj.append(item)

        return self.beams, self.beam_ids

        # TODO check for EOS

    def build_sequence_from_obj(self, obj: object):
        """
        Builds string and ID sequence from a single parsed object using `parent` attribute.
        # TODO upodate for token ids as well
        Args:
            obj: `Node` object.

        Returns:
            str: Corresponding sequence string for the provided object.
        """
        sequence_build = ""
        sequence_ids_build = []

        def build_sequence(part: object):
            """
            Builds string sequence from a single parsed object using `parent` attribute.
            A subfunction of `build_sequence_from_obj` function and uses recursion.
            """
            if part.parent == None:
                return part.token, [part.id]

            str_part, id_part = build_sequence(part.parent)

            return str_part + " " + part.token, id_part + [part.id]

        sequence_build, sequence_ids_build = build_sequence(obj)

        return sequence_build, sequence_ids_build

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
            """
            Calculates total probability of a sequence given a `Node` object using recursion.

            Args:
                obj: `Node` object.

            Returns:
                str: Total probability of sequence from `Node` object.
            """
            if obj.parent == None:  # base case
                return obj.probability

            # has parents
            return obj.probability + calculate_sequence_prob(
                obj.parent
            )  # sum of logprobs

        probabilities = []

        for proposal in proposed:
            probabilities.append(calculate_sequence_prob(proposal))

        # rank and keep top beam size
        top_indices = sorted(
            range(len(probabilities)), key=lambda i: probabilities[i], reverse=True
        )[: self.beam_size]
        top_n_items = [proposed[i] for i in top_indices]

        return top_n_items

    def length_penalty(length: int, alpha: float = 0.6) -> float:
        """
        Wu et al. 2016 (Google NMT) length penalty.
        alpha=0:   no normalization (raw log-prob, biased to short)
        alpha=1:   full normalization (divide by length)
        alpha=0.6: empirically good default for most tasks
        """
        return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

    def calculate_normalized_probability(
        self, log_prob_sum: float, length: int, alpha: float = 0.6
    ) -> float:
        return log_prob_sum / self.length_penalty(length, alpha)

    def reset(self):
        """
        Resets sequence, tree, and references to current beams.

        Args:
            None

        Returns:
            None
        """
        self.sequence = ""
        self.tree = []
        self.beams = []
        self.beam_ids = []
        self.beam_obj = []

    def visualize_tree(self, filename: str):
        """
        Visualizes beam tree using GraphViz library.

        Args:
            filename: File name of the generated png file

        Returns:
            None
        """
        u = graphviz.Digraph(
            "Beam Tree",
            node_attr={"color": "lightblue2", "style": "filled"},
        )

        for item in self.tree:
            if item.parent is not None:
                parent_node_name = f"node_{id(item.parent)}"
                child_node_name = f"node_{id(item)}"
                parent_label = (
                    f"({round(item.parent.probability, 2)}) {item.parent.token}"
                )
                child_label = f"({round(item.probability, 2)}) {item.token}"

                u.node(parent_node_name, label=parent_label)
                u.node(child_node_name, label=child_label)
                u.edge(parent_node_name, child_node_name)

        u.save(f"figures/{filename}.gv")
        u.attr(size="12,12", dpi="1000")

        # u.format = "png"
        u.render(f"figures/{filename}", view=False, format="png")

        os.remove(f"figures/{filename}")  # removes extra no codex .dot file

# TODO:
# Refactor add, get_best, and clear cause of Copilot generation
# Make sure it picks top branches based on allowed vocab
# Allow some slack to make sure new words are used
# Measure naturalness of generated sequences
# Have some kind of check to see if the final output actually makes sense to the input given.

import graphviz
import os
import spacy
import re
import contractions
import time


class Node:
    """
    Node object for use in `BeamSearch` and creating a beam tree.
    """

    def __init__(
        self,
        token_id: list[int],
        token_string: str,
        parent: object | None,
        probability: float,
    ):
        self.id = token_id
        self.token = token_string
        self.parent = parent
        self.probability = probability
        self.children = []


class Beam:
    """
    A Beam object
    """

    def __init__(
        self,
        sequence_ids: list[int],
        sequence: str,
        logprob: float,
        normalized_logprob: float,
        node_end: object,
        is_completed: bool = False,
    ):
        self.ids = sequence_ids
        self.sequence = sequence
        self.logprob = logprob
        self.normalized_logprob = normalized_logprob
        self.node_end = node_end
        self.completed = is_completed


# TODO remake for token ids and not raw strings (until return)
class BeamSearch:
    """
    The `BeamSearch` class contains attributes and methods for doing beam search on a sequence.
    Each object supports one sequence and does operations on that sequence.

    Args:
        beam_size: Size of beams to expands for each iteration
    """

    def __init__(
        self,
        beam_size: int,
        allowed_words: list[str],
        tokenizer: object,
        allowed_word_penalty: float = 2.5,
        # min_allowed_len: int = 5,
        alpha: float = 0.6,
    ):
        self.sequence = ""
        self.initialized = False  # to check if root has been added to tree yet

        self.beam_size = beam_size
        self.allowed_words = allowed_words
        self.eos_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        # self.min_allowed_len = min_allowed_len
        self.alpha = alpha

        # self.nlp = spacy.load("sv_core_news_lg")  # for Swedish

        self.tree = []
        self.beams = []  # Holds beam objects
        # self.beams = []  # Holds beam strings
        # self.beam_ids = []  # Holds beam token ids
        self.beam_obj = []  # Holds object for last token in beams

        self.best_beam_probability = float("-inf")
        self.best_completed_beams = []

        self.best_beam = ""
        self.finished = False

    def update(self, layer: list[dict[str, any]] | dict[str, any]):
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

        # Check if tree already completed
        if self.finished:
            print(
                "Beam Tree is already finished. Access best beam with `object.best_beam`."
            )
            return

        if len(self.tree) == 0 and type(layer[0]) is dict:  # Check if root
            # root

            self.initialized = True

            proposed_objs = []
            for token in layer:
                # TODO HERE when creating logprob node, apply lambda penalty
                # TODO directly on logprob.
                root = Node(token["ids"], token["word"], None, token["logprob"])
                proposed_objs.append(root)

            # rank the best
            chosen_beams, top_results = self.pick_best_paths(proposed_objs)

            for chosen_beam, item in zip(chosen_beams, top_results):
                self.tree.append(item)  # Only add selected nodes to tree
                # self.beams.append(
                #     Beam(
                #         [item.id],
                #         item.token,
                #         item.probability,
                #         self.calculate_normalized_probability(
                #             item.probability,
                #             self.alpha,
                #         ),
                #     )
                # )
                self.beams.append(chosen_beam)
                # self.beams.append(item.token)
                # self.beam_ids.append([item.id])
                self.beam_obj.append(item)

        else:
            # normal sequence generation

            proposed_objs = []
            for i, beam_i_proposal in enumerate(layer):
                for token in beam_i_proposal:
                    # TODO HERE when creating logprob node, apply lambda penalty
                    # TODO directly on logprob.

                    node = Node(
                        token["ids"],
                        token["word"],
                        self.beam_obj[i],
                        token["logprob"],
                    )
                    self.beam_obj[i].children.append(node)
                    proposed_objs.append(node)

            chosen_beams, top_results = self.pick_best_paths(proposed_objs)

            if self.finished:
                # Returning best beam and best completed beams (top few)
                return chosen_beams, top_results  # Custom for finishing, just return

            self.beams = []
            # self.beam_ids = []
            self.beam_obj = []
            for chosen_beam, item in zip(chosen_beams, top_results):
                self.tree.append(item)
                # string_sequence, id_sequence = self.build_sequence_from_obj(item)
                # beam_prob = self.calculate_sequence_prob(item)
                # self.beams(
                #     Beam(
                #         id_sequence,
                #         string_sequence,
                #         beam_prob,
                #         self.calculate_normalized_probability(
                #             beam_prob, len(id_sequence), self.alpha
                #         ),
                #         self.check_beam_completion(id_sequence, self.eos_token_id),
                #     )
                # )
                self.beams.append(chosen_beam)
                # self.beams.append(string_sequence)
                # self.beam_ids.append(id_sequence)
                self.beam_obj.append(item)

        # TODO check for each beam when doing pick best paths if its completed or worse than best completed, dont use those, if no candidates, then cancel early and finalize result.

        # TODO check for completed beams MOVE TO PICK BEST PATHS
        # for beam in self.beams:
        #     if beam.completed:
        #         if beam.normalized_logprob > self.best_beam_probability:
        #             self.best_beam_probability = beam.normalized_logprob

        #         self.best_completed_beams.append(beam)

        return self.beams

    def check_beam_completion(self, id_sequence: list[int], eos_id):
        if eos_id in id_sequence:
            return True
        else:
            return False

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
            # print("Current Node")
            # print(part.token)

            if part.parent == None:
                return part.token, part.id

            str_part, id_part = build_sequence(part.parent)

            return str_part + part.token, id_part + part.id

        sequence_build, sequence_ids_build = build_sequence(obj)

        return sequence_build, sequence_ids_build

    def calculate_sequence_prob(self, obj: object):
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
        return obj.probability + self.calculate_sequence_prob(
            obj.parent
        )  # sum of logprobs

    def finalize_results(self):
        """
        Finds and sets the best final chosen beam and finishes beam search
        """

        best_beam = max(
            self.best_completed_beams, key=lambda obj: obj.normalized_logprob
        )

        self.best_beam = best_beam
        self.finished = True

    def add_new_best_beam(self, beam: object):
        self.best_beam_probability = beam.normalized_logprob
        self.best_completed_beams.append(beam)

    def prune_proposed_beams(self, proposed_beams: list[object]) -> list[object]:
        """
        Prunes list of beam objects, returns same format.
        Flashcard limits and prunes beam that has no chance.
        """
        # TODO Add a hyperparameter that stops early if an optimal sequence is found early (for example if normalized logprob is good enough, stops overly analyzing for something tiny bit better)

        pruned_beams = []
        # print("Started pruning")

        for beam in proposed_beams:
            # First check if it's completed and the best one yet (last token was EOS), then save before removing
            if beam.completed and beam.normalized_logprob > self.best_beam_probability:
                self.add_new_best_beam(beam)
                # print("FOUND A NEW BEST")
                continue

            # TODO Make argument where you can turn flashcard checking off
            # Remove all words that are not in flashcard list, don't forget to lemmatize
            # TODO make this something imported so I can dynamically use the correct one for each language
            # TODO implement soft constraint

            # last_generated_word = re.split(r"\s+", contractions.fix(beam.sequence))[-1]

            # last_generated_word = re.sub(
            #     r"^\W+|\W+$", "", last_generated_word
            # )  # Remove only spcial characters before and after

            # # print("Started lemmatizing.")
            # doc = self.nlp(last_generated_word)

            # last_generated_word_lemmas = [
            #     item.lemma_ for item in doc
            # ]  # Can be several, usually 1 for the single last word

            # # print("Found lemmas:")
            # # print(last_generated_word_lemmas)

            # # print("Allowed Words:")
            # # print(self.allowed_words)

            # if set(last_generated_word_lemmas) & set(
            #     self.allowed_words
            # ):  # Any overlap between word lemma and allowed words
            #     # print("Found overlap!, Is ok.")
            #     # print(last_generated_word_lemmas)
            #     # print(self.allowed_words)
            #     pass
            # else:
            #     # print("No overlap? PRUUUUNEE")
            #     continue

            # print("Done checking flashcards.")

            # Remove all completed paths
            if beam.completed:
                # print("Pruned a completed one, not best.")
                continue

            # Remove all paths where normalized probability < current best
            if beam.normalized_logprob < self.best_beam_probability:
                # print("Pruned a bad branch")
                continue

            pruned_beams.append(beam)

        # TODO Handle in case it accidentally prunes all of them and there is nothing to continue on.

        # print(f"{len(pruned_beams)} after pruning, but before dummy fill.")

        if not pruned_beams:  # is empty
            self.finalize_results()

        return pruned_beams

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

        # probabilities = []

        proposed_beams = []

        for proposal in proposed:
            string_sequence, id_sequence = self.build_sequence_from_obj(proposal)
            beam_prob = self.calculate_sequence_prob(proposal)

            norm_beam_prob = self.calculate_normalized_probability(
                beam_prob, len(id_sequence), self.alpha
            )
            is_completed = self.check_beam_completion(id_sequence, self.eos_token_id)
            beam_obj = Beam(
                id_sequence,
                string_sequence,
                beam_prob,
                norm_beam_prob,
                proposal,
                is_completed,
            )
            proposed_beams.append(beam_obj)

        # pruning
        pruned_beams = self.prune_proposed_beams(proposed_beams)

        # check if finished, else rank
        if self.finished:
            print("No more paths could be found. Finished.")
            return self.best_beam, self.best_completed_beams

        # rank and keep top beam size

        # sort by
        chosen_beams = sorted(
            pruned_beams, key=lambda obj: obj.normalized_logprob, reverse=True
        )[: self.beam_size]
        top_n_items = [obj.node_end for obj in chosen_beams]

        # top_indices = sorted(
        #     range(len(probabilities)), key=lambda i: probabilities[i], reverse=True
        # )[: self.beam_size]
        # top_n_items = [proposed[i] for i in top_indices]

        if len(chosen_beams) < self.beam_size:
            # add dummies
            diff_amount = self.beam_size - len(chosen_beams)
            for i in range(diff_amount):
                dummy_node = Node([0], "", None, float("-inf"))
                dummy_beam = Beam(
                    [0], "", float("-inf"), float("-inf"), dummy_node, True
                )
                top_n_items.append(dummy_node)
                chosen_beams.append(dummy_beam)
        return chosen_beams, top_n_items

    def apply_word_constraint(self):
        pass

    def calculate_normalized_probability(
        self, log_prob_sum: float, length: int, alpha: float = 0.6
    ) -> float:
        def length_penalty(length: int, alpha: float = 0.6) -> float:
            """
            Wu et al. 2016 (Google NMT) length penalty.
            alpha=0:   no normalization (raw log-prob, biased to short)
            alpha=1:   full normalization (divide by length)
            alpha=0.6: empirically good default for most tasks
            """
            return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

        return log_prob_sum / length_penalty(length, alpha)

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
        # self.beam_ids = []
        self.beam_obj = []

    def visualize_tree(self, filename: str):
        # TODO fix a bug with the last token(s) not showing (prob never added to tree)
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

        u.save(f"figures/trees/{filename}.gv")
        u.attr(size="12,12", dpi="1000")

        # u.format = "png"
        u.render(f"figures/trees/{filename}", view=False, format="png")

        os.remove(f"figures/trees/{filename}")  # removes extra no codex .dot file

"""This module contains an implementation of a class that help /
    to clean orthographic or IPA transcripts of utterances. /
    Crucially, this class will clean utterances by removing or replacing /
    markers. See the file markers.json to see what kinds of markers are /
    accounted.
"""
import re
import string
import json

class UtterancesCleaner :
    """
    This class will clean utterances from CHILDES,\
    by deleting words, patterns, ponctuation or replacing\
    or replacing them by other things.
    """
    def __init__(self, markers_json: str) :
        with open(markers_json, encoding="UTF-8") as markers_file:
            markers = json.load(markers_file)
        self.delete_marker_pattern = '|'.join(markers["marker_to_delete"])
        self.word_contains_delete_pattern = '|'.join(markers["word_contains_delete"])
        self.poncts_to_delete_pattern = '|'.join(markers["poncts_to_delete"])
        self.delete_comments_pattern = r"(\(|\<|\*)(.+?)(\)|\>|\*)"
        self.replace_unk_pattern = r"xxx|xx|yyy|yy|www|ww|[0-9]+|\*"
        self.pattern_letter = re.compile(r"(\s?)([^ ]*)\s\[x (\d+)\]")
        self.pattern_repetition = re.compile(r"(\s?)([^ ]*)\s\[x (\d+)\]")
        self.punctuations = "".join(set(string.punctuation) - {"'"})

    def replace_marker(self, utterance: str, pattern: str, replacement: str="∑") -> list:
        """
        Method that replace some markers by an other symbol

        Parameters
        ----------
        - utterance : str
            Utterance from which markers will be replaced
        - pattern : str
            Regex pattern containing markers to delete from the utterance
        - replacement :
            Symbol that will replace markers
        """
        return " ".join(re.sub(pattern, replacement, word) for word in utterance.split(" "))

    def delete_words(self, utterance: str) -> str:

        """
        Method that delete some words from a given utterance.

        Parameters
        ----------
        - utterance : str
            Utterance from which those words will be removed
        """
        return " ".join(word for word in utterance.split(" ") \
            if not re.match(self.word_contains_delete_pattern, word))

    def remove_ponctuations(self, utterance: str) -> str :
        """
        Remove ponctuations from a given utterance.

        Parameters
        ----------
        - utterance : str
            The utterance from which the punctuation will be removed.

        Returns
        -------
        str :
            The utterance without punctuations.
        """
        return utterance.translate(str.maketrans('', '', self.punctuations))

    def remove_brackets(self, utterance: str) -> str :
        """
        Remove brackets from a given utterance.

        Parameters
        ----------
        - utterance : str
            The utterance from which the brackets will be removed.

        Returns
        -------
        str :
            The utterance without brackets.
        """
        return re.sub(r"[\(\[].*?[\)\]]", '', utterance)

    def handle_repetitions(self, utterance: str) -> str:
        """
        This function will repeat n times some units from\
        a give utterance.

        Parameters
        ----------
        utterance: str
            Utterance from which some units will be repeated.
        """
        while True:
            matched = re.search(self.pattern_repetition, utterance)

            if not matched:
                break

            all_match = matched.group(0)
            separator = matched.group(1)
            word, repetitions = matched.group(2),matched.group(3)
            repeated_word = f"{separator}{' '.join([word] * int(repetitions))}"

            utterance = utterance.replace(all_match, repeated_word, 1)

        return utterance

    def remove_multiple_spaces(self, utterance: str) -> str :
        """
        Remove multiple spaces from a given utterance.

        Parameters
        ----------
        utterance: str
            Utterance from which multiple successive spaces\
            will be replaced.

        Returns
        -------
        - str
            Utterance without multiple successive spaces.
        """
        return re.sub(' +', ' ', utterance)

    def clean(self, utterance: str) -> str :

        """
        Method that clean utterances by deleting or replacing /
        markers.

        Parameters
        ----------
        - utterances : str
            Utterance to clean
        Returns
        -------
        - str
            Cleaned utterance
        """
        utterance = self.handle_repetitions(utterance)
        utterance = self.replace_marker(utterance, self.delete_marker_pattern, "")
        utterance = self.delete_words(utterance)
        utterance = self.replace_marker(utterance, self.poncts_to_delete_pattern, "")
        utterance = self.replace_marker(utterance, self.delete_comments_pattern, "")
        utterance = self.replace_marker(utterance, self.replace_unk_pattern, "")
        utterance = self.remove_brackets(utterance)
        utterance = self.remove_ponctuations(utterance)
        utterance = self.remove_multiple_spaces(utterance)
        utterance = utterance.strip()
        return utterance
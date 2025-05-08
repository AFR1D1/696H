import nltk
from nltk.tree import Tree
import json
import re


def get_sentiment(sentiment):
    sentiment = int(sentiment)  # sanity check
    if sentiment <= 1:
        return "NEGATIVE"
    if sentiment >= 3:
        return "POSITIVE"
    return "NEUTRAL"


# def preprocess_review(text):
#     # Replace  brackets and tokens special chars
#     text = re.sub(r"-LRB-", "(", text)
#     text = re.sub(r"-RRB-", ")", text)
#     text = re.sub(r"-LSB-", "[", text)
#     text = re.sub(r"-RSB-", "]", text)
#     text = re.sub(r"-LCB-", "{", text)
#     text = re.sub(r"-RCB-", "}", text)

#     # inverted commas
#     text = re.sub(r" '", "'", text)  # get rid of extra space
#     text = re.sub(r" , ", ", ", text)  # get rid of extra space
#     text = re.sub(r"\\+", "", text)  # Remove backslashes
#     # text = re.sub(r"`+", "'", text)  # Replace backticks with apostrophes
#     return text


def generate_rationales(sentence, phrases, sentiment):

    extreme_f = False
    no_rationale = True

    # sentence = preprocess_review(sentence)

    # Question
    question = "Classify this movie review into positive or negative: "
    question += f"[{sentence}]"
    question += "\nJustify your decision by giving a well-formed explanation describing your rationale. Divide the review into sub-phrases, and use phrase numbers in your rationale."

    classification = get_sentiment(sentiment)

    # we aren't finetuning with neutral stuff
    if classification != "NEUTRAL":
        # print("***********************************")

        # answer
        answer = f"I would classify this review as {classification}.\n"
        answer += f"To justify my decision, here is my rationale: \n\n"

        # rationale logic
        # phrase breakup
        rationale = f"The review given by the prompt can be broken down into the following units: \n"
        for i, (phrase, label) in enumerate(phrases):
            # rationale += f"Phrase {i+1}: {phrase} [{label}]\n"
            rationale += f"Phrase {i+1}: {phrase}\n"
        rationale += "\n"

        # if there any single word units I want to ignore
        if any(phrase[1] == -1 for phrase in phrases):
            phrases_ignored = [
                str(i + 1) for i, phrase in enumerate(phrases) if phrase[1] == -1
            ]
            rationale += (
                "I decided the following phrase number(s) are irrelevant in classifying the review: "
                + ", ".join(phrases_ignored)
                + ".\nAlso, "
            )

        # there are positive phrases
        if any(phrase[1] == 4 or phrase[1] == 3 for phrase in phrases):
            no_rationale = False
            positive_phrases = [
                str(i + 1)
                for i, phrase in enumerate(phrases)
                if phrase[1] == 4 or phrase[1] == 3
            ]
            rationale += (
                "I decided that the following phrase number(s) are positive: "
                + ", ".join(positive_phrases)
                + ".\n"
            )

        # negative phrases
        if any(phrase[1] == 0 or phrase[1] == 1 for phrase in phrases):
            no_rationale = False
            positive_phrases = [
                str(i + 1)
                for i, phrase in enumerate(phrases)
                if phrase[1] == 0 or phrase[1] == 1
            ]
            rationale += (
                "I decided that the following phrase number(s) are negative: "
                + ", ".join(positive_phrases)
                + ".\n"
            )

        # if there are any extremely positive or negative units
        if any(phrase[1] == 4 or phrase[1] == 0 for phrase in phrases):
            no_rationale = False
            polar_phrases = [
                str(i + 1)
                for i, phrase in enumerate(phrases)
                if phrase[1] == 4 or phrase[1] == 0
            ]
            rationale += (
                "I decided that the following phrase number(s) are extremely positive or negative and important for classification: "
                + ", ".join(polar_phrases)
                + ".\n"
            )
            extreme_f = True

        # slightly negative or positive sentences (neutral-ish reviews)
        if int(sentiment) == 3 or int(sentiment) == 2:
            if not extreme_f:
                rationale += "This review was slightly ambiguous to classify. "

        number_of_positive_phrases = sum(1 for phrase in phrases if phrase[1] > 2)
        number_of_negative_phrases = sum(
            1 for phrase in phrases if phrase[1] < 2 and phrase[1] != -1
        )

        # count based logic
        if (
            classification == "POSITIVE"
            and number_of_negative_phrases < number_of_positive_phrases
        ):
            no_rationale = False
            rationale += "This review had more positive sentences than negative sentences, hence my classification.\n"
        if (
            classification == "NEGATIVE"
            and number_of_negative_phrases > number_of_positive_phrases
        ):
            no_rationale = False
            rationale += "This review had more negative sentences than positive sentences, hence my classification.\n"

        # no logic I can think of
        if no_rationale:
            rationale += "This review was difficult to reason about, so I guessed.\n"
        print("**********************")
        print("QUESTION: " + question)
        print("RESPONSE: " + answer + rationale)
        print("**********************")
        return [{"question": question, "answer": answer + rationale}]
    return []


def make_new_data(data):
    dataset = []
    count = 0
    for line in data.split("\n"):
        if line.strip():
            tree = Tree.fromstring(line.strip())
            sentence = " ".join(tree.leaves())
            if len(sentence) > 80:
                phrases = extract_phrases(tree)
                dataset.extend(generate_rationales(sentence, phrases, tree.label()))
    print(f"Total reviews over 80 characters: {len(dataset)}")
    return dataset


def extract_phrases(tree):
    """Extract all multi-token phrases and their sentiment labels from a tree."""
    phrases = []

    def recurse(subtree, lvl):
        if isinstance(subtree, Tree):
            if lvl == 3:
                # Only include phrases that are more than one word
                leaves = subtree.leaves()
                if len(leaves) > 1:
                    phrase = " ".join(leaves)
                    label = int(subtree.label())
                    phrases.append((phrase, label))
                else:
                    phrase = " ".join(leaves)
                    # unless it is an important word
                    label = int(subtree.label())
                    if label > 1 and label < 3:
                        label = -1
                    phrases.append((phrase, label))
            elif lvl < 3 and len(subtree.leaves()) == 1:
                # no child
                label = int(subtree.label())
                if label > 1 and label < 3:
                    label = -1
                phrases.append((subtree.leaves()[0], label))

            # Continue recursion
            for child in subtree:
                recurse(child, lvl + 1)

    recurse(tree, 0)
    return phrases


def read_data():
    tree_files = ["trees/train.txt", "trees/dev.txt", "trees/test.txt"]
    combined_content = ""
    for file in tree_files:
        with open(file, "r") as infile:
            data = infile.read()
            combined_content += data
    return combined_content


if __name__ == "__main__":
    data = read_data()
    final_dataset = make_new_data(data)
    with open("fine_tune_data_improved.jsonl", "w") as f:
        for example in final_dataset:
            json_line = json.dumps(example)
            f.write(json_line + "\n")

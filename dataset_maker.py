import nltk
from nltk.tree import Tree
import json


def get_sentiment(sentiment):
    sentiment = int(sentiment)  # sanity check
    if sentiment <= 1:
        return "NEGATIVE"
    if sentiment >= 3:
        return "POSITIVE"
    return "NEUTRAL"


def generate_rationales(sentence, phrases, sentiment):

    extreme_f = False
    no_rationale = True

    # Question
    question = "Classify this movie review into positive or negative: "
    question += sentence
    question += "\nJustify your decision by giving a well-formed explanation describing your rationale."

    classification = get_sentiment(sentiment)

    # we aren't finetuning with neutral stuff
    if classification != "NEUTRAL":
        # print("***********************************")

        # answer
        answer = f"I would classify this review as {classification}.\n"
        answer += f"To justify my decision, here is my rationale: \n\n"

        # rationale logic
        # phrase breakup
        rationale = (
            f'The review "{sentence}" can be broken down into the following units: \n'
        )
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
                rationale += "This sentence was slightly ambiguous to classify. "

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
    with open("fine_tune_data.jsonl", "w") as f:
        for example in final_dataset:
            json_line = json.dumps(example)
            f.write(json_line + "\n")



#Sample output:

#{"question": "Classify this movie review into positive or negative: The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal .\nJustify your decision by giving a well-formed explanation describing your rationale.", "answer": "I would classify this review as POSITIVE.\nTo justify my decision, here is my rationale: \n\nThe review \"The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal .\" can be broken down into the following units: \nPhrase 1: The\nPhrase 2: Rock\nPhrase 3: is\nPhrase 4: destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal\nPhrase 5: .\n\nI decided the following phrase number(s) are irrelevant in classifying the review: 1, 2, 3, 5.\nAlso, I decided that the following phrase number(s) are extremely positive or negative and important for classification: 4.\nThis review had more positive sentences than negative sentences, hence my classification.\n"}
#{"question": "Classify this movie review into positive or negative: The gorgeously elaborate continuation of `` The Lord of the Rings '' trilogy is so huge that a column of words can not adequately describe co-writer\\/director Peter Jackson 's expanded vision of J.R.R. Tolkien 's Middle-earth .\nJustify your decision by giving a well-formed explanation describing your rationale.", "answer": "I would classify this review as POSITIVE.\nTo justify my decision, here is my rationale: \n\nThe review \"The gorgeously elaborate continuation of `` The Lord of the Rings '' trilogy is so huge that a column of words can not adequately describe co-writer\\/director Peter Jackson 's expanded vision of J.R.R. Tolkien 's Middle-earth .\" can be broken down into the following units: \nPhrase 1: The\nPhrase 2: gorgeously elaborate continuation\nPhrase 3: of ``\nPhrase 4: The Lord of the Rings '' trilogy\nPhrase 5: is so huge\nPhrase 6: that a column of words can not adequately describe co-writer\\/director Peter Jackson 's expanded vision of J.R.R. Tolkien 's Middle-earth\nPhrase 7: .\n\nI decided the following phrase number(s) are irrelevant in classifying the review: 1, 7.\nAlso, I decided that the following phrase number(s) are extremely positive or negative and important for classification: 2.\nThis review had more positive sentences than negative sentences, hence my classification.\n"}
#{"question": "Classify this movie review into positive or negative: Singer\\/composer Bryan Adams contributes a slew of songs -- a few potential hits , a few more simply intrusive to the story -- but the whole package certainly captures the intended , er , spirit of the piece .\nJustify your decision by giving a well-formed explanation describing your rationale.", "answer": "I would classify this review as POSITIVE.\nTo justify my decision, here is my rationale: \n\nThe review \"Singer\\/composer Bryan Adams contributes a slew of songs -- a few potential hits , a few more simply intrusive to the story -- but the whole package certainly captures the intended , er , spirit of the piece .\" can be broken down into the following units: \nPhrase 1: Singer\\/composer Bryan Adams contributes a slew of songs -- a few potential hits , a few more simply intrusive to the story --\nPhrase 2: but\nPhrase 3: the whole package\nPhrase 4: certainly captures the intended , er , spirit of the piece\nPhrase 5: .\n\nI decided the following phrase number(s) are irrelevant in classifying the review: 2, 5.\nAlso, I decided that the following phrase number(s) are extremely positive or negative and important for classification: 3.\nThis review had more positive sentences than negative sentences, hence my classification.\n"}

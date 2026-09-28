import string


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = []
    start = 0
    for index, char in enumerate(text):
        if char not in ".!?":
            continue
        next_index = index + 1
        if next_index < len(text) and text[next_index] not in " \t\n\r":
            continue
        sentence = text[start:next_index].strip()
        if sentence:
            sentences.append(sentence)
        start = next_index
        while start < len(text) and text[start] in " \t\n\r":
            start += 1
    rest = text[start:].strip()
    if rest:
        sentences.append(rest)
    return sentences


def word_frequencies(text: str) -> dict[str, int]:
    frequencies = {}
    for token in text.lower().split():
        word = token.strip(string.punctuation)
        if word:
            frequencies[word] = frequencies.get(word, 0) + 1
    return frequencies


def count_word(text: str, word: str, *, case_sensitive: bool = False) -> int:
    haystack = text if case_sensitive else text.lower()
    needle = word if case_sensitive else word.lower()
    count = 0
    for token in haystack.split():
        if token.strip(string.punctuation) == needle:
            count += 1
    return count


def numbered_contents(text: str) -> list[str]:
    contents = []
    for line in text.splitlines():
        stripped = line.lstrip()
        index = 0
        while index < len(stripped) and stripped[index].isdigit():
            index += 1
        if index > 0 and index < len(stripped) - 1 and stripped[index] in ".)":
            if stripped[index + 1] == " ":
                contents.append(stripped[index + 2 :])
    return contents


def check_hidden_word(response: str, word: str) -> float:
    return 1.0 if word.lower() in response.lower() else 0.0


def run_check(check_type: str, response: str, params: dict) -> float:
    if check_type == "count_sentences":
        return 1.0 if len(split_sentences(response)) == params["target"] else 0.0
    if check_type == "words_per_sentence_range":
        sentences = split_sentences(response)
        if not sentences:
            return 0.0
        return (
            1.0
            if all(params["min_w"] <= len(sentence.split()) <= params["max_w"] for sentence in sentences)
            else 0.0
        )
    if check_type == "sentences_start_different_letter":
        letters = []
        for sentence in split_sentences(response):
            first = next((char.lower() for char in sentence if char.isalpha()), None)
            if first is None or first in letters:
                return 0.0
            letters.append(first)
        return 1.0 if letters else 0.0
    if check_type == "sentences_contain_long_word":
        sentences = split_sentences(response)
        if not sentences:
            return 0.0
        min_length = params["min_length"]
        for sentence in sentences:
            words = [word.strip(string.punctuation) for word in sentence.split()]
            if not any(len(word) >= min_length for word in words):
                return 0.0
        return 1.0
    if check_type == "min_unique_words":
        return 1.0 if len(word_frequencies(response)) >= params["min_unique"] else 0.0
    if check_type == "max_word_frequency":
        return 1.0 if all(count <= params["max_count"] for count in word_frequencies(response).values()) else 0.0
    if check_type == "keyword_min_count":
        return 1.0 if count_word(response, params["word"]) >= params["min_count"] else 0.0
    if check_type == "keyword_min_count_case_sensitive":
        count = count_word(response, params["word"], case_sensitive=True)
        return 1.0 if count >= params["min_count"] else 0.0
    if check_type == "forbidden_char":
        return 1.0 if params["char"] not in response else 0.0
    if check_type == "forbidden_char_insensitive":
        return 1.0 if params["char"].lower() not in response.lower() else 0.0
    if check_type == "forbidden_word":
        return 1.0 if count_word(response, params["word"]) == 0 else 0.0
    if check_type == "all_lowercase":
        letters = [char for char in response if char.isalpha()]
        return 1.0 if letters and all(char.islower() for char in letters) else 0.0
    if check_type == "all_uppercase":
        letters = [char for char in response if char.isalpha()]
        return 1.0 if letters and all(char.isupper() for char in letters) else 0.0
    if check_type == "ends_with_phrase":
        return 1.0 if response.strip().endswith(params["phrase"]) else 0.0
    if check_type == "count_numbered_items":
        return 1.0 if len(numbered_contents(response)) == params["target"] else 0.0
    if check_type == "numbered_items_word_range":
        contents = numbered_contents(response)
        if not contents:
            return 0.0
        return (
            1.0
            if all(params["min_w"] <= len(content.split()) <= params["max_w"] for content in contents)
            else 0.0
        )
    if check_type == "numbered_items_one_sentence":
        contents = numbered_contents(response)
        return 1.0 if contents and all(len(split_sentences(content)) == 1 for content in contents) else 0.0
    if check_type == "numbered_items_contain_long_word":
        contents = numbered_contents(response)
        if not contents:
            return 0.0
        min_length = params["min_length"]
        for content in contents:
            words = [word.strip(string.punctuation) for word in content.split()]
            if not any(len(word) >= min_length for word in words):
                return 0.0
        return 1.0
    return 0.0

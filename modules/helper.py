#----------------------------------------------------
# Imports
#----------------------------------------------------
import imports

#----------------------------------------------------
# Helper Functions
#----------------------------------------------------
def extract_sentiment_score(text: str) -> float:
    """
    Computes the sentiment score from a given text.

    text -> str
        Given input text

    Returns -> float
        Returns the sentiment score

    Example
        extract_sentiment_score("I love this!")
    """
    analyzer = imports.SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]

def remove_emoji(text: str) -> str:
    """
    Removes the emojis from a text

    text -> str
        Given input text

    Returns -> str
        Returns de-emojied text

    Example
        remove_emoji("I love this! \U0001f602")
    """
    regrex_pattern = imports.re.compile(pattern = "["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags = imports.re.UNICODE)
    return regrex_pattern.sub(r'',text)

def calc_limit_lwrbound(num_scores: list) -> float:
    """
    Calculates the optimal lowerbound for the y_lim when zooming in on values

    num_scores -> list
        Given iterable of values

    Returns -> float
        Returns the optimal lowerbound

    Example
        calc_limit_lwrbound([2,3,4,5])
    """
    min_value = min(num_scores)
    max_value = max(num_scores)
    range_value = max_value - min_value
    return min_value - range_value * 0.2

def get_ngrams(series, n) -> dict:
    """
    Extracts a dictionary of the n-grams in a given text

    text -> Pandas.Series
        The given text

    n -> int
        The n in n-grams
    
    Returns -> dict
        The frequency dictionary of the n-gram

    Example
        get_ngrams(df["Text"], 3)
    """
    ngram_dict = {}
    for text_row in series:
        ngrams_list = list(imports.ngrams(text_row.split(" "), n))
        for gram in ngrams_list:
            if gram in ngram_dict:
                ngram_dict[gram] += 1
            else:
                ngram_dict[gram] = 1
    return ngram_dict

#----------------------------------------------------
# Main
#----------------------------------------------------
def main():
    return None

#----------------------------------------------------
# Entry
#----------------------------------------------------
if __name__ == "__main__":
    main()
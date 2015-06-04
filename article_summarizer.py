__author__ = 'bene'
from goose import Goose
from requests import get
from requests.exceptions import RequestException
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import string
from collections import Counter
from operator import itemgetter


stop_words = set([
    "-", " ", ",", ".", "'", "re", "ll", "ve", "about", "above",
    "above", "across", "after", "afterwards", "again", "against", "all",
    "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and",
    "another", "any", "anyhow", "anyone", "anything", "anyway",
    "anywhere", "are", "around", "as", "at", "back", "be", "became",
    "because", "become", "becomes", "becoming", "been", "before",
    "beforehand", "behind", "being", "below", "beside", "besides",
    "between", "beyond", "both", "bottom", "but", "by", "call", "can",
    "cannot", "can't", "co", "con", "could", "couldn't", "de",
    "describe", "detail", "did", "do", "done", "down", "due", "during",
    "each", "eg", "eight", "either", "eleven", "else", "elsewhere",
    "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty",
    "fill", "find", "fire", "first", "five", "for", "former",
    "formerly", "forty", "found", "four", "from", "front", "full",
    "further", "get", "give", "go", "got", "had", "has", "hasnt",
    "have", "he", "hence", "her", "here", "hereafter", "hereby",
    "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "into", "is", "it", "its", "it's", "itself", "just", "keep", "last",
    "latter", "latterly", "least", "less", "like", "ltd", "made", "make",
    "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",
    "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
    "name", "namely", "neither", "never", "nevertheless", "new", "next",
    "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing",
    "now", "nowhere", "of", "off", "often", "on", "once", "one", "only",
    "onto", "or", "other", "others", "otherwise", "our", "ours",
    "ourselves", "out", "over", "own", "part", "people", "per",
    "perhaps", "please", "put", "rather", "re", "said", "same", "see",
    "seem", "seemed", "seeming", "seems", "several", "she", "should",
    "show", "side", "since", "sincere", "six", "sixty", "so", "some",
    "somehow", "someone", "something", "sometime", "sometimes",
    "somewhere", "still", "such", "take", "ten", "than", "that", "the",
    "their", "them", "themselves", "then", "thence", "there",
    "thereafter", "thereby", "therefore", "therein", "thereupon",
    "these", "they", "thickv", "thin", "third", "this", "those",
    "though", "three", "through", "throughout", "thru", "thus", "to",
    "together", "too", "top", "toward", "towards", "twelve", "twenty",
    "two", "un", "under", "until", "up", "upon", "us", "use", "very",
    "via", "want", "was", "we", "well", "were", "what", "whatever",
    "when", "whence", "whenever", "where", "whereafter", "whereas",
    "whereby", "wherein", "whereupon", "wherever", "whether", "which",
    "while", "whither", "who", "whoever", "whole", "whom", "whose",
    "why", "will", "with", "within", "without", "would", "yet", "you",
    "your", "yours", "yourself", "yourselves", "the", "reuters", "news",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "rappler", "rapplercom", "inquirer", "yahoo", "home", "sports",
    "1", "10", "2012", "sa", "says", "tweet", "pm", "home", "homepage",
    "sports", "section", "newsinfo", "stories", "story", "photo",
    "2013", "na", "ng", "ang", "year", "years", "percent", "ko", "ako",
    "yung", "yun", "2", "3", "4", "5", "6", "7", "8", "9", "0", "time",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "government", "police", "a", "b", "c", "d", "e", "f", "g", "h", "i",
    "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
    "x", "y", "z"
])



class ArticleSummary():

    def __init__(self, url):
        self.url = url

        self.summary = None
        self.sentence_count = None
        self.stripped_title = None
        self.keywords = None

        self.article = self.get_article()
        self.summary = self.summarize_article()

    def get_article(self):
        response = get(url=self.url)
        if response.status_code == 200:
            article = Goose().extract(raw_html=response.text)
        else:
            raise RequestException("Unsuccesful status code returned (%s)" % response.status_code)
        return article

    def summarize_article(self):
        self.summary = self.parse_body_to_sentences()
        self.sentence_count = len(self.summary)
        self.stripped_title = self.stripped_words(self.article.title)
        self.keywords, self.word_count = self.top_n_keywords(10)
        for sentence in self.summary:
            sentence['title_relevance'] = sum([1 for word in sentence['word_list'] if word in self.stripped_title]) /\
                (len(self.stripped_title) * 1.0)
            sentence['position_score'] = self.position_score(sentence)
            sentence['length_score'] = (25.0 - abs(25.0 - sentence['word_count'])) / 25.0
            sentence['keywords_density'] = self.keyword_density(sentence)
            sentence['keyword_distance'] = self.keyword_distance(sentence)

            total_score = sentence['title_relevance'] * 1.0 + \
                sentence['position_score'] * 1.0 + \
                sentence['length_score'] * 1.0 + \
                sentence['keywords_density'] * 3.0 + \
                sentence['keyword_distance'] * 2.5

            sentence['score'] = total_score
        return sorted(self.summary, key=lambda x: -x['score'])

    def position_score(self, _sentence):
        n = _sentence['position'] / (self.sentence_count * 1.0)

        if 0 < n <= 0.1:
            return 0.17
        elif 0.1 < n <= 0.2:
            return 0.23
        elif 0.2 < n <= 0.3:
            return 0.14
        elif 0.3 < n <= 0.4:
            return 0.08
        elif 0.4 < n <= 0.5:
            return 0.05
        elif 0.5 < n <= 0.6:
            return 0.04
        elif 0.6 < n <= 0.7:
            return 0.06
        elif 0.7 < n <= 0.8:
            return 0.04
        elif 0.8 < n <= 0.9:
            return 0.04
        elif 0.9 < n <= 1.0:
            return 0.15
        else:
            return 0

    def keyword_density(self, _sentence):
        """
            Score the sentence based on the ratio of keywords it contains and the score of those
            keywords based on their frequency
        """
        if len(_sentence['word_list']) <= 0:
            return 0

        score = sum([self.keywords[word] for word in _sentence['word_list'] if word in self.keywords])
        return score / len(_sentence['word_list'])

    def top_n_keywords(self, n=15):
        word_count = sum([s['word_count'] for s in self.summary])

        word_frequency = Counter(word for s in self.summary for word in s['word_list'])
        min_size = min(n, len(word_frequency))
        keywords = {x: (y * 1.5 / word_count + 1) for x, y in word_frequency.most_common(min_size)}

        return keywords, word_count

    def keyword_distance(self, _sentence):
        words = _sentence['word_list']
        if len(words) == 0:
            return 0

        summ = 0
        first = []

        for i, word in enumerate(words):
            if word in self.keywords:
                score = self.keywords[word]
                if len(first) == 0:
                    first = [i, score + 1]
                else:
                    second = first
                    first = [i, score + 1]
                    dif = first[0] - second[0]
                    summ += first[1] * second[1] / (dif ** 2)
        k = len(set(self.keywords.keys()).intersection(set(words))) + 1

        return (1 / k * (k + 1.0)) * summ

    def stripped_words(self, original_sentence):
        _sentence = filter(self.printable_char_filter, original_sentence)
        _sentence = _sentence.replace(u'\u2013', ' ')
        _sentence = _sentence.replace(u'\u2014', ' ')
        tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        tokens = tokenizer.tokenize(_sentence)
        return [word.lower() for word in tokens if word.lower() not in stop_words]

    @staticmethod
    def printable_char_filter(c):
        if c in string.printable and c not in string.punctuation:
            return c
        else:
            return ' '

    def parse_body_to_sentences(self):
        _sentences = sent_tokenize(self.article.cleaned_text)
        return [{"raw_text": sentence,
                 "word_list": self.stripped_words(sentence),
                 "word_count": len(sentence.split(" ")),
                 "position": i} for i, sentence in enumerate(_sentences)]

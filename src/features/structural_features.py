import nltk
from nltk.tokenize import RegexpTokenizer




def generate_structural_features(review_text):
	sentence_tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
	sentences = sentence_tokenizer.tokenize(review_text)

	word_tokenizer = RegexpTokenizer(r'\w+')
	f1 = len(word_tokenizer.tokenize(review_text.strip())) # Total number of tokens
	f2 = len(sentences)
	question_sentences = 0
	for sentence in sentences:
		if sentence.endswith('?'):
			question_sentences += 1

	if f2 == 0:
		print 'yolo'+review_text
		f3, f5 = 0, 0
	else:
		f3 = 1.0 * f1 / f2 # Average number of words per sentence
		f5 = 1.0 * question_sentences / f2 # Ratio of question sentences

	f4 = review_text.count('!')

	return [f1, f2, f3, f4, f5]

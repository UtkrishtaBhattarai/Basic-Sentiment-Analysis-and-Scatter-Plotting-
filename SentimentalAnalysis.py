import re
import matplotlib.pyplot as plt


def analyze_sentiment(text):
    # Define positive and negative words to train the algorithm
    positive_words = ['love', 'happiness', 'joy', 'excitement',
                      'wonderful', 'fantastic', 'awesome', 'amazing', 'incredible', 'great','nice'
                      'excellent', 'fabulous', 'delightful', 'pleasure', 'good', 'satisfying', 'beneficial', 
                      'admirable', 'brilliant', 'creative', 'courageous', 'dazzling', 'dynamic', 'effervescent',
                      'efficient', 'energetic', 'enthusiastic', 'extraordinary', 'fascinating', 'graceful', 'harmonious',
                      'helpful', 'inspirational', 'intelligent', 'intriguing', 'marvelous', 'motivating', 'optimistic', 'passionate',
                      'peaceful', 'perfect', 'radiant', 'remarkable', 'sensational', 'spectacular', 'terrific', 'thrilling',
                      'triumphant', 'uplifting', 'happy']

    negative_words = ['hate', 'sad', 'angry', 'bad', 'awful', 'terrible', 'horrible',
                      'unpleasant', 'annoying', 'frustrating', 'disappointing', 'displeasing', 'unfortunate',
                      'unsatisfactory', 'depressing', 'heartbreaking', 'miserable', 'painful', 'pathetic', 'regrettable',
                      'sorrowful', 'tragic', 'unhappy', 'unlucky', 'damaging', 'defective', 'disgusting', 'dreadful', 'gross',
                      'inadequate', 'inferior', 'insulting', 'lousy', 'mediocre', 'nasty', 'offensive', 'poor', 'repulsive',
                      'shocking', 'stupid', 'ugly', 'underwhelming', 'unpleasant', 'unsatisfactory', 'upsetting', 'vile',
                      'wretched', 'worst', 'wrong']

    # Tokenize the text into individual words
    #Make use of regex to find the text or word we are looking for. 
    words = re.findall(r'\b\w+\b', text.lower())

    # Calculate the sentiment score based on the frequency of positive and negative words
    #Making use of regex here to differenciate the positive and negative words. 
    num_positive_words = sum(1 for word in words if re.match(fr'\b({ "|".join(positive_words) })\b', word))
    num_negative_words = sum(1 for word in words if re.match(fr'\b({ "|".join(negative_words) })\b', word))
    sentiment_score = (num_positive_words - num_negative_words) / len(words)

    # Classify the sentiment as positive, negative, or neutral
    # Now making simple use of if else algorithm to show the
    if sentiment_score > 0:
        sentiment = "positive"
        color = 'green'
    elif sentiment_score < 0:
        sentiment = "negative"
        color = 'red'
    else:
        sentiment = "neutral"
        color = 'blue'

    # Return the sentiment and score
    return sentiment, sentiment_score, color



sentences = ["I love this movie! ",
             "This food is terrible.",
             "She is so sad about.",
             "I'm angry that my flight was cancelled.",
             "This vacation was wonderful."]

# Analyze sentiment for multiple sentences
results = [analyze_sentiment(sentence) for sentence in sentences]

# Plot the results in a bar chart
sentiments, scores, colors = zip(*results)
print(results)
plt.figure(figsize=(25, 10))

print(sentences)
plt.bar(sentences, scores, color=colors)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Sentences')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis')
plt.show()
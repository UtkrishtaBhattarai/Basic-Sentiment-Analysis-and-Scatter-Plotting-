from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Define the training data
training_data = [
    ("I love this product", "positive"),
    ("This is a great movie", "positive"),
    ("The service was terrible", "negative"),
    ("I hate this restaurant", "negative"),
    ("The weather is nice today", "neutral"),
    ("I feel very terrible today", "negative"),
    ("I'm feeling happy today", "positive"),
    ("This is terrible man. ", "negative"),
    ("I feel neutral about this topic", "neutral")
]

# Split the data into text and labels
text = [data[0] for data in training_data]
labels = [data[1] for data in training_data]

# Create a pipeline for the classification model
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Fit the model on the training data
pipeline.fit(text, labels)

# Use the model to predict the sentiment of new text
new_text = ["This is a terrible product", "I'm not feeling happy today"]
predicted_labels = pipeline.predict(new_text)
print(predicted_labels)

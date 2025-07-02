import pandas as pd
from transformers import pipeline

# Load feedback data
df = pd.read_csv("feedback.csv")

# Create zero-shot classifier pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define your categories
candidate_labels = ["Academics", "Facilities", "Administration"]

# Classify each feedback entry
predicted_labels = []
for feedback in df['feedback']:
    result = classifier(feedback, candidate_labels)
    predicted_labels.append(result['labels'][0])  # take top predicted label

# Add predictions to dataframe
df['predicted_category'] = predicted_labels

# Save to new CSV
df.to_csv("classified_feedback.csv", index=False)

print("Classification completed! Check classified_feedback.csv")

import matplotlib.pyplot as plt

# Count per category
counts = df['predicted_category'].value_counts()

# Plot
counts.plot(kind='bar')
plt.title('Feedback Category Distribution')
plt.xlabel('Category')
plt.ylabel('Number of Feedbacks')
plt.show()

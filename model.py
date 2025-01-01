import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


train_data = pd.read_csv('Twitter_Data.csv')
train_data.dropna(axis=0, how='any', inplace=True)

def remove_emoji(text):
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"  
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_url(text):
    url_pattern  = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(r'', text)

def clean_text(text):
    text = str(text)
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr = text1.split()
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and len(w) > 2)])
    return text2.lower()

train_data['clean_text'] =train_data['text'].apply(clean_text)
train_data['clean_text'] = train_data['clean_text'].apply(remove_url)
train_data['clean_text'] = train_data['clean_text'].apply(remove_emoji)

train_data['Num_words_text'] = train_data['text'].apply(lambda x: len(str(x).split()))
train_data = train_data[train_data['Num_words_text'] > 2]

# Encode sentiments to integer
label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['sentiment'])

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(train_data['clean_text'], train_data['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

X_train_tensor = torch.FloatTensor(X_train_vectorized)
X_test_tensor = torch.FloatTensor(X_test_vectorized)
y_train_tensor = torch.LongTensor(y_train.values)
y_test_tensor = torch.LongTensor(y_test.values)

# Model Definition: Modify to handle the sequence input for LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # LSTM expects (batch_size, seq_len, input_size), so we reshape accordingly
        x = x.unsqueeze(1)  # Add sequence length dimension (1 in this case)
        out, (hn, cn) = self.lstm(x)  # Process through LSTM
        out = self.fc1(hn[-1])  # Take the last hidden state
        out = self.leaky_relu(out)  # Apply Leaky ReLU
        out = self.fc2(out)  # Final output layer
        return out

# Initialize model parameters
input_size = X_train_tensor.shape[1]  # 1000 features from TF-IDF
hidden_size = 64
num_classes = len(np.unique(y_train))  # 3 classes for sentiment

# Instantiate the model
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 8

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i in tqdm(range(0, len(X_train_tensor), 64)):
        X_batch = X_train_tensor[i:i + 64]
        y_batch = y_train_tensor[i:i + 64]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        
        # Compute loss
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        accuracy = (outputs.argmax(dim=1) == y_batch).float().mean()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(X_train_tensor):.4f}, Accuracy: {accuracy:.4f}')


from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import os


tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

df = pd.read_csv('data/tweets_main_sentiment.csv')
X = df['content']
y = df['sentiment']

df.head()

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

class TwitterDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets.iloc[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',  # changed from pad_to_max_length=True
            truncation=True,  # added this line
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(X, y, tokenizer, max_len, batch_size):
    ds = TwitterDataset(
        tweets=X,
        labels=y,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

BATCH_SIZE = 16
MAX_LEN = 128

train_data_loader = create_data_loader(X_train, y_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(X_val, y_val, tokenizer, MAX_LEN, BATCH_SIZE)

device = "cuda:0"
model = model.to(device)

EPOCHS = 10

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss().to(device)
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
for epoch in range(EPOCHS):
    for data in train_data_loader:
    
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for data in val_data_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs[0], dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.shape[0]

print(f'Accuracy: {correct_predictions.double() / total_predictions}')



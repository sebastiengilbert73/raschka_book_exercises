# Cf. p. 513
import logging
import torch
from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
import re
from collections import Counter, OrderedDict
from torchtext.vocab import vocab

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def tokenizer(text):
    text = re.sub('<[^>]*', '', text)
    emoticons = re.findall(
        '(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower()
    )
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized

class RNN_test(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
        """
        self.rnn = torch.nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        """
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = hidden[-1, :, :]  # We use the final hidden state from the last hidden layer
                                # as the input to the fully connected layer
        out = self.fc(out)
        return out

class RNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = torch.nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = torch.nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(fc_hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = torch.nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train(model, dataloader, optimizer, loss_fn):
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += (
            (pred >= 0.5).float() == label_batch
        ).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += (
                (pred >= 0.5).float() == label_batch
            ).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)


def main():
    logging.info("predicting_sentiment_imdb.main()")

    train_dataset = IMDB(split='train')
    test_dataset = IMDB(split='test')

    # Step 1: Create the datasets
    torch.manual_seed(1)
    train_dataset, valid_dataset = random_split(list(train_dataset), [20000, 5000])

    # Step 2: Find unique tokens (words)
    token_counts = Counter()
    for label, line in train_dataset:
        tokens = tokenizer(line)
        token_counts.update(tokens)
    logging.info(f"Vocab-size: {len(token_counts)}")

    # Step 3: Encoding each unique token into integers
    sorted_by_freq_tuples = sorted(
        token_counts.items(), key=lambda x: x[1], reverse=True
    )
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocabulary = vocab(ordered_dict)  # In the book, the variable name 'vocab' is used
    vocabulary.insert_token("<pad>", 0)
    vocabulary.insert_token("<unk>", 1)
    vocabulary.set_default_index(1)

    # Step 3-A: Define the functions for transformation
    text_pipeline = lambda x: [vocabulary[token] for token in tokenizer(x)]
    label_pipeline = lambda x: 1. if x == 'pos' else 0.  # Positive => 1.0; Negative => 0.0

    def collate_batch(batch):
        label_list, text_list, lengths = [], [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            lengths.append(processed_text.size(0))
        label_list = torch.tensor(label_list)
        lengths = torch.tensor(lengths)
        padded_text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        return padded_text_list, label_list, lengths

    # Take a small batch
    from torch.utils.data import DataLoader
    """dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)

    text_batch, label_batch, length_batch = next(iter(dataloader))
    logging.info(f"text_batch = \t{text_batch}")
    logging.info(f"label_batch = \t{label_batch}")
    logging.info(f"length_batch = \t{length_batch}")
    logging.info(f"text_batch.shape = {text_batch.shape}")
    """
    batch_size = 32
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    embedding = torch.nn.Embedding(
        num_embeddings=10,
        embedding_dim=3,
        padding_idx=0
    )
    # A batch of 2 samples of 4 indices each
    text_encoded_input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 0]])
    logging.info(f"embedding(text_encoded_input) = {embedding(text_encoded_input)}")

    model = RNN_test(64, 32)
    logging.info(f"model:\n{model}")
    output_tsr = model(torch.randn(5, 3, 64))
    logging.info(f"output_tsr.shape = {output_tsr.shape}")

    vocab_size = len(vocabulary)
    embed_dim = 20
    rnn_hidden_size = 64
    fc_hidden_size = 64
    torch.manual_seed(1)
    model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
    logging.info(f"model:\n{model}")

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    torch.manual_seed(1)
    for epoch in range(num_epochs):
        acc_train, loss_train = train(model, train_dl, optimizer, loss_fn)
        acc_valid, loss_valid = evaluate(model, valid_dl, loss_fn)
        logging.info(f"Epoch {epoch} train_accuracy: {acc_train:.4f}; val_accuracy: {acc_valid:.4f}")

    acc_test, _ = evaluate(model, test_dl, loss_fn)
    logging.info(f"test_accuracy: {acc_test:.4f}")

if __name__ == '__main__':
    main()
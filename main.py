import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, TensorDataset

# Определим параметры
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1

class TransformerModel(nn.Module):
    """Модель трансформера для классификации языков."""
    def __init__(self, vocab_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 5)  # 5 классов

    def forward(self, src):
        embedded = self.embedding(src)
        embedded = embedded.permute(1, 0, 2)
        output = self.transformer(embedded)
        output = output.mean(dim=0)
        return self.fc_out(output)

def load_data():
    """Загрузка данных из файлов."""
    with open('data/sentences.txt', 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    with open('data/labels.txt', 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    return sentences, labels

def yield_tokens(data_iter):
    tokenizer = get_tokenizer('basic_english')
    for sentence in data_iter:
        yield tokenizer(sentence)

def encode_sentences(sentences, vocab):
    return [torch.tensor(vocab(tokenizer(sentence))) for sentence in sentences]

def main():
    # Загрузка данных
    sentences, labels = load_data()

    # Создание словаря
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(sentences), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # Преобразование предложений в индексы
    encoded_sentences = encode_sentences(sentences, vocab)

    # Создание батчей
    X = torch.nn.utils.rnn.pad_sequence(encoded_sentences, batch_first=True)
    y = torch.tensor(labels)

    # Создание датасета и загрузчика данных
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Создаем модель
    model = TransformerModel(len(vocab))

    # Определяем оптимизатор и функцию потерь
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Обучение
    model.train()
    for epoch in range(10):
        for batch in dataloader:
            src, tgt = batch
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Пример использования модели для предсказания
    model.eval()
    with torch.no_grad():
        test_sentence = "Карандаш лежит на столе"
        encoded_test = torch.tensor(vocab(tokenizer(test_sentence)))
        encoded_test = encoded_test.unsqueeze(0)
        output = model(encoded_test)
        predicted_class = output.argmax(dim=1)
        print("Predicted class index:", predicted_class.item())

if __name__ == "__main__":
    main()
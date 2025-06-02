import json
import torch
import pickle
from torch.utils.data import Dataset

class DataBundle:
    def __init__(self, base_path="../data"):
        self.base_path = base_path

        self.golden_set              = self.json_load("golden_set.json")
        self.partitiioned_golden_set = self.json_load("partitioned_golden_set.json")
        self.anchors                 = self.json_binary("anchors.json")

        self.pos_weight         = self.pkl_load("pos_weight.pkl")

        self.vocab_zh           = self.pkl_load("vocab_zh.pkl")
        self.zh_embedding       = self.pkl_load("zh_embedding.pkl")
        self.word_to_index_zh   = self.pkl_load("word_to_index_zh.pkl")
        self.idx_to_embed_zh    = self.pkl_load("idx_to_embed_zh.pkl")
        
        self.vocab_en           = self.pkl_load("vocab_en.pkl")
        self.en_embedding       = self.pkl_load("en_embedding.pkl")
        self.word_to_index_en   = self.pkl_load("word_to_index_en.pkl")
        self.idx_to_embed_en    = self.pkl_load("idx_to_embed_en.pkl")

        self.training_data      = self.pkl_load("training_data.pkl")
        self.validation_data    = self.pkl_load("validation_data.pkl")
        self.testing_data       = self.pkl_load("testing_data.pkl")
        self.training_index      = self.pkl_load("training_index.pkl")
        self.validation_index    = self.pkl_load("validation_index.pkl")
        self.testing_index       = self.pkl_load("testing_index.pkl")

    def pkl_load(self, filename):
        with open(f"{self.base_path}/{filename}", "rb") as f:
            return pickle.load(f)
    
    def json_load(self, filename):
        with open(f"{self.base_path}/{filename}", encoding="utf-8-sig") as f:
            return json.load(f)
    
    def json_binary(self, filename):
        with open(f"{self.base_path}/{filename}", 'rb') as f:
            return json.load(f)

data = DataBundle()

# Define dataset to load to dataloader
class TrainData(Dataset):

    def __init__(self, trained_data):
        """Loads the data from the pretrained model"""
        self.data = trained_data

    def __getitem__(self, idx):
        """Returns the datapoint at a given index"""
        return self.data[idx]

    def __len__(self):
        """Returns the number of datapoints in the dataset"""
        return len(self.data)
    
# Define collate_batch for batch of training samples
def wordToindex(word):
    """Get the corresponding index for the Chinese word"""
    return data.word_to_index_zh[word]

def collate_batch(batch):
    """Converts a batch of data into packed PyTorch tensor format,
    and collates the results by index, word, and one-hot vector
    for use in an Autoencoder.
    """
    # Initialize lists that separate out the 3 components
    word_list       = list()
    index_list      = list()
    
    for word in batch:
        # Convert to PyTorch format
        index   = wordToindex(word)

        # Add converted data to separate component lists
        index_list.append(index)
        word_list.append(word)

    # Convert to mini-batch tensors
    index_tensor = torch.tensor(index_list, dtype=torch.int64)

    return (word_list, index_tensor)

def load_data(train, batch_size):
    """
    return train & validation loader if train is True
    ow return test loader
    """
    train_data = TrainData(data.training_data)
    valid_data = TrainData(data.validation_data)
    test_data  = TrainData(data.testing_data)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                collate_fn=collate_batch)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                collate_fn=collate_batch)
    test_loader  = torch.utils.data.DataLoader(dataset=test_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                collate_fn=collate_batch)
    if train:
        return train_loader, valid_loader
    else:
        return test_loader
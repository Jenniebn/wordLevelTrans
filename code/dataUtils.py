import json
import torch
import pickle
from torch.utils.data import Dataset

class DataBundle:
    def __init__(self, base_path="./data"):
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

        self.vocab_en = list(self.word_to_index_en.keys())
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_zh = list(self.word_to_index_zh.keys())
        self.vocab_size_zh = len(self.vocab_zh)

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

class EnZhTrainData(Dataset):

    def __init__(self, data):
        """
        Loads the data in format
        data = {"en_index:..., zh_index:..."}
        en_list = [('petard', 67815), ('relinquishment', 50663),...]
        zh_list = [[('融化', 8830)], [('自制', 6059), ('即席', 82036)],...]
        """
        self.data    = data
        self.en_list = self.data["en_index"]
        self.zh_list = self.data["zh_index"]

    def __getitem__(self, idx):
        """Returns the datapoint at a given index"""
        en_index = self.en_list[idx][1]
        zh_onehot = torch.zeros(data.vocab_size_zh)
        for pair in self.zh_list[idx]:
            zh_onehot[pair[1]] = 1
        sample = {"en_index": en_index, "zh_index": zh_onehot}
        return sample

    def __len__(self):
        """Returns the number of datapoints in the dataset"""
        return len(self.data["en_index"])

class ZhZhTrainData(Dataset):

    def __init__(self, trained_data):
        """Loads the data from the pretrained model"""
        self.data = trained_data

    def __getitem__(self, idx):
        """Returns the datapoint at a given index"""
        return self.data[idx]

    def __len__(self):
        """Returns the number of datapoints in the dataset"""
        return len(self.data)

def enzh_collate_fn(batch):
    """Converts a batch of data into packed PyTorch tensor format
    """
    # Initialize lists that separate out the 3 components
    en_index_list  = list()
    zh_index_list  = list()

    for pair in batch:
        en_index = pair["en_index"]
        zh_index = pair["zh_index"]
        # Convert to PyTorch format
        # Add converted data to separate component lists
        en_index_list.append(en_index)
        zh_index_list.append(zh_index)

    # Convert to mini-batch tensors
    en_index_tensor = torch.tensor(en_index_list).to(torch.int64)
    zh_index_tensor = torch.stack(zh_index_list, dim=0).to(torch.int64)

    return (en_index_tensor, zh_index_tensor)
    
# Define collate_batch for batch of training samples
def wordToindex(word):
    """Get the corresponding index for the Chinese word"""
    return data.word_to_index_zh[word]

def zhzh_collate_fn(batch):
    """Converts a batch of data into packed PyTorch tensor format,
    and collates the results by index, word, and one-hot vector
    for use in an Autoencoder.
    """
    # Initialize lists that separate out the 3 components
    index_list      = list()
    word_list       = list()

    for word in batch:
        # Convert to PyTorch format
        index   = wordToindex(word)

        # Add converted data to separate component lists
        index_list.append(index)
        word_list.append(word)

    # Convert to mini-batch tensors
    index_tensor = torch.tensor(index_list, dtype=torch.int64)

    return (word_list, index_tensor)

def load_data(prefix, batch_size, model="enzh"):
    """
    return train & validation loader for training
    ow return test loader
    """
    if model == "enzh":
        train_data = EnZhTrainData(data.training_data)
        valid_data = EnZhTrainData(data.validation_data)
        test_data  = EnZhTrainData(data.testing_data)

        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    collate_fn=enzh_collate_fn)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    collate_fn=enzh_collate_fn)
        test_loader  = torch.utils.data.DataLoader(dataset=test_data,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    collate_fn=enzh_collate_fn)
        if prefix == "Train":
            return train_loader, valid_loader
        else:
            return test_loader
    else:
        training_data = ZhZhTrainData(data.vocab_zh)
        train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                collate_fn=zhzh_collate_fn)
        return train_loader
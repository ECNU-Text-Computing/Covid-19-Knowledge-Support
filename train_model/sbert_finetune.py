import copy
import csv
import random
import sys

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader


class SbertFinetune(object):
    def __init__(self, args):
        self.dataset = './Dataset/'
        self.cord_root = self.dataset + '2022-01-03/'
        self.metadata_path = self.cord_root + 'metadata.csv'
        self.args = args
        self.sbert_ori_model_root = args.sbert_model_dir_root + args.sbert_model_name
        self.sbert_ft_model_root = args.sbert_model_dir_root + args.sbert_ft_model_name
        self.epoch = args.epoch
        self.loss = args.loss

    def load_data(self):
        print("load data from metadata...")
        title_list = []
        abstract_list = []
        cord_uid_set = set()
        with open(self.metadata_path, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                cord_uid = row['cord_uid'].strip()
                title = row['title'].strip()
                abstract = row['abstract'].strip()
                if title.strip() and abstract.strip() and cord_uid not in cord_uid_set:
                    title_list.append(title)
                    abstract_list.append(abstract)
                    cord_uid_set.add(cord_uid)
        print("title length:", len(title_list))
        print("abstract length:", len(abstract_list))
        print()
        print(str(sys.getsizeof(abstract_list) / 1024 / 1024) + ' MB')
        return title_list, abstract_list

    '''
    get the negtive examples and positive examples
    '''

    def get_inputexample(self, title_list, abstract_list):
        title_list_pos = copy.deepcopy(title_list)
        random.shuffle(title_list)
        title_list_neg = title_list
        del title_list

        pos_tuples = list(zip(title_list_pos, abstract_list))
        neg_tuples = list(zip(title_list_neg, abstract_list))

        train_examples = []
        for pos_tuple in pos_tuples:
            train_examples.append(InputExample(texts=list(pos_tuple), label=1))
        for neg_tuple in neg_tuples:
            train_examples.append(InputExample(texts=list(neg_tuple), label=0))

        del pos_tuples
        del neg_tuples
        random.shuffle(train_examples)
        return train_examples

    def train(self):
        title_list, abstract_list = self.load_data()
        train_examples = self.get_inputexample(title_list, abstract_list)

        print("original model root:", self.sbert_ori_model_root)
        model = SentenceTransformer(self.sbert_ori_model_root)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        if self.loss == "ContrastiveLoss":
            train_loss = losses.ContrastiveLoss(model=model)
        elif self.loss == "OnlineContrastiveLoss":
            train_loss = losses.OnlineContrastiveLoss(model=model)

        print("save finetune model to {}".format(self.sbert_ft_model_root))
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.epoch, warmup_steps=100,
                  output_path=self.sbert_ft_model_root, show_progress_bar=True, steps_per_epoch=10)


if __name__ == "__main__":
    SbertFinetune = SbertFinetune()
    SbertFinetune.train()

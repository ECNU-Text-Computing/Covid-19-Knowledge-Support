### Dowload Dataset

Download the dataset from <a>https://allenai.org/data/cord-19 </a>

* Download the dataset to the root ```./Dataset/``` which is used in the attribute of ```DataProcessor``` in 
```./data_processor/data_processor.py``` and ```SbertFinetune``` in ```./train_model/sbert_finetune.py```


### Prepocess Data
Processing dataset to the json and txt files that are easy to process in the following steps.
```bash
cd data_processor
python data_processor.py
```

### Train models & Calculate indicators
1. Training the TFIDF, LDA and fine-tuned sentence-BERT models.
```bash
python main.py --phase lda --train_model
python main.py --phase tfidf --train_model
python main.py --phase sbert --train_model --finetune --epoch 1 --loss ContrastiveLoss --sbert_model_name all-mpnet-base-v2
```


2. Calculating the indicators of reference support and diversity.
```bash
python main.py --phase lda --train_model --similarity
python main.py --phase tfidf --train_model --diversity
python main.py --phase sbert --finetune --epoch 1 --loss ContrastiveLoss --sbert_model_name all-mpnet-base-v2 --similarity
python main.py --phase sbert --finetune --epoch 1 --loss ContrastiveLoss --sbert_model_name all-mpnet-base-v2 --diversity
```

### Others
#### Word frequency count
```bash
python ./data_processor/word_frequency.py --count_paper_num
python ./data_processor/word_frequency.py --get_tf # get the word tf in the year and month dimension
python ./data_processor/word_frequency.py --analysis # select the words by 3 different strategies
python ./data_processor/word_frequency.py --viz # visualize the result of word frequency
```
#### Other visualizaiton

```bash
cd draw_pic
python draw_pic.py
```

import argparse
import datetime
import json
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from train_model.data_loader import DataLoader
from train_model.model_training import ModelTraining
from calculate_similarity.calculate_similarity import get_similarity
from calculate_similarity.calculate_diversity import get_diversity
from train_model.sbert_finetune import SbertFinetune

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='test', help='the function name.')
    parser.add_argument('--sbert_model_name', default='all-MiniLM-L6-v2', help='the model name of sbert')

    parser.add_argument('--finetune', action="store_true", default=False, help="finetune sbert.")
    parser.add_argument('--similarity', action="store_true", default=False, help="calculate similarity")
    parser.add_argument('--train_model', action="store_true", default=False, help="train model")
    parser.add_argument('--diversity', action="store_true", default=False, help="calculate diversity")
    parser.add_argument('--epoch', default=3, type=int, help="finetune epoch")
    parser.add_argument('--loss', type=str, default="ContrastiveLoss",
                        choices=["ContrastiveLoss", "OnlineContrastiveLoss"])
    args = parser.parse_args()
    args.sbert_model_dir_root = "./tmp_file/sbert_model/"
    if args.finetune:
        args.sbert_ft_model_name = "_".join([args.sbert_model_name, str(args.epoch), args.loss, "finetune"])

    analysis_function = args.phase.strip()

    if args.phase == "sbert" and args.finetune and args.train_model:
        SbertFinetune = SbertFinetune(args)
        SbertFinetune.train()

    if analysis_function in ("lda", "w2v", "tfidf", "sbert"):
        config_path = './config/{}_config.json'.format(analysis_function)
        with open(config_path, "r", encoding="utf-8") as fr:
            config = json.load(fr)
        if analysis_function != "sbert":
            if args.train_model:
                DataLoader = DataLoader(model=analysis_function)
                DataLoader.prepare_data()
                ModelTraining = ModelTraining(model=analysis_function)
                ModelTraining.get_model(**config)
        if args.similarity:
            calculate_similarity = get_similarity(args, model=analysis_function)
            calculate_similarity.get_similarity(**config)
        if args.diversity:
            calculate_similarity = get_diversity(args, model=analysis_function)
            calculate_similarity.get_diversity(**config)
    else:
        raise RuntimeError("There is no {} analysis function.".format(analysis_function))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))
    print('Done main!')

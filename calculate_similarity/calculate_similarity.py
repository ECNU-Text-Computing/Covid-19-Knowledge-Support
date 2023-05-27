import sys

sys.path.append('../')
from train_model.data_loader import DataLoader
from gensim.models import ldamodel
import json
import numpy as np
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer


class get_similarity(object):
    def __init__(self, args, model):
        print("init...")
        self.args = args
        self.model = model
        self.DataLoader = DataLoader(self.model)
        self.dataset_root = './tmp_file/'
        self.model_root = self.dataset_root + "{}_model/".format(self.model)
        self.results_root = './Results/'
        self.batch_size = 32
        if self.model == "sbert":
            self.sbert_model_name = args.sbert_model_name
            self.sbert_model_root = args.sbert_model_dir_root + args.sbert_model_name
        if self.model and args.finetune:
            self.sbert_model_name = args.sbert_ft_model_name
            self.sbert_model_root = args.sbert_model_dir_root + args.sbert_ft_model_name

    def load_model(self, kwargs):
        if self.model == "tfidf":
            model = models.TfidfModel.load(self.model_root + "TFIDF_model.tfidf")
        if self.model == "lda":
            model = ldamodel.LdaModel.load(
                self.model_root + 'LDA_model_{}_dimension.model'.format(kwargs["n_components"]))
        if self.model == "w2v":
            model = Word2Vec.load(self.model_root + 'w2v_model_{}_epochs.model'.format(kwargs["epochs"]))
        if self.model == "sbert":
            print("load model {}".format(self.sbert_model_root))
            model = SentenceTransformer(self.sbert_model_root, device="cuda")
        return model

    def get_ref_dic(self):
        print("load ref dic...")
        with open(self.dataset_root + "cord_uid_ref.json", "r", encoding="utf-8") as f:
            ref_dic = json.load(f)
        print("done.")
        return ref_dic

    def get_dictionary(self):
        dictionary = corpora.Dictionary.load(self.model_root + "dictionary.dic")
        return dictionary

    def get_vector(self, word_list, w2v_model, index2vec):
        vector_list = []
        for word in word_list:
            if word in index2vec:
                vector_list.append(w2v_model.wv[word])
        if vector_list:
            vector_sentence = np.mean(np.array(vector_list), axis=0)
            return vector_sentence
        else:
            return []

    def get_tfidf_median_simi(self, dictionary, texts_ref, texts_info, TFIDF, dic_length):
        bow_ref = [i for i in [dictionary.doc2bow(text) for text in texts_ref] if i != []]
        if bow_ref:
            tfidf_vector_ref = TFIDF[bow_ref]
            index = similarities.MatrixSimilarity(tfidf_vector_ref, num_features=dic_length)
            bow_info = dictionary.doc2bow(texts_info)
            if bow_info:
                tfidf_vector_info = TFIDF[bow_info]
                sims = list(index[tfidf_vector_info])
                median_simi = np.median(sims).item()
                return median_simi

    def get_lda_median_simi(self, dictionary, texts_ref, texts_info, LDA, n_components):
        bow_ref = [i for i in [dictionary.doc2bow(text) for text in texts_ref] if i != []]
        if bow_ref:
            LDA_vector_ref = LDA[bow_ref]
            index = similarities.MatrixSimilarity(LDA_vector_ref, num_features=n_components)
            bow_info = dictionary.doc2bow(texts_info)
            if bow_info:
                LDA_vector_info = LDA[bow_info]
                sims = list(index[LDA_vector_info])
                median_simi = np.median(sims).item()
                return median_simi

    def get_w2v_median_simi(self, texts_ref, texts_info, w2v_model, index2vec):
        w2v_vector_info = self.get_vector(texts_info, w2v_model, index2vec)
        if w2v_vector_info != []:
            w2v_vector_refs = []
            for text_ref in texts_ref:
                w2v_vector_ref = self.get_vector(text_ref, w2v_model, index2vec)
                if w2v_vector_ref != []:
                    w2v_vector_refs.append(w2v_vector_ref)
            if w2v_vector_refs:
                simi_all = np.dot(np.array(w2v_vector_refs), np.array(w2v_vector_info)) / np.linalg.norm(
                    np.array(w2v_vector_info)) / np.linalg.norm(np.array(w2v_vector_refs), axis=1)
                median_simi = np.median(simi_all).item()
                return median_simi

    def get_sbert_median_simi(self, contents_ref, content_info, sbert_model):

        content_info_ref = [content_info] + contents_ref
        vector_info_ref = []

        for i in range(0, len(content_info_ref), self.batch_size):
            start = i
            end = i + self.batch_size
            content_list_batch = content_info_ref[start: end]
            vector_list_batch = list(sbert_model.encode(content_list_batch, batch_size=self.batch_size, device="cuda"))
            vector_info_ref += vector_list_batch

        if len(vector_info_ref) > 1:
            vector_info = np.array(vector_info_ref[0])
            vector_refs = np.array(vector_info_ref[1:])
            simi_all = np.dot(np.array(vector_refs), np.array(vector_info)) / np.linalg.norm(
                np.array(vector_info)) / np.linalg.norm(np.array(vector_refs), axis=1)
            median_simi = np.median(simi_all).item()

            return median_simi

    def get_similarity(self, **kwargs):
        ref_dic = self.get_ref_dic()
        if self.model == "tfidf":
            TFIDF = self.load_model(kwargs)
            dictionary = self.get_dictionary()
            dic_length = len(dictionary)
        if self.model == "lda":
            LDA = self.load_model(kwargs)
            dictionary = self.get_dictionary()
            n_components = kwargs["n_components"]
        if self.model == "w2v":
            w2v_model = self.load_model(kwargs)
            index2vec = w2v_model.wv.index_to_key
        if self.model == "sbert":
            sbert_model = self.load_model(kwargs)
            sbert_model.max_seq_length = 256
            print("Max Sequence Length:", sbert_model.max_seq_length)

        print("已加载模型，即将获取向量...")
        sims_dict = {}
        with open(self.dataset_root + "cord_uid_info.txt", "r", encoding="utf-8") as fp:
            num = 0
            while True:
                num += 1

                line = fp.readline().strip()
                if not line:
                    break
                info = json.loads(line)
                pub_time = info["pub_time"]
                if pub_time:
                    if int(pub_time.split("-")[0]) >= 2000:
                        cord_uid = info["cord_uid"]
                        if cord_uid in ref_dic:
                            content_info = info["title"] + " " + info["abstract"]
                            refs = ref_dic[cord_uid]
                            if self.model != "sbert":
                                texts_info = self.DataLoader.clean_list(content_info.split(" "))
                                if texts_info:
                                    texts_ref = []
                                    for ref in refs:
                                        text_ref = self.DataLoader.clean_list(ref['title'].split(" "))
                                        if text_ref:
                                            texts_ref.append(text_ref)
                                    if texts_ref:
                                        if self.model == "tfidf":
                                            median_simi = self.get_tfidf_median_simi(dictionary, texts_ref, texts_info,
                                                                                     TFIDF, dic_length)
                                        if self.model == "lda":
                                            median_simi = self.get_lda_median_simi(dictionary, texts_ref, texts_info,
                                                                                   LDA,
                                                                                   n_components)
                                        if self.model == "w2v":
                                            median_simi = self.get_w2v_median_simi(texts_ref, texts_info, w2v_model,
                                                                                   index2vec)
                                        if median_simi != None:
                                            sims_dict[cord_uid] = median_simi
                            else:
                                contents_ref = []
                                for ref in refs:
                                    content_ref = ref['title']
                                    if content_ref.strip():
                                        contents_ref.append(content_ref)
                                if len(contents_ref) > 0:
                                    median_simi = self.get_sbert_median_simi(contents_ref, content_info, sbert_model)
                                if median_simi != None:
                                    sims_dict[cord_uid] = median_simi

        print("共生成{}个相似度值".format(len(sims_dict)))
        if self.model == "lda":
            file_name = self.model + "_" + str(kwargs["n_components"])
        if self.model == "w2v":
            file_name = self.model + "_" + str(kwargs["epochs"])
        if self.model == "tfidf":
            file_name = self.model
        if self.model == "sbert":
            file_name = self.model + "_" + self.sbert_model_name
        with open(self.results_root + "cord_uid_{}_similarity_new_version_test.json".format(file_name), "w",
                  encoding="utf-8") as fw:
            json.dump(sims_dict, fw, ensure_ascii=False)
        print(self.results_root + "cord_uid_{}_similarity_new_version_test.json".format(file_name))

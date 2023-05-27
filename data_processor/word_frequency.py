import csv
import json
import re
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

params = {'font.family': 'serif',
              'font.serif': 'Times New Roman',
              'font.style': 'normal',
              'font.weight': 'normal',  # or 'blod'
              'font.size': 10,  # or large,small'medium'  # 28 in relation
              }
plt.rcParams.update(params)


class WordFrequency(object):
    def __init__(self, args=None):
        if args:
            self.cal_method = args.cal_method
            self.before_2000 = args.before_2000
        self.results_root = "./Results/"
        self.jif_match_root = self.tmp_root + "jif_match/"
        self.tmp_root = './tmp_file/'
        if not self.before_2000:
            self.ref_dic_root = self.tmp_root + "cord_uid_ref.json" ####
            self.info_root = self.tmp_root + "cord_uid_info.txt"
        else:
            self.ref_dic_root = self.tmp_root + "cord_uid_ref_before_2000.json"
            self.info_root = self.tmp_root + "cord_uid_info_before_2000.txt"
        print(self.ref_dic_root)
        print(self.info_root)
        self.stopwords_path = "./Dataset/stop_words/English_stopwords.txt"
        self.stop_words = self._get_stop_words()
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def _clean_word(self, word, pos):
        # ret = re.sub("\W*", "", input.lower())
        word = "".join(re.findall(r"[a-zA-Z0-9|-]", word))
        if word:
            word = self._lemmatize_pos(word.lower(), pos)
            return word

    def _get_stop_words(self):
        with open(self.stopwords_path, "r", encoding="utf-8") as f:
            stop_words = set([i.strip() for i in f.readlines()])
            return stop_words

    def _get_dic(self, root):
        with open(root, "r", encoding="utf-8") as fp:
            dic = json.load(fp)
        return dic

    def _get_id_pubtime_dic(self):  # 2019,2020,2021
        # get the year of ref
        id_pubtime_dic = {}
        num = 0
        with open(self.info_root, "r", encoding="utf-8") as fp:
            while True:
                num += 1
                line = fp.readline().strip()
                if not line:
                    break
                info = json.loads(line)
                pub_time = info["pub_time"]
                if pub_time:
                    pub_time = "-".join(pub_time.split("-")[:2])
                    id_pubtime_dic[info["cord_uid"]] = pub_time
        return id_pubtime_dic

    def _lemmatize_pos(self, word, pos):
        if pos.startswith('NN'):
            return self.wordnet_lemmatizer.lemmatize(word, pos='n')
        elif pos.startswith('VB'):
            return self.wordnet_lemmatizer.lemmatize(word, pos='v')
        elif pos.startswith('JJ'):
            return self.wordnet_lemmatizer.lemmatize(word, pos='a')
        elif pos.startswith('R'):
            return self.wordnet_lemmatizer.lemmatize(word, pos='r')
        else:
            return word

    def _word_frequency_count(self):
        year_word_tf = {}
        month_word_tf = {}
        ref_dic = self._get_dic(self.ref_dic_root)
        id_pubtime_dic = self._get_id_pubtime_dic()
        for id ,refs in ref_dic.items():
            if id in id_pubtime_dic:
                pub_time = id_pubtime_dic[id]
                year = int(pub_time.split("-")[0])
                for ref in refs:
                    ref_content = ref["title"].strip().split()
                    word_pos = nltk.pos_tag(ref_content)
                    for (word,pos) in word_pos:
                        word = self._clean_word(word, pos)
                        if word and word not in self.stop_words:
                            if year not in year_word_tf:
                                year_word_tf[year] = {}
                            year_word_tf[year][word] = year_word_tf[year].get(word, 0) + 1
                            # if year in 2019, 2020, 2021, save pubtime
                            if year in {2019, 2020, 2021}:
                                if pub_time not in month_word_tf:
                                    month_word_tf[pub_time] = {}
                                month_word_tf[pub_time][word] = month_word_tf[pub_time].get(word, 0) + 1
        return year_word_tf, month_word_tf

    def count_paper_num(self):  # with ref
        year_paper_count = {}
        month_paper_count = {}
        ref_dic = self._get_dic(self.ref_dic_root)
        id_pubtime_dic = self._get_id_pubtime_dic()
        for id, refs in ref_dic.items():
            if id in id_pubtime_dic:
                pub_time = id_pubtime_dic[id]
                year = int(pub_time.split("-")[0])
                year_paper_count[year] = year_paper_count.get(year, 0) + 1
                if year in {2019, 2020, 2021}:
                    month_paper_count[pub_time] = month_paper_count.get(pub_time, 0) + 1
        with open(self.tmp_root + "month_paper_count.json", "w", encoding="utf-8") as fw:
            json.dump(month_paper_count, fw, ensure_ascii=False)
        with open(self.tmp_root + "year_paper_count.json", "w", encoding="utf-8") as fw:
            json.dump(year_paper_count, fw, ensure_ascii=False)

    def _get_word_year_p(self, select_word_list, pubtime_word_tf, pubtime_tf_all, attribute=None, attribute_name=None):
        if not attribute:
            word_year_p = {"word":[],"rank":[]}
            rank = 1
            for word in select_word_list:
                word_year_p["word"].append(word)
                word_year_p["rank"].append(rank)
                for pubtime, word_tf_dic in pubtime_word_tf.items():
                    p = word_tf_dic.get(word,0)/pubtime_tf_all[pubtime]
                    if pubtime not in word_year_p:
                        word_year_p[pubtime] = [p]
                    else:
                        word_year_p[pubtime].append(p)
                rank += 1
            return word_year_p

        else:
            word_year_p = {"word": [],"rank": [],attribute_name:[]}
            rank = 1
            for i in range(len(select_word_list)):
                word = select_word_list[i]
                word_year_p["word"].append(word)
                word_year_p["rank"].append(rank)
                if isinstance(attribute, dict):
                    word_year_p[attribute_name].append(attribute[word])
                elif isinstance(attribute, list):
                    word_year_p[attribute_name].append(attribute[i])
                for pubtime, word_tf_dic in pubtime_word_tf.items():
                    p = word_tf_dic.get(word, 0) / pubtime_tf_all[pubtime]
                    if pubtime not in word_year_p:
                        word_year_p[pubtime] = [p]
                    else:
                        word_year_p[pubtime].append(p)
                rank += 1
            return word_year_p

    def word_tf_main(self):
        year_word_tf, month_word_tf = self._word_frequency_count()
        with open(self.tmp_root + "word_frequency_count_year" + self.before_2000*"_before_2000" + ".json", "w", encoding="utf-8") as fw:
            json.dump(year_word_tf, fw, ensure_ascii=False)
        with open(self.tmp_root + "word_frequency_count_month" + self.before_2000*"_before_2000" + ".json", "w", encoding="utf-8") as fw:
            json.dump(month_word_tf, fw, ensure_ascii=False)

    def word_tf_process_main(self):
        cal_method = self.cal_method
        # the analysis method: x-axis word, categories year(2019, 2020, 2021), y-axis tf
        if cal_method == "3year_word":
            self._analysis_3year_word()

    def _analysis_3year_word(self):
        year_word_tf = self._get_dic(self.tmp_root + "word_frequency_count_year.json")  # get the year dic
        year_word_tf = dict(filter(lambda x: int(x[0]) in {2019, 2020, 2021}, year_word_tf.items()))

        # total word frequency of every year
        year_tf_all = {}
        for year, word_tf_dic in year_word_tf.items():
            tf_all = sum(list(word_tf_dic.values()))
            year_tf_all[year] = tf_all
        print(year_tf_all)

        # select words by avg_rank
        num = 0
        for year, word_tf_dic in year_word_tf.items():
            word_set = set(word_tf_dic.keys())
            if num == 0:
                word_intersection = word_set  # init
            else:
                word_intersection = word_intersection.intersection(word_set)  # the word appeared in three years
            num += 1

        year_word_rank = {}
        for year, word_tf_dic in year_word_tf.items():
            sorted_word_tf_dic = sorted(word_tf_dic.items(), reverse=True, key=lambda x: x[1])
            ranks = list(np.array(list(range(len(sorted_word_tf_dic)))) + 1)
            word_rank_dic = dict(zip([i[0] for i in sorted_word_tf_dic], ranks))
            year_word_rank[year] = word_rank_dic

        word_avg_rank = {}
        for word in list(word_intersection):
            rank_list = []
            for year, word_rank_dic in year_word_rank.items():
                rank = word_rank_dic[word]
                rank_list.append(rank)
            avg_rank = np.mean(rank_list)
            word_avg_rank[word] = avg_rank

        sorted_word_avg_rank = sorted(word_avg_rank.items(), reverse=False, key=lambda x:x[1])
        select_words_avg_rk = [i[0] for i in sorted_word_avg_rank][:100]
        word_year_p = self._get_word_year_p(select_words_avg_rk, year_word_tf, year_tf_all, attribute=word_avg_rank, attribute_name="avg_rk_3year")
        pd.DataFrame(word_year_p).to_excel(self.tmp_root + "word_year_p_avg_rk.xlsx")  # The word with the highest
        # average ranking in three years

        # get top num of word
        num = 0
        for year, word_tf_dic in year_word_tf.items():
            word_set = set(word_tf_dic.keys())
            if num == 0:
                word_union = word_set  # init
            else:
                word_union = word_intersection | word_set
            num += 1

        word_tf_all = {}
        for word in list(word_union):
            word_tf_list = []
            for year, word_tf_dic in year_word_tf.items():
                if word in word_tf_dic:
                    word_tf_list.append(word_tf_dic[word])
            tf_sum = sum(word_tf_list)
            word_tf_all[word] = tf_sum
        sorted_word_tf_all = sorted(word_tf_all.items(), reverse=True, key=lambda x: x[1])[:100]
        selected_word_tf_all = [i[0] for i in sorted_word_tf_all]
        word_year_p = self._get_word_year_p(selected_word_tf_all, year_word_tf, year_tf_all, attribute=word_tf_all, attribute_name="total_tf_3year")
        pd.DataFrame(word_year_p).to_excel(self.tmp_root + "word_year_p_tf_all.xlsx")
        # select the words with highest tf


        # get the word frequency of every year
        word_yearsource_dic = {}
        word_list = []
        for year, word_tf_dic in year_word_tf.items():
            sorted_word_tf_dic = sorted(word_tf_dic.items(), key=lambda x:x[1], reverse=True)[:100]
            sorted_words = [i[0] for i in sorted_word_tf_dic]
            word_list.extend(sorted_words)
            for word in sorted_words:
                if word not in word_yearsource_dic:
                    word_yearsource_dic[word] = [":".join([str(year), str(word_tf_dic[word])])]
                else:
                    word_yearsource_dic[word].append(":".join([str(year), str(word_tf_dic[word])]))

        word_yearsource_dic = {word: ";".join(year_info) for (word, year_info) in word_yearsource_dic.items()}
        selected_word_list = list(set(word_list))
        word_year_p = self._get_word_year_p(selected_word_list, year_word_tf, year_tf_all, attribute=word_yearsource_dic, attribute_name="year_source")
        pd.DataFrame(word_year_p).to_excel(self.tmp_root + "word_year_p_year_tf.xlsx")
        # The word with the highest ranking in each year

        word_list = []
        year_list = []
        for year, word_tf_dic in year_word_tf.items():
            sorted_word_tf_dic = sorted(word_tf_dic.items(), key=lambda x: x[1], reverse=True)[:100]
            sorted_words = [i[0] for i in sorted_word_tf_dic]
            word_list.extend(sorted_words)
            year_list.extend([year]*len(sorted_words))

        word_year_p = self._get_word_year_p(word_list, year_word_tf, year_tf_all,
                                            attribute=year_list, attribute_name="year_source")
        pd.DataFrame(word_year_p).to_excel(self.tmp_root + "word_year_p_year_tf_noconcat.xlsx")


    def visualization_main(self, word_3year=False, curious_word=False, uniq_word=True):
        if word_3year:
            self._visualizaiton_3year_word()
        if curious_word:
            self._visualization_word_trend()
        if uniq_word:
            self._unique_word()

    def _visualizaiton_3year_word(self, tf_all=False, year_tf=False, year_tf_subplots=True):
        # 1 using the top 100 of tf to visualize
        ban_word_list = ["2019"]
        if tf_all:
            fig, ax = plt.subplots(figsize=(13, 7))
            df = pd.read_excel(self.tmp_root + "/word_frequency/word_year_p_tf_all.xlsx")
            df = self.ban_word(df, ban_word_list=ban_word_list)
            df = dict(df.sort_values(by="rank", ascending=True)[:30])
            word_list = df["word"]
            year_p_list_dic = {}
            year_p_list_dic["2019"] = df["2019"]
            year_p_list_dic["2020"] = df["2020"]
            year_p_list_dic["2021"] = df["2021"]
            self._draw_multi_bar(ax, word_list, year_p_list_dic)
            plt.savefig(self.tmp_root + "/word_frequency/word_year_p_tf_all.png", dpi=1000)

        # 2 find the words that appear most frequently
        if year_tf:
            df = pd.read_excel(self.tmp_root + "/word_frequency/word_year_p_year_tf_noconcat.xlsx")
            ban_word_list = ["2019"]
            select_word_list = list(df["word"])
            for word in ban_word_list:
                select_word_list.remove(word)
            df = df[df["word"].isin(select_word_list)]
            fig, ax = plt.subplots(figsize=(13, 7))
            years = [2019,2020,2021]
            word_list = []
            year_p_list_dic = {"2019": [],"2020": [], "2021": []}
            for year in years:
                df_new = df[df["year_source"]==year]
                df_new = df_new.sort_values(by="rank", ascending=True)
                df_new = df_new[:50]
                word_list.append(list(df_new["word"]))
            print(word_list)

            num = 0
            for year_word_list in word_list:
                if num == 0:
                    new_word_list = year_word_list
                else:
                    for word in year_word_list:
                        if word not in new_word_list:
                            new_word_list.append(word)
                num += 1
            print(new_word_list)

            for word in new_word_list:
                word_info = df[df["word"]==word]
                year_p_list_dic["2019"].append(list(word_info["2019"])[0])
                year_p_list_dic["2020"].append(list(word_info["2020"])[0])
                year_p_list_dic["2021"].append(list(word_info["2021"])[0])

            self._draw_multi_bar(ax, new_word_list, year_p_list_dic)
            plt.savefig(self.tmp_root+"/word_frequency/word_year_p_year_concat.png", dpi=1000)

        if year_tf_subplots:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
            plt.subplots_adjust(bottom=0.32, wspace=0.17,left=0.07, right=0.97)
            df = pd.read_excel(self.tmp_root + "/word_frequency/word_year_p_year_tf_noconcat.xlsx")
            df = self.ban_word(df, ban_word_list)
            years = [2019, 2020, 2021]
            colors = ["#A5C8E1", "#F5D8A3","#D7E9D4" ]#D7E9D4#F5D8A3#A5C8E1#A3CC9C#EFBE67#78ABD2["#1F77B4", "#e69e19", "#9BC995"]
            edge_colors = ["#78ABD2","#EFBE67","#A3CC9C"]
            num = 0
            for year in years:
                ax = axes[num]
                color = colors[num]
                ec = edge_colors[num]
                df_new = df[df["year_source"] == year]
                df_new = df_new.sort_values(by="rank", ascending=True)
                df_new = df_new[:17]
                word_list = list(df_new["word"])
                value_list = list(np.array(df_new[str(year)])*100)
                self._draw_bar(ax, word_list, value_list, color, ec, title=year)
                num += 1
            plt.savefig(self.tmp_root + "word_frequency/word_year_p_tf_subplots.pdf", dpi=600)
            plt.show()

    def _draw_bar(self, ax, word_list, value_list, color, ec, title):
        bwith = 1
        # ax.grid(zorder=0, linestyle="dotted", color="slategray", alpha=0.6, linewidth=0.8, axis="y")
        # ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.tick_params(width=bwith)
        ax.bar(word_list, value_list, alpha=1, edgecolor=ec, color=color)
        # ax.legend(frameon=False)
        ax.set_xticks(ax.get_xticks(), [word.capitalize() for word in word_list], rotation=90)
        ax.set_title(title, fontweight="semibold")
        ax.text(-0.12,1.05,"times/100words",transform=ax.transAxes)


    def _draw_multi_bar(self, ax, word_list, year_p_list_dic):
        self._clean_ax(ax)
        plt.subplots_adjust(bottom=0.2)
        for year,p_list in year_p_list_dic.items():
            ax.bar(word_list, p_list, alpha=0.3, edgecolor="black", label=year)
        ax.legend(frameon=False)
        ax.set_xticks(ax.get_xticks(),word_list, rotation=90)
        pass

    def _clean_ax(self, ax):
        bwith = 1
        # ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(width=bwith)

    def ban_word(self, df, ban_word_list):
        select_word_list = list(set(df["word"]))
        for word in ban_word_list:
            select_word_list.remove(word)
        df = df[df["word"].isin(select_word_list)]
        return df

    def _tf_per_paper(self, time_word_tf_dic, time="month"):
        # count paper num from the view of year and month
        count_dic = self._get_dic(self.tmp_root + "word_frequency/{}_paper_count.json".format(time))
        time_word_p_dic = {}
        for time, word_tf_dic in time_word_tf_dic.items():
            word_tf_item = word_tf_dic.items()
            word_list = [i[0] for i in word_tf_item]
            tf_list = [i[1] for i in word_tf_item]
            p_list = np.array(tf_list)/count_dic[time]
            time_word_p_dic[time] = dict(zip(word_list, p_list))
        return time_word_p_dic

    def _get_word_tf_dic(self):
        with open(self.tmp_root + "word_frequency/word_frequency_count_month.json", "r", encoding="utf-8") as fp:
            month_word_tf_dic = json.load(fp)
            month_word_tf_dic = dict(filter(lambda x: x[0] not in {'2019','2021','2020'}, month_word_tf_dic.items()))
        with open(self.tmp_root + "word_frequency/word_frequency_count_year.json", "r", encoding="utf-8") as fp:
            year_word_tf_dic = json.load(fp)
        return month_word_tf_dic, year_word_tf_dic

    def _visualization_word_trend(self):
        month_word_tf_dic, year_word_tf_dic = self._get_word_tf_dic()
        curious_words_1 = ["covid-19","sars-cov-2","coronavirus",]  # 2020"vaccine"
        curious_words_2 = ["sars","sars-cov","mers","mers-cov",]  # 2021
        curious_words_3 = ["influenza","respiratory","pneumonia"]
        curious_words = curious_words_1 + curious_words_2 + curious_words_3
        # month_word_p_dic = self._tf_per_paper(month_word_tf_dic, time="month")
        # year_word_p_dic = self._tf_per_paper(year_word_tf_dic, time="year")
        month_word_p_dic = self._tf2p(month_word_tf_dic)
        year_word_p_dic = self._tf2p(year_word_tf_dic)
        month_word_time_p, month_time_line = self._get_word_time_p(curious_words, month_word_p_dic, time="month")
        year_word_time_p, year_time_line = self._get_word_time_p(curious_words, year_word_p_dic, time="year")

        # month_plot
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,6))
        plt.subplots_adjust(bottom=0.08, wspace=0.2, left=0.07, hspace=0.45, right=0.95,top=0.92)
        colors = ["gray",'#C9C9C9','#a5b6c5']
        self._draw_plot_axes(curious_words_1, axes, year_word_time_p, year_time_line, month_word_time_p, month_time_line, colors, row=0)
        colors = ["#2878b5","#9ac9db",'#c82423',"#f8ac8c"]
        self._draw_plot_axes(curious_words_2, axes, year_word_time_p, year_time_line, month_word_time_p,
                             month_time_line, colors, row=1)
        colors = ['#496c88','#a5b6c5','#9dc3e7','#a9b8c6' ]
        self._draw_plot_axes(curious_words_3, axes, year_word_time_p, year_time_line, month_word_time_p,
                             month_time_line, colors, row=2)
        plt.savefig(self.tmp_root + "word_frequency/word_time_line.pdf", dpi=1000)
        plt.show()

    def _draw_plot_axes(self,curious_words, axes, year_word_time_p, year_time_line, month_word_time_p, month_time_line, colors, row):
        num = 0
        ax = axes[row][0]
        self._clean_ax(ax)
        self._set_xticks(ax, year_time_line, time="year")
        ax.text(-0.06, 1.05, "times/100words", transform=ax.transAxes)
        if row == 0:
            ax.set_title("2000-2021",fontweight="semibold")
        for word in curious_words:
            color = colors[num]
            year_time_p = year_word_time_p[word]
            self._draw_plot(ax, year_time_p, color, label=word.capitalize(),markersize=2)
            num += 1
        if row == 2:
            ax.set_xlabel("Year")

        num = 0
        ax = axes[row][1]
        self._clean_ax(ax)
        self._set_xticks(ax, month_time_line, time="month")
        ax.text(-0.06, 1.05, "times/100words", transform=ax.transAxes)
        for word in curious_words:
            color = colors[num]
            month_time_p = month_word_time_p[word]
            self._draw_plot(ax, month_time_p, color, label=word.capitalize(),markersize=2)
            num += 1
        if row == 0:
            ax.set_title("2019-2021",fontweight="semibold")
        if row == 2:
            ax.set_xlabel("Month")
        ax.legend(frameon=False,labelspacing=0.2)

    def _draw_plot(self,ax, month_time_p, color=None, label=None,marker='o',markersize=2.5, linewidth=1.3):
        ax.plot(month_time_p, color=color,label=label,linewidth=linewidth,alpha=0.95, marker=marker,markersize=markersize)
        ax.grid(linestyle="dotted", color="slategray", alpha=0.3, linewidth=0.8, axis="both")

    def _set_xticks(self, ax, time_line,time, rotation):
        if time=="month": cut_length = 5
        if time=="year": cut_length = 2
        new_time_line = []
        for i, j in enumerate(time_line):
            if i % 5 == 0:
                new_time_line.append(j[-cut_length:])
            else:
                new_time_line.append("")
        ax.set_xticks(range(len(time_line)), new_time_line, rotation=rotation)

    def _tf2p(self, time_word_tf_dic):
        time_word_p_dic = {}
        for time, word_tf in time_word_tf_dic.items():
            word_tf_items = word_tf.items()
            word_list = [i[0] for i in word_tf_items]
            tf_list = [i[1] for i in word_tf_items]
            p_list = list(np.array(tf_list)/sum(tf_list)*100)
            word_p_dic = dict(zip(word_list, p_list))
            time_word_p_dic[time] = word_p_dic
        return time_word_p_dic

    def _get_right_time_line(self,time_word_p_dic, time="month"):
        if time == "month":
            time_line = sorted(list(set(time_word_p_dic.keys())),
                                key=lambda x: (x.split("-")[0], x.split("-")[1]))
        elif time == "year":
            time_line = sorted(list(time_word_p_dic.keys()), reverse=False)  # from small to big
        return time_line

    def _get_word_time_p(self, curious_word, time_word_p_dic, time="month"):
        time_line = self._get_right_time_line(time_word_p_dic, time)
        print("***"*10, time, "***"*10)
        print(time_line)
        word_time_p = {}
        for word in curious_word:
            time_p_list = []
            for time in time_line:
                word_p_dic = time_word_p_dic[time]
                p = word_p_dic.get(word, 0)
                time_p_list.append(p)
            assert len(time_p_list) == len(time_line)
            word_time_p[word] = time_p_list
        return word_time_p, time_line

    def _unique_word(self):
        month_word_tf_dic, year_word_tf_dic = self._get_word_tf_dic()
        year_time_line = self._get_right_time_line(year_word_tf_dic, time="year")
        month_time_line = self._get_right_time_line(month_word_tf_dic, time="month")
        df_div = self._get_div()
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 4.5))
        plt.subplots_adjust(hspace=0.5, bottom=0.15)
        self._draw_uniq_word(year_time_line, year_word_tf_dic, month_time_line, month_word_tf_dic, axes)
        self._draw_div(axes, month_time_line, year_time_line, df_div)
        plt.savefig(self.tmp_root + "word_frequency/wf_uniq_word.pdf", dpi=1000)
        plt.show()

    def _get_time_div(self, time_line, df, indicator, time="month"):
        if time == "month": ref = "special_date"
        elif time == "year": ref = "year"
        time_divs_dic = {}
        for time in time_line:
            divs = list(1 - np.array(list(df[df[ref] == time][indicator])))
            time_divs_dic[time] = divs

        time_div_median_dic = {}
        for time, divs in time_divs_dic.items():
            div_median = np.nanmedian(divs)
            time_div_median_dic[time] = div_median

        div_median_list = []
        for time in time_line:
            div_median_list.append(time_div_median_dic[time])

        return time_divs_dic, time_div_median_dic, div_median_list

    def _draw_div(self, axes, month_list, year_list, df):
        indicator_name = 'sbert_div'

        time_divs_dic, time_div_median_dic, div_median_list = self._get_time_div(time_line=year_list, df=df,
                                                                                 indicator=indicator_name, time="year")
        col = 0
        ax = axes[1][col]
        self._clean_ax(ax)
        self._set_xticks(ax, time_line=year_list, time="year", rotation=90)
        self._draw_plot(ax, div_median_list,linewidth=2, markersize=4,color='#496c88')
        ax.set_ylabel("Semantic diversity\nin References")
        ax.set_xlabel("Year")

        time_divs_dic, time_div_median_dic, div_median_list = self._get_time_div(time_line=month_list, df=df,
                                                                                 indicator=indicator_name, time="month")
        col = 1
        ax = axes[1][col]
        self._clean_ax(ax)
        self._set_xticks(ax, time_line=month_list, time="month", rotation=90)
        self._draw_plot(ax, div_median_list, linewidth=2, markersize=4,color='#a5b6c5')
        ax.set_xlabel("Month")



    def _draw_uniq_word(self, year_time_line, year_word_tf_dic, month_time_line, month_word_tf_dic, axes):
        # plt.subplots_adjust(left=0.07, right=0.96, bottom=0.2, top=0.95)
        existed_words = self._word_before_2000()
        uniq_r = []
        ax = axes[0][0]
        for year in year_time_line:
            word_tf_dic = year_word_tf_dic[year]
            words = set(word_tf_dic.keys())
            # print(len(words), end="——")
            new_words = words - existed_words
            # print(len(new_words))
            uniq_r.append(len(new_words) / len(words))
            existed_words = existed_words | words
        self._clean_ax(ax)
        self._draw_plot(ax, uniq_r, label=None, linewidth=2, markersize=4, marker='o', color='#496c88')
        self._set_xticks(ax, time_line=year_time_line, time="year", rotation=90)
        ax.set_ylabel("Unique words/total words\nin References")
        ax.set_title("2000-2021",fontweight="semibold")
        # ax.set_xlabel("Year")

        month_uniq_r = []
        existed_words = self._word_before_2000()
        ax = axes[0][1]
        for year in year_time_line:
            if year not in {"2019", "2020", "2021"}:
                word_tf_dic = year_word_tf_dic[year]
                words = set(word_tf_dic.keys())
                existed_words = existed_words | words
        for month in month_time_line:
            word_tf_dic = month_word_tf_dic[month]
            words = set(word_tf_dic.keys())
            new_words = words - existed_words
            month_uniq_r.append(len(new_words) / len(words))
        self._clean_ax(ax)
        self._draw_plot(ax, month_uniq_r, label=None, linewidth=2, markersize=4, marker='o', color='#a5b6c5')
        self._set_xticks(ax, time_line=month_time_line, time="month", rotation=90)
        ax.set_title("2019-2021", fontweight="semibold")
        # ax.set_ylabel("Unique words/total words")
        # ax.set_xlabel("Month")

    def _get_div(self):
        tfidf_div_dic = self._get_dic(self.results_root + 'cord_uid_tfidf_diversity.json')
        sbert_div_dic = self._get_dic(
            self.results_root + 'cord_uid_sbert_all-mpnet-base-v2_1_ContrastiveLoss_finetune_diversity.json')  # pay attention to the version of sbert
        lda_div_dic = self._get_dic(self.results_root + 'cord_uid_lda_100_diversity.json')  # 'cord_uid_lda_200_similarity.json'
        pub_time_dic = self._get_dic(self.results_root + 'pub_time_dic.json')
        cord_uids = list(set(tfidf_div_dic.keys()) & set(sbert_div_dic.keys()) & set(lda_div_dic.keys()) & set(pub_time_dic.keys()))
        df_dic = {"cord_uid":[], "pub_time":[], "year":[], "special_date":[], "tfidf_div":[], "lda_div":[], "sbert_div":[]}
        for cord_uid in cord_uids:
            pub_time = pub_time_dic[cord_uid]
            df_dic["cord_uid"].append(cord_uid)
            df_dic["pub_time"].append(pub_time)
            df_dic["tfidf_div"].append(tfidf_div_dic[cord_uid])
            df_dic["lda_div"].append(lda_div_dic[cord_uid])
            df_dic["sbert_div"].append(sbert_div_dic[cord_uid])
            if len(pub_time) == 10:
                df_dic["year"].append(pub_time.split("-")[0])
                df_dic["special_date"].append(pub_time[:7])
                # month_list.append(t.split("-")[1])
            elif len(pub_time) == 4:
                df_dic["year"].append(pub_time)
                df_dic["special_date"].append(np.nan)
                # month_list.append(np.nan)
            else:
                df_dic["year"].append(np.nan)
                df_dic["special_date"].append(np.nan)
        df_div = pd.DataFrame(df_dic)
        indicator_names = ["tfidf_div", "lda_div", "sbert_div"]
        # sns.lineplot(x="year", y="sbert_div", data=df[df["year"].isin([str(i) for i in list(range(2000,2021))])])
        for indicator_name in indicator_names:
            mean_of_ind = np.nanmean(df_div[indicator_name])
            std_of_ind = np.nanstd(df_div[indicator_name])
            df_div[indicator_name + "_std"] = df_div[indicator_name].apply(lambda x: (x - mean_of_ind) / std_of_ind)
        df_div['ts_std'] = df_div[[i + "_std" for i in indicator_names]].mean(axis=1)
        return df_div

    def _word_before_2000(self):
        wf_before_2000 = self._get_dic(self.tmp_root + "word_frequency/word_frequency_count_year_before_2000.json")
        existed_words = set()
        for word_value_dic in list(wf_before_2000.values()):
            existed_words = existed_words | set(word_value_dic.keys())
        return existed_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_tf", action="store_true", default=False)
    parser.add_argument("--count_paper_num", action="store_true", default=False)
    parser.add_argument("--analysis", action="store_true", default=False)
    parser.add_argument("--cal_method", choices=["3year_word"], default="3year_word")
    parser.add_argument("--viz", action="store_true", default=False)
    parser.add_argument("--before_2000", action="store_true", default=False)
    args = parser.parse_args()
    wf = WordFrequency(args)
    if args.get_tf:
        wf.word_tf_main()
    if args.analysis:
        wf.word_tf_process_main()
    if args.viz:
        wf.visualization_main()
    if args.count_paper_num:
        wf.count_paper_num()

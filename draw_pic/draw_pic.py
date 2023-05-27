import json
import math
import os
from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


class draw_pic(object):
    def __init__(self):
        self.tmp_root = '../tmp_file/'
        self.dataset = '../Results/'
        self.results_root = '../Results/Imgs/'
        self.author_num_dic = self.read_dic('author_num_dic.json')
        self.if_dic = self.read_dic('jif_dic.json')
        self.pub_time_dic = self.read_dic('pub_time_dic.json')
        self.ref_age_median_dic = self.read_dic('ref_age_dic.json')
        self.ref_num_dic = self.read_dic('ref_num_dic.json')
        self.tfidf_median_dic = self.read_dic('cord_uid_tfidf_similarity.json')
        self.lda_median_dic = self.read_dic('cord_uid_lda_100_similarity.json')
        self.w2v_median_dic = self.read_dic('cord_uid_w2v_20_similarity.json')
        self.sbert_median_dic = self.read_dic(
            'cord_uid_sbert_all-mpnet-base-v2_1_ContrastiveLoss_finetune_similarity.json')
        self.tfidf_div_dic = self.read_dic('cord_uid_tfidf_diversity.json')
        self.sbert_div_dic = self.read_dic(
            'cord_uid_sbert_all-mpnet-base-v2_1_ContrastiveLoss_finetune_diversity.json')
        self.lda_div_dic = self.read_dic('cord_uid_lda_100_diversity.json')
        self.df = self.construct_dataframe()
        self.indicator_exchange = self.indicator_exchange()
        self.restrict_dic = self.get_restrict_dic()

    def read_dic(self, json_name):
        with open(self.dataset + json_name, "r", encoding="utf-8") as fp:
            dic = json.load(fp)
        return dic

    def get_percentile(self, dictionary):
        dictionary_value = []
        for i in list(dictionary.values()):
            if i == "Not Available":
                continue
            if i == i:
                dictionary_value.append(i)
        return np.percentile(dictionary_value, 95)

    def get_restrict_dic(self):
        restrict_dic = {}
        for indicator in ["author_num", "ref_age_median", "ref_num", "lda_median", "if", "tfidf_median", "w2v_median",
                          "sbert_median"]:
            dict_name = indicator + "_dic"
            dictionary = dict(eval("self." + dict_name))
            restrict_dic[indicator] = self.get_percentile(dictionary)
        return restrict_dic

    def construct_dataframe(self):
        cord_uid_list = list(self.pub_time_dic.keys())
        pub_time_list = [self.pub_time_dic.get(i, np.nan) for i in cord_uid_list]
        author_num_list = [self.author_num_dic.get(i, np.nan) for i in cord_uid_list]
        if_list = [np.nan if i == "Not Available" else i for i in [self.if_dic.get(i, np.nan) for i in cord_uid_list]]
        ref_age_median_list = [self.ref_age_median_dic.get(i, np.nan) for i in cord_uid_list]
        ref_num_list = [self.ref_num_dic.get(i, np.nan) for i in cord_uid_list]
        lda_median_list = [self.lda_median_dic.get(i, np.nan) for i in cord_uid_list]
        tfidf_median_list = [self.tfidf_median_dic.get(i, np.nan) for i in cord_uid_list]
        w2v_median_list = [self.w2v_median_dic.get(i, np.nan) for i in cord_uid_list]
        sbert_median_list = [self.sbert_median_dic.get(i, np.nan) for i in cord_uid_list]
        lda_div_list = [self.lda_div_dic.get(i, np.nan) for i in cord_uid_list]
        tfidf_div_list = [self.tfidf_div_dic.get(i, np.nan) for i in cord_uid_list]
        sbert_div_list = [self.sbert_div_dic.get(i, np.nan) for i in cord_uid_list]

        year_list = []
        month_list = []
        for t in pub_time_list:
            if len(t) == 10:
                year_list.append(t.split("-")[0])
                month_list.append(t.split("-")[1])
            elif len(t) == 4:
                year_list.append(t)
                month_list.append(np.nan)
            else:
                year_list.append(np.nan)
                month_list.append(np.nan)

        special_date_list = []
        for t in pub_time_list:
            if len(t) == 10:
                special_date_list.append(t[:7])
            else:
                special_date_list.append(np.nan)

        if_class_list = []
        for i in if_list:
            if i == "Not Available":
                if_class_list.append(np.nan)
            elif not i == i:
                if_class_list.append(np.nan)
            elif i <= 1:
                if_class_list.append("<1")
            elif i <= 5:
                if_class_list.append("1-5")
            elif i <= 10:
                if_class_list.append("5-10")
            elif i <= 15:
                if_class_list.append("10-15")
            elif i <= 20:
                if_class_list.append("15-20")
            elif i <= 25:
                if_class_list.append("20-25")
            elif i <= 30:
                if_class_list.append("25-30")
            elif i > 30:
                if_class_list.append(">30")
            else:
                print(i)

        df = pd.DataFrame({"cord_uid": cord_uid_list, "pub_time": pub_time_list,
                           "year": year_list, "month": month_list, "author_num": author_num_list,
                           "if_class": if_class_list, "if": if_list,
                           "ref_age_median": ref_age_median_list, "ref_num": ref_num_list,
                           "special_date": special_date_list, "tfidf_median": tfidf_median_list,
                           "w2v_median": w2v_median_list, "lda_median": lda_median_list,
                           "sbert_median": sbert_median_list,
                           "lda_div": lda_div_list, "tfidf_div": tfidf_div_list, "sbert_div": sbert_div_list})
        return df

    def draw_lineplot(self):
        year_dic = defaultdict(int)
        month_dic = defaultdict(int)
        for pub_time in self.pub_time_dic.values():
            if pub_time:
                year = pub_time.split("-")[0]
                if len(pub_time) == 10:
                    special_date = pub_time[:7]
                    if year in ("2019", "2020", "2021"):
                        month_dic[special_date] += 1
                year_dic[year] += 1

        year_tuple = sorted(list(year_dic.items()), key=lambda X: eval(X[0]))
        month_tuple = sorted(list(month_dic.items()), key=lambda X: (X[0].split("-")[0], X[0].split("-")[1]))

        year_list = [i[0] for i in year_tuple]
        new_year_list = []
        for i, j in enumerate(year_list):
            if i % 3 != 0:
                new_year_list.append("")
            else:
                new_year_list.append(j[2:])

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=range(22), y=[math.log(i[1], 100) for i in year_tuple], marker="o", markersize=7, color="black")
        plt.grid(axis="y")
        plt.xticks(range(22), new_year_list)
        plt.xlabel("Year")
        plt.ylabel("log$_\mathrm{\mathregular{100}}$(Paper number)")
        plt.savefig(self.results_root + 'count_year.pdf', bbox_inches='tight', dpi=200)
        plt.close()

        month_list = [i[0] for i in month_tuple]
        new_month_list = []
        for i, j in enumerate(month_list):
            if i % 5 != 0:
                new_month_list.append("")
            else:
                new_month_list.append(j[2:])

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=range(36), y=[math.log(i[1], 100) for i in month_tuple], marker="o", markersize=7, color="black")
        plt.grid(axis="y")
        plt.xticks(range(36), new_month_list)
        plt.xlabel("Month")
        plt.ylabel("log$_\mathrm{\mathregular{100}}$(Paper number)")
        plt.savefig(self.results_root + 'count_month.pdf', bbox_inches='tight', dpi=200)
        plt.close()

    def mkdir(self, dir_name):
        if not os.path.isdir(self.results_root + dir_name):
            os.mkdir(self.results_root + dir_name)

    def indicator_exchange(self):
        indicator_exchange = {"author_num": "Author number",
                              "ref_age_median": "Reference age",
                              "ref_num": "Reference number",
                              "lda_median": "Topic Support",
                              "if": "Journal impact factor",
                              "tfidf_median": "Lexical support",
                              "w2v_median": "Semantic support",
                              "ref_support": "Textual support",
                              "sbert_median": "Semantic support"}
        return indicator_exchange

    def restrict_min(self, x, indicator):

        if x < 0:
            x = 0
        return x

    def draw_heatmap_year(self):
        for indicator in ["author_num", "ref_age_median", "ref_num", "tfidf_median", "lda_median", "w2v_median",
                          "sbert_median"]:
            pt = self.df.pivot_table(index='if_class', columns='year', values=indicator, aggfunc=np.median)
            l = ['>30', '25-30', '20-25', '15-20', '10-15', '5-10', '1-5', '<1']
            pt["st"] = pt.index.astype('category')
            pt['st'].cat.reorder_categories(l, inplace=True)
            pt.sort_values('st', inplace=True)
            pt.set_index(["st"])
            del pt["st"]
            pt = pt.applymap(lambda X: self.restrict_min(X, indicator))
            pt.columns = [i[2:] for i in pt.columns]
            f, ax = plt.subplots(figsize=(20, 4), dpi=80)
            vmax = self.restrict_dic[indicator]
            if indicator == "ref_num" or indicator == "author_num":
                fmt = ".0f"
            elif indicator == "ref_age_median":
                fmt = ".1f"
            else:
                fmt = ".2f"
            if indicator == "w2v_median":
                sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, cbar=True, fmt=fmt, vmin=0.35, vmax=0.8,
                            annot_kws={'size': 15})
            elif indicator == "sbert_median":
                sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, mask=False, fmt=fmt, vmin=0.7, vmax=0.9,
                            annot_kws={'size': 15})
            else:
                sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, cbar=True, fmt=fmt, vmax=vmax,
                            annot_kws={'size': 15})
            ax.set_xlabel('Year')
            ax.set_ylabel('Journal impact factor')

            dir_name = indicator
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/2000_2021_{}_heatmap.pdf'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_heatmap_month(self):

        df_2019_2020_2021 = self.df[(self.df["year"].isin(["2019", "2020", "2021"]))]
        for indicator in ["author_num", "ref_age_median", "ref_num", "tfidf_median", "lda_median", "w2v_median",
                          "sbert_median"]:
            pt = df_2019_2020_2021.pivot_table(index='if_class', columns='special_date', values=indicator,
                                               aggfunc=np.median)
            l = ['>30', '25-30', '20-25', '15-20', '10-15', '5-10', '1-5', '<1']
            pt["st"] = pt.index.astype('category')
            pt['st'].cat.reorder_categories(l, inplace=True)
            pt.sort_values('st', inplace=True)
            pt.set_index(["st"])
            del pt["st"]
            pt = pt.applymap(lambda X: self.restrict_min(X, indicator))
            pt.columns = [i[2:] for i in pt.columns]
            f, ax = plt.subplots(figsize=(26, 4), dpi=80)
            vmax = self.restrict_dic[indicator]
            if indicator == "ref_num" or indicator == "author_num":
                fmt = ".0f"
            elif indicator == "ref_age_median":
                fmt = ".1f"
            else:
                fmt = ".2f"
            if indicator in ("lda_median", "tfidf_median"):
                sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, mask=False, fmt=fmt,
                            annot_kws={'size': 14}, vmax=vmax)
            elif indicator in ("w2v_median"):
                sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, mask=False, fmt=fmt, vmin=0.4, vmax=0.75,
                            annot_kws={'size': 14})
            elif indicator in ("sbert_median"):
                sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, mask=False, fmt=fmt, vmin=0.7, vmax=0.9,
                            annot_kws={'size': 15})
            else:
                sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, mask=False, fmt=fmt, vmax=vmax,
                            annot_kws={'size': 15})
            ax.set_xlabel('Month')
            ax.set_ylabel('Journal impact factor')
            dir_name = indicator
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/2019_2021_{}_heatmap.pdf'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)

    def draw_boxplot_journal_year(self):
        df = self.df
        l = ['>30', '25-30', '20-25', '15-20', '10-15', '5-10', '1-5', '<1']
        for indicator in ["author_num", "ref_age_median", "ref_num", "tfidf_median", "lda_median", "w2v_median",
                          "sbert_median"]:
            fig = plt.figure(figsize=(4, 6), dpi=60)
            ax = sns.boxplot(y=df["if_class"], x=df[indicator], showfliers=False, order=l,
                             orient='h')
            plt.xticks(rotation=90, fontsize=20)
            plt.grid(axis="x")
            plt.xlabel("")
            plt.ylabel("")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)
            for patch in ax.artists:
                patch.set_facecolor((0, 0, 0, 0))
            dir_name = indicator
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/journals_class_{}_2000_2021.pdf'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_boxplot_journal_month(self):
        df = self.df
        df_2019_2020_2021 = df[(df["year"].isin(
            ["2019", "2020", "2021"]))]
        l = ['>30', '25-30', '20-25', '15-20', '10-15', '5-10', '1-5', '<1']
        for indicator in ["author_num", "ref_age_median", "ref_num", "tfidf_median", "lda_median", "w2v_median",
                          "sbert_median"]:
            plt.figure(figsize=(4, 6), dpi=60)
            ax = sns.boxplot(y=df_2019_2020_2021["if_class"], x=df_2019_2020_2021[indicator], showfliers=False, order=l,
                             orient='h')
            plt.xticks(rotation=90, fontsize=20)
            plt.ylabel("")
            plt.xlabel("")
            plt.grid(axis="x")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)
            for patch in ax.artists:
                patch.set_facecolor((0, 0, 0, 0))
            dir_name = indicator
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/journals_class_{}_2019_2021.pdf'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_boxplot_year(self):
        df = self.df
        df_2000_2021 = df[df["year"].notna()]
        grouped_year_median = df_2000_2021.groupby(df_2000_2021["year"]).median()
        old_list = [str(i) for i in list(range(2000, 2022))]
        new_list = [str(i)[-2:] for i in old_list]
        max_length = len(new_list)
        for indicator in ["author_num", "if", "ref_age_median", "ref_num", "tfidf_median", "lda_median", "w2v_median",
                          "sbert_median"]:
            plt.figure(figsize=(12, 8), dpi=60)
            ax = sns.boxplot(x=df_2000_2021["year"], y=df_2000_2021[indicator], showfliers=False,
                             order=old_list)
            sns.lineplot(data=grouped_year_median, x=grouped_year_median.index, y=indicator, color="black", linewidth=2,
                         marker="o", markersize=8)
            plt.xticks(range(max_length), new_list, rotation=90, fontsize=22)
            plt.yticks(fontsize=22)
            plt.xlabel("Year", fontsize=25)
            plt.ylabel(self.indicator_exchange[indicator], fontsize=25)
            plt.grid(axis="y")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            num = 0
            for patch in ax.artists:
                num += 1
                if num in (4, 14, 21, 22):
                    patch.set_facecolor((168 / 255, 206 / 255, 228 / 255, 1))
                else:
                    r, g, b, a = patch.get_facecolor()
                    patch.set_facecolor((r, g, b, 0))
            dir_name = indicator
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/2000_2021_{}_boxplot_blue.pdf'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_boxplot_month(self):
        df = self.df
        df_2019_2020_2021 = df[(df["year"].isin(["2019", "2020", "2021"])) & (df["special_date"].notna())]

        grouped_month_median = df_2019_2020_2021.groupby(df_2019_2020_2021["special_date"]).median()
        month_list = sorted(list(set(df_2019_2020_2021["special_date"])),
                            key=lambda x: (x.split("-")[0], x.split("-")[1]))
        old_list = [i[2:] for i in month_list]
        max_length = len(set(df_2019_2020_2021["special_date"]))
        new_list = []
        for i, j in enumerate(old_list):
            if i % 5 == 0:
                new_list.append(j)
            else:
                new_list.append("")
        for indicator in ["author_num", "if", "ref_age_median", "ref_num", "tfidf_median", "lda_median", "w2v_median",
                          "sbert_median"]:
            plt.figure(figsize=(12, 8), dpi=60)
            ax = sns.boxplot(x=df_2019_2020_2021["special_date"], y=df_2019_2020_2021[indicator], showfliers=False,
                             order=month_list)
            sns.lineplot(data=grouped_month_median, x=grouped_month_median.index, y=indicator, color="black",
                         linewidth=2, marker="o", markersize=8)
            plt.xticks(range(max_length), new_list, rotation=90, fontsize=22)
            plt.yticks(fontsize=22)
            plt.grid(axis="y")
            plt.xlabel("Month", fontsize=25)
            plt.ylabel(self.indicator_exchange[indicator], fontsize=25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            num = 0
            for patch in ax.artists:
                num += 1
                if num >= 13 and num < 25:
                    patch.set_facecolor((168 / 255, 206 / 255, 228 / 255, 0.5))
                elif num >= 25:
                    patch.set_facecolor((168 / 255, 206 / 255, 228 / 255, 1))
                else:
                    r, g, b, a = patch.get_facecolor()
                    patch.set_facecolor((r, g, b, 0))
            dir_name = indicator
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/2019_2021_{}_boxplot_blue.pdf'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_relation(self):
        df = self.df
        df["ref_support"] = df[["w2v_median", "lda_median", "tfidf_median", "sbert_median"]].mean(axis=1)
        indicator_exchange = self.indicator_exchange
        num = 0
        for year_scope in [["2019"], ["2020"], ["2021"]]:
            num += 1
            df_select_year = df[df["year"].isin(year_scope)]
            corr_df = df_select_year[
                ["ref_support", "ref_age_median", "ref_num", "author_num", "if"]]
            corr_df.columns = [indicator_exchange[i] for i in
                               ["ref_support", "ref_age_median", "ref_num",
                                "author_num", "if"]]
            mask = np.zeros_like(corr_df.corr(), dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            plt.subplots(figsize=(8, 8))
            sns.diverging_palette(240, 10, n=100)
            sns.heatmap(corr_df.corr(), square=True, annot=True, cmap="vlag", center=0, mask=mask, vmin=-0.2,
                        vmax=0.2)
            dir_name = "relation"
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/relation_{}.pdf'.format(dir_name, year_scope[0]), bbox_inches='tight',
                        dpi=200)
            plt.close()

    def lineplot_ref_age(self):
        indicators = ["ref_num", "tfidf_median", "lda_median", "sbert_median"]

        df = self.df

        for i in range(4):
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
            indicator = indicators[i]
            df_select = df[(df["ref_age_median"] <= 1) & (df["year"] == str(2019))]
            grouped_month_median_small_ra = list(df_select.groupby(df_select["special_date"]).median()[indicator])
            ax1.plot(grouped_month_median_small_ra, marker="o", markersize=7, color="black")
            df_select = df[(df["ref_age_median"] >= 5) & (df["year"] == str(2019))]
            grouped_month_median_small_ra = list(df_select.groupby(df_select["special_date"]).median()[indicator])
            ax1.plot(grouped_month_median_small_ra, marker="o", markersize=7, color="lightsteelblue")
            new_year_list = ["19-" + str(i) for i in range(1, 12 + 1, 3)]
            ax1.set_xticks(range(0, 12, 3), new_year_list)

            df_select = df[(df["ref_age_median"] <= 1) & (df["year"] == str(2020))]
            grouped_month_median_small_ra = list(df_select.groupby(df_select["special_date"]).median()[indicator])
            ax2.plot(grouped_month_median_small_ra, marker="o", markersize=7, color="black")
            df_select = df[(df["ref_age_median"] >= 5) & (df["year"] == str(2020))]
            grouped_month_median_small_ra = list(df_select.groupby(df_select["special_date"]).median()[indicator])
            ax2.plot(grouped_month_median_small_ra, marker="o", markersize=7, color="lightsteelblue")
            new_year_list = ["20-" + str(i) for i in range(1, 12 + 1, 3)]
            ax2.set_xticks(range(0, 12, 3), new_year_list)
            plt.draw()
            plt.savefig("lineplot_ra_{}.pdf".format(indicator), dpi=400)

    def relation(self):
        years = [2020, 2019]
        for year in years:
            if year == 2020:
                color_new = "powderblue"
            elif year == 2019:
                color_new = "lightsteelblue"
            labels = [str(i) for i in range(0, 11)]
            df = self.df

            if year == 2019:
                df_select = df[(df["ref_age_median"] < 12) & (df["year"] == str(year))]
                print(df_select.groupby(df_select["ref_age_median"]).median()["tfidf_median"])
            elif year == 2020:
                df_select = df[
                    (df["ref_age_median"] < 12) & (df["year"] == str(year))]

            ref_age_list = np.array(df_select["ref_age_median"])
            fig = plt.figure(figsize=(18, 10))
            gs = gridspec.GridSpec(2, 6, figure=fig, bottom=0.1, top=0.99, left=0.08, right=0.99, hspace=0.3)
            gs.update(wspace=0.9)

            ax1 = plt.subplot(gs[0, :2])
            ax2 = plt.subplot(gs[0, 2:4])
            ax3 = plt.subplot(gs[0, 4:6])

            ax4 = plt.subplot(gs[1, 1:3])
            ax5 = plt.subplot(gs[1, 3:5])
            indicator_ori_labels = ["ref_num", "tfidf_median", "lda_median", "sbert_median"]
            indicator_labels = ["Ref Num", "Lexical Support", "Topic Support", "Semantic Support"]
            indicator_axes = [ax1, ax2, ax4, ax5]
            for i in range(4):
                indicator = indicator_ori_labels[i]
                indicator_label = indicator_labels[i]
                ax = indicator_axes[i]
                indicator_list = np.array(df_select[indicator])
                violin_data = []
                for ra in range(0, 11):
                    data = [ra_i for ra_i in indicator_list[ref_age_list == ra] if ra_i == ra_i]
                    violin_data.append(data)

                bplot = ax.boxplot(violin_data,
                                   notch=True,
                                   vert=False,
                                   patch_artist=True,
                                   labels=labels,
                                   showfliers=False)
                colors = ['white'] * 11
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)

                ax.yaxis.grid(False)
                ax.set_xlabel(indicator_label)
                if i == 0 or i == 2:
                    print("Label it!")
                    ax.set_ylabel("Ref Age")
                else:
                    ax.set_yticks([])

            bar_data = []
            paper_num_all = len(ref_age_list)
            for ra in range(0, 11):
                num = list(ref_age_list).count(ra) / paper_num_all
                bar_data.append(num)
            ax3.barh(range(0, 11), bar_data, align='center', color=color_new)
            ax3.set_yticks(range(0, 11))

            ax3.yaxis.grid(False)
            ax3.set_xlabel('Paper Num')
            ax3.xaxis.set_major_formatter(PercentFormatter(xmax=1))
            plt.show()

    def _get_right_time_line(self, ori_time_list, time="month"):
        if time == "month":
            ori_time_list = [i for i in ori_time_list if i == i]
            time_line = sorted(list(set(ori_time_list)),
                               key=lambda x: (x.split("-")[0], x.split("-")[1]))
        elif time == "year":
            ori_time_list = [i for i in ori_time_list if i == i]
            time_line = sorted(list(ori_time_list), reverse=False)
        return time_line

    def diversity_viz(self):
        df = self.df
        month_list = self._get_right_time_line(list(df[df["year"].isin(["2019", "2020", "2021"])]["special_date"]),
                                               time="month")
        year_list = [str(i) for i in list(range(2000, 2022))]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 2))
        plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.3)
        indicator_names = ["tfidf_div", "lda_div", "sbert_div"]

        for indicator_name in indicator_names:
            mean_of_ind = np.nanmean(df[indicator_name])
            std_of_ind = np.nanstd(df[indicator_name])
            df[indicator_name + "_std"] = df[indicator_name].apply(lambda x: (x - mean_of_ind) / std_of_ind)
        df['ts_std'] = df[[i + "_std" for i in indicator_names]].mean(axis=1)

        indicator_name = 'ts_std'
        time_divs_dic, time_div_median_dic, div_median_list = self.get_time_div(time_line=month_list, df=df,
                                                                                indicator=indicator_name, time="month")
        col = 1
        ax = axes[col]
        self._clean_ax(ax)
        self._set_xticks(ax, time_line=month_list, time="month")
        ax.grid(linestyle="dotted", color="slategray", alpha=0.3, linewidth=0.8, axis="both")
        ax.plot(div_median_list, marker='o', linewidth=3)

        time_divs_dic, time_div_median_dic, div_median_list = self.get_time_div(time_line=year_list, df=df,
                                                                                indicator=indicator_name, time="year")
        col = 0
        ax = axes[col]
        self._clean_ax(ax)
        self._set_xticks(ax, time_line=year_list, time="year")
        ax.plot(div_median_list, marker='o', linewidth=3)
        ax.grid(linestyle="dotted", color="slategray", alpha=0.3, linewidth=0.8, axis="both")
        plt.show()

    def get_time_div(self, time_line, df, indicator, time="month"):
        if time == "month":
            ref = "special_date"
        elif time == "year":
            ref = "year"
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

    def _clean_ax(self, ax):
        bwith = 1

        ax.spines['left'].set_linewidth(bwith)
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(width=bwith)

    def _set_xticks(self, ax, time_line, time):
        if time == "month": cut_length = 5
        if time == "year": cut_length = 2
        new_time_line = []
        for i, j in enumerate(time_line):
            if i % 5 == 0:
                new_time_line.append(j[-cut_length:])
            else:
                new_time_line.append("")
        ax.set_xticks(range(len(time_line)), new_time_line, rotation=90)


if __name__ == '__main__':
    params = {'font.family': 'serif',
              'font.serif': 'Times New Roman',
              'font.style': 'normal',
              'font.weight': 'normal',
              'font.size': 10,
              }
    plt.rcParams.update(params)
    print("init...")
    draw_pic = draw_pic()
    draw_pic.draw_lineplot()
    print("done lineplot...")
    draw_pic.draw_heatmap_year()
    print("done heatmap_year...")
    draw_pic.draw_heatmap_month()
    print("done heatmap_month...")
    draw_pic.draw_boxplot_journal_year()
    print("done boxplot_journal_year...")
    draw_pic.draw_boxplot_journal_month()
    print("done boxplot_journal_month...")
    draw_pic.draw_boxplot_year()
    print("done boxplot_year...")
    draw_pic.draw_boxplot_month()
    print("done boxplot_month...")
    draw_pic.relation()
    draw_pic.diversity_viz()
    print("done relation...")
    draw_pic.lineplot_ref_age()

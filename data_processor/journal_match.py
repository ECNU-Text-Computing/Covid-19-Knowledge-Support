import argparse
import json
import re
from collections import Counter

import pandas as pd


class JournalMatch(object):
    def __init__(self):
        self.tmp_file_root = "../tmp_file/"
        self.jif_match_root = self.tmp_file_root + "jif_match/"
        self.xlsx_abb2fu = self.jif_match_root + "SCI&SSCI-2019-abb-fix.xlsx"
        self.cord_uid_info = self.tmp_file_root + "cord_uid_info.txt"
        self.cord_uid_ref = self.tmp_file_root + "cord_uid_ref.json"
        self.xlsx_journal2field = "../Dataset/scimagojr 2021.csv"
        self.subject_categories_dic = "../Dataset/subject_categories_dic.json"
        pass

    def get_abb2fu_dic(self):
        df_abb2fu = pd.read_excel(self.xlsx_abb2fu)
        funame_list = df_abb2fu['Full Journal Title']
        abb_list = df_abb2fu['abbreviation']
        abb_list_clean = [self.clean_journal(i) for i in abb_list]
        funame_list_clean = [self.clean_journal(i) for i in funame_list]
        abb2fu_dic = dict(zip(abb_list_clean, funame_list_clean))
        print("got the abb2fu dic.")
        return abb2fu_dic

    def get_journals_of_info(self):
        num = 0
        journal_list = []
        with open(self.cord_uid_info, "r", encoding="utf-8") as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                num += 1

                info = json.loads(line)
                journal = info["journal"]
                if journal.strip():
                    journal_list.append(journal)
        journal_list = list(set(journal_list))
        print("got the journal in cord-19 dataset info, the journal num is (before processing):", len(journal_list))
        return journal_list

    def clean_journal(self, content):
        new_content = "".join(re.findall("[a-z]*", re.sub("(\(.*\)|)", "", content.lower())))
        return new_content

    def get_journal_clean2ori(self, journal_list, abb2fu_dic):
        journal_clean2ori = {}
        for journal in journal_list:
            journal_key = self.clean_journal(journal)
            if journal_key in abb2fu_dic.keys():
                journal_key = abb2fu_dic[journal_key]
            if journal_key not in journal_clean2ori:
                journal_clean2ori[journal_key] = [journal]
            else:
                journal_clean2ori[journal_key].append(journal)
        print("journal after clean & abb2fu, the journal num is:", len(journal_clean2ori))
        return journal_clean2ori

    def clean_discipline(self, discipline):
        discipline = discipline.replace("\"", "").strip()
        discipline = re.sub(r"\(Q\d*\)", '', discipline).strip().lower()
        return discipline

    def get_journal2field_dic(self):
        journal2field_dic = {}
        with open(self.xlsx_journal2field, "r", encoding="utf-8") as csvfile:
            while True:
                line = csvfile.readline().strip()
                if not line:
                    break
                cut_line = line.split(";")
                journal_info = cut_line[:18]
                discipline_info = cut_line[19:]

                journal_name = journal_info[2].replace("\"", "").strip()
                journal_key = self.clean_journal(journal_name)
                if journal_key and journal_info[3] == "journal":
                    journal2field_dic[journal_key] = [self.clean_discipline(i) for i in discipline_info]
        print("the length of journal2field:", len(journal2field_dic))
        print("used sjr classification to get the journal2field dic.")
        return journal2field_dic

    def get_journal_ori2field(self, journal2field_dic, journal_clean2ori):
        journal_ori2field = {}
        for journal_key, journal_oris in journal_clean2ori.items():
            for journal_ori in journal_oris:
                if journal_key in journal2field_dic.keys():
                    journal_field = journal2field_dic[journal_key]
                    journal_ori2field[journal_ori] = journal_field
        print("length of journal_ori2field:", len(journal_ori2field))
        print("got journal_ori2field")
        return journal_ori2field

    def journal_field_match(self):
        abb2fu_dic = self.get_abb2fu_dic()
        journal_list = self.get_journals_of_info()
        journal_clean2ori = self.get_journal_clean2ori(journal_list, abb2fu_dic)
        journal2field_dic = self.get_journal2field_dic()
        journal_ori2field = self.get_journal_ori2field(journal2field_dic, journal_clean2ori)
        category_subject_dic = self.get_category_subject_dic()
        self.journal_match_info(journal_ori2field, category_subject_dic)

    def journal_match_info(self, journal_ori2field, category_subject_dic):
        num = 0
        journal_field_list = []
        journals_with_field = set(journal_ori2field.keys())
        with open(self.cord_uid_info, "r", encoding="utf-8") as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                num += 1

                info = json.loads(line)
                journal = info["journal"]
                if journal.strip() and journal in journals_with_field:
                    journal_field = journal_ori2field[journal]
                    journal_field_list += journal_field
        print("got it!")
        print(len(journal_field_list))
        category_count_dic = Counter(journal_field_list)
        with open(self.tmp_file_root + "journal_category_count_dic.json", "w", encoding="utf-8") as fw:
            json.dump(category_count_dic, fw, ensure_ascii=False)

        subject_count_dic = {}
        for category, count_num in category_count_dic.items():
            if category in category_subject_dic:
                subject = category_subject_dic[category]
                subject_count_dic[subject] = subject_count_dic.get(subject, 0) + count_num
            else:
                print(category)
        with open(self.tmp_file_root + "journal_subject_count_dic.json", "w", encoding="utf-8") as fw:
            json.dump(subject_count_dic, fw, ensure_ascii=False)

    def get_category_subject_dic(self):
        with open(self.subject_categories_dic, "r", encoding="utf-8") as fp:
            subject_categories_dic = json.load(fp)
        category_subject_dic = {}
        for subject, categories in subject_categories_dic.items():
            for category in categories:
                category_subject_dic[category] = subject
        return category_subject_dic


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--func", type=str)
    JournalMatch = JournalMatch()
    JournalMatch.journal_field_match()

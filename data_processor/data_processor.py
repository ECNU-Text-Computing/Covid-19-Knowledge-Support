import csv
import json
import re

import numpy as np
import pandas as pd
from nltk.metrics import edit_distance


class DataProcessor(object):
    def __init__(self):
        self.dataset = '../Dataset/'
        self.tmp_root = "../tmp_file/"
        self.cord_root = self.dataset + '2022-01-03/'
        self.document_parses_path = self.cord_root + 'document_parses/'
        self.metadata_path = self.cord_root + 'metadata.csv'
        self.results_root = "../Results/"
        self.jif_match_root = self.tmp_root + "jif_match/"

    def get_info(self, before_2000=True):
        if before_2000:
            time_select_condition = lambda year: year < 2000
        else:
            time_select_condition = lambda year: (year >= 2000) and (year <= 2021)
        cord_uid_dict = {}
        same_title = 0
        diff_title = 0
        with open(self.metadata_path, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                cord_uid = row['cord_uid'].strip()
                title = row['title'].strip()
                abstract = row['abstract'].strip()
                pub_time = row['publish_time'].strip()
                authors = row['authors'].strip()
                authors_num = len(authors.strip().split(';'))
                journal = row['journal'].strip()
                pdf_json_files = row['pdf_json_files'].strip().split(';')
                pmc_json_files = row['pmc_json_files'].strip().split(';')
                json_files = pmc_json_files + pdf_json_files
                if pub_time:
                    year = int(pub_time.split("-")[0])
                    if time_select_condition(year):
                        if cord_uid not in cord_uid_dict:
                            cord_uid_dict[cord_uid] = {'cord_uid': cord_uid,
                                                       'title': title,
                                                       'abstract': abstract,
                                                       'pub_time': pub_time,
                                                       'authors': authors,
                                                       'authors_num': authors_num,
                                                       'journal': journal,
                                                       'json_files': ';'.join(json_files)}
                        else:
                            if edit_distance(cord_uid_dict[cord_uid]['title'].lower(), title.lower()) < 5:
                                if len(json_files) > 0:
                                    cord_uid_dict[cord_uid]['json_files'] = cord_uid_dict[cord_uid][
                                                                                'json_files'] + ';' + ';'.join(
                                        json_files)
                                same_title += 1
                            else:
                                diff_title += 1
                                print(cord_uid_dict[cord_uid]['title'])
                                print(title)
                                print('*' * 20)
        print("There are {} same title when facing the same uid.".format(same_title))
        print("There are {} different title when facing the same uid.".format(diff_title))

        if before_2000:
            cord_uid_info_list_path = self.tmp_root + 'cord_uid_info_before_2000.txt'
        else:
            cord_uid_info_list_path = self.tmp_root + 'cord_uid_info.txt'
        with open(cord_uid_info_list_path, 'w', encoding="utf-8") as fw:
            for cord_uid, info_dict in cord_uid_dict.items():
                fw.write(json.dumps(info_dict, ensure_ascii=False) + '\n')

    def get_ref(self, before_2000=True):
        if before_2000:
            cord_uid_info_list_path = self.tmp_root + 'cord_uid_info_before_2000.txt'
            cord_uid_ref_dict_path = self.tmp_root + 'cord_uid_ref_before_2000.json'
        else:
            cord_uid_info_list_path = self.tmp_root + 'cord_uid_info.txt'
            cord_uid_ref_dict_path = self.tmp_root + 'cord_uid_ref.json'
        count = 0
        error_count = 0
        cord_uid_ref_dict = {}
        with open(cord_uid_info_list_path, 'r', encoding='utf-8') as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                count += 1
                info = json.loads(line)
                cord_uid = info['cord_uid']
                json_files = info['json_files']
                key = True
                json_file_list = json_files.strip().split(';')
                for path in json_file_list:
                    if path.strip():
                        new_bib_entries = []
                        print(path)
                        with open(self.cord_root + "/document_parses/" + path, 'r',
                                  encoding='utf-8') as fp1:
                            paper = json.load(fp1)
                            bib_entries = paper['bib_entries']
                            for _, bib_ref in bib_entries.items():
                                title = bib_ref['title'].strip()
                                authors = len(bib_ref['authors'])
                                year = bib_ref['year']
                                venue = bib_ref['venue']
                                new_bib_ref = {'title': title, 'authors': authors, 'year': year, 'venue': venue}
                                new_bib_entries.append(new_bib_ref)
                        if len(new_bib_entries):
                            cord_uid_ref_dict[cord_uid] = new_bib_entries
                            key = False
                            break
                if key:
                    error_count += 1
                    print(cord_uid)
        print('There are {} lines in metadata.'.format(count))
        print('There are {} lines in ref_dict.'.format(len(cord_uid_ref_dict)))
        print('There are {} error lines in metadata.'.format(error_count))
        with open(cord_uid_ref_dict_path, 'w', encoding="utf-8") as fw:
            json.dump(cord_uid_ref_dict, fw, ensure_ascii=False)

    def get_ref_before(self):

        cord_uid_dict = {}
        same_title = 0
        diff_title = 0
        with open(self.metadata_path, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                cord_uid = row['cord_uid'].strip()
                title = row['title'].strip()
                abstract = row['abstract'].strip()
                pub_time = row['publish_time'].strip()
                authors = row['authors'].strip()
                authors_num = len(authors.strip().split(';'))
                journal = row['journal'].strip()
                pdf_json_files = row['pdf_json_files'].strip().split(';')
                pmc_json_files = row['pmc_json_files'].strip().split(';')
                json_files = pmc_json_files + pdf_json_files
                if pub_time:
                    year = int(pub_time.split("-")[0])
                    if year < 2000:
                        if cord_uid not in cord_uid_dict:
                            cord_uid_dict[cord_uid] = {'cord_uid': cord_uid,
                                                       'title': title,
                                                       'abstract': abstract,
                                                       'pub_time': pub_time,
                                                       'authors': authors,
                                                       'authors_num': authors_num,
                                                       'journal': journal,
                                                       'json_files': ';'.join(json_files)}
                        else:
                            if edit_distance(cord_uid_dict[cord_uid]['title'].lower(),
                                             title.lower()) < 5:
                                if len(json_files) > 0:
                                    cord_uid_dict[cord_uid]['json_files'] = cord_uid_dict[cord_uid][
                                                                                'json_files'] + ';' + ';'.join(
                                        json_files)
                                same_title += 1
                            else:
                                diff_title += 1
                                print(cord_uid_dict[cord_uid]['title'])
                                print(title)
                                print('*' * 20)
        print("There are {} same title when facing the same uid.".format(same_title))
        print("There are {} different title when facing the same uid.".format(diff_title))

        cord_uid_info_list_path = self.tmp_root + 'cord_uid_info_before_2020.txt'
        with open(cord_uid_info_list_path, 'w', encoding="utf-8") as fw:
            for cord_uid, info_dict in cord_uid_dict.items():
                fw.write(json.dumps(info_dict, ensure_ascii=False) + '\n')
        pass

    def clean_list(self, content_list):
        new_list = ["".join(re.findall("[a-z]*", re.sub("(\(.*\)|)", "", content.lower()))) for content in content_list]
        return new_list

    def clean_journal(self, content, subdic_abb2fu):
        new_content = "".join(re.findall("[a-z]*", re.sub("(\(.*\)|)", "", content.lower())))
        if new_content in subdic_abb2fu.keys():
            new_content = subdic_abb2fu[new_content]
        return new_content

    def get_journal(self):
        with open(self.tmp_root + "cord_uid_ref.json", "r", encoding="utf-8") as fp:
            ref_dic = json.load(fp)
        journal_list = []
        print("read journal...")
        with open(self.tmp_root + "cord_uid_info.txt", "r", encoding="utf-8") as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                info = json.loads(line)
                cord_uid = info["cord_uid"]
                if cord_uid in ref_dic.keys():
                    journal = info["journal"]
                    journal_list.append(journal)
        journal_set_ori = set(journal_list)
        print("初始状态下，一共有{}种不同的journal".format(len(journal_set_ori)))
        df_abb_fu = pd.read_excel(self.jif_match_root + "SCI&SSCI-2019-abb-fix.xlsx")
        subdic_abb2fu = dict(
            zip(self.clean_list(df_abb_fu["abbreviation"]), self.clean_list(df_abb_fu["Full Journal Title"])))
        sub_dic = {}
        for journal in journal_set_ori:
            new_journal = self.clean_journal(journal, subdic_abb2fu)
            if new_journal not in sub_dic:
                sub_dic[new_journal] = [journal]
            else:
                sub_dic[new_journal].append(journal)
        journal_set = set(sub_dic.keys())
        print("经过大小写清洗、去除括号、仅保留字母a-z过后，一共有{}种不同的journal".format(len(journal_set)))

        df_jcr2021 = pd.read_excel(self.jif_match_root + "JCR2021.xlsx")
        ifs = df_jcr2021["Journal Impact Factor"]
        funame_set = self.clean_list(df_jcr2021["Full Journal Title"])
        fu_clean2ori = dict(zip(funame_set, df_jcr2021["Full Journal Title"]))
        funame2if_dic = dict(zip(funame_set, ifs))
        print("已经获取了匹配JCR2021.xlsx文档中的全名和IF的对照关系")

        original_journal_name = []
        ori_funame = []
        journal_in_jcr_list = []
        ifs_list = []
        for journal_in_jcr in journal_set & set(funame_set):
            for i in sub_dic[journal_in_jcr]:
                original_journal_name.append(i)
                ori_funame.append(fu_clean2ori[journal_in_jcr])
                journal_in_jcr_list.append(journal_in_jcr)
                ifs_list.append(funame2if_dic[journal_in_jcr])
        df = pd.DataFrame(
            {"original_name": original_journal_name, "clean_name": journal_in_jcr_list, "full_name": ori_funame,
             "jif": ifs_list})
        df.to_excel(self.jif_match_root + "JCRmatch.xlsx")
        print("已生成对照excel...")

    def get_dic(self):
        JCRmatch = pd.read_excel(self.jif_match_root + "JCRmatch.xlsx")
        jif_match_dic = dict(zip(JCRmatch["original_name"], JCRmatch["jif"]))

        jif_dic = {}
        author_num_dic = {}
        ref_num_dic = {}
        ref_age_dic = {}
        pub_time_dic = {}
        with open(self.tmp_root + "cord_uid_ref.json", "r", encoding="utf-8") as fp:
            ref_dic = json.load(fp)

        num_paper_2000_2021 = 0
        num_paper_have_refs = 0
        num_paper_have_refs_journals = 0
        with open(self.tmp_root + "cord_uid_info.txt", "r", encoding="utf-8") as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                info = json.loads(line)
                cord_uid = info["cord_uid"]
                author_num = info["authors_num"]
                pub_time = info["pub_time"]
                year = int(pub_time.split("-")[0])
                journal = info["journal"]
                if cord_uid in ref_dic.keys():
                    num_paper_have_refs += 1
                    refs = ref_dic[cord_uid]
                    ref_num = len(refs)
                    ref_year_list = [ref["year"] for ref in refs if ref["year"]]
                    ref_age_median = year - np.median(ref_year_list)
                    if ref_age_median < 0:
                        ref_age_median = 0
                    ref_num_dic[cord_uid] = ref_num
                    ref_age_dic[cord_uid] = ref_age_median
                    author_num_dic[cord_uid] = author_num
                    pub_time_dic[cord_uid] = pub_time
                    if journal:
                        if journal in jif_match_dic.keys():
                            num_paper_have_refs_journals += 1
                            jif = jif_match_dic[journal]
                            jif_dic[cord_uid] = jif

        print("the number of papers between 2000 to 2021:{}".format(num_paper_2000_2021))
        print("the number of papers which published between 2000 to 2021 and have refs:{}".format(num_paper_have_refs))
        print("the number of papers which published between 2000 to 2021, have refs, have journals:{}".format(
            num_paper_have_refs_journals))

        with open(self.results_root + "ref_num_dic.json", "w", encoding="utf-8") as fw:
            json.dump(ref_num_dic, fw, ensure_ascii=False)
        with open(self.results_root + "ref_age_dic.json", "w", encoding="utf-8") as fw:
            json.dump(ref_age_dic, fw, ensure_ascii=False)
        with open(self.results_root + "author_num_dic.json", "w", encoding="utf-8") as fw:
            json.dump(author_num_dic, fw, ensure_ascii=False)
        with open(self.results_root + "pub_time_dic.json", "w", encoding="utf-8") as fw:
            json.dump(pub_time_dic, fw, ensure_ascii=False)
        with open(self.results_root + "jif_dic.json", "w", encoding="utf-8") as fw:
            json.dump(jif_dic, fw, ensure_ascii=False)

    def see_info(self):

        cord_uid_info_list_path = self.tmp_root + 'cord_uid_info.txt'
        cord_uid_ref_dict_path = self.tmp_root + 'cord_uid_ref.json'
        count = 0
        error_count = 0
        cord_uid_ref_dict = {}
        with open(cord_uid_info_list_path, 'r', encoding='utf-8') as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                count += 1
                if count > 1:
                    break
                info = json.loads(line)
                cord_uid = info['cord_uid']
                json_files = info['json_files']
                key = True
                json_file_list = json_files.strip().split(';')
                for path in json_file_list:
                    if path.strip():
                        new_bib_entries = []
                        print(path)
                        with open(self.cord_root + "/document_parses/" + path, 'r',
                                  encoding='utf-8') as fp1:
                            paper = json.load(fp1)
                            bib_entries = paper['bib_entries']
                            for _, bib_ref in bib_entries.items():
                                print(bib_ref)
                                title = bib_ref['title'].strip()
                                authors = len(bib_ref['authors'])
                                year = bib_ref['year']
                                venue = bib_ref['venue']
                                new_bib_ref = {'title': title, 'authors': authors, 'year': year, 'venue': venue}
                                new_bib_entries.append(new_bib_ref)
                            if len(new_bib_entries):
                                cord_uid_ref_dict[cord_uid] = new_bib_entries
                                key = False
                                break
                if key:
                    error_count += 1
                    print(cord_uid)


if __name__ == '__main__':
    dp = DataProcessor()
    dp.get_info()
    dp.get_ref()
    dp.get_journal()
    dp.get_dic()
    dp.see_info()

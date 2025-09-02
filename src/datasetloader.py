# there are five dataset: ajmc, hipe2020, letemps, newseye, sonar, topres19th
import os
import warnings
from src.settings import find_dataset


class Dataset_loader:
        def __init__(self,dataset,language,split,file_path=None):
            self.dataset_name = dataset
            self.language = language
            self.split = split
            if file_path:
                dataset_file = file_path
            else:
                dataset_file = find_dataset(self.dataset_name, self.language, self.split)

            self.documents = dict()
            self.texts = dict()
            self.annotations_coarse = None
            self.annotations_fine = None
            self.annotation_nested = None
            # I got this from train and dev of dataset combined together. It turns out the table in paper contains some errors
            self.entity_label_set = {"ajmc_coarse":{'date', 'pers', 'scope', 'object', 'work', 'loc'},
                                     "hipe2020_coarse":{'pers', 'prod', 'org', 'loc', 'time'},
                                     "letemps_coarse":{'org', 'pers', 'loc'},
                                     "newseye_coarse":{'humanprod', 'org', 'loc', 'per'},
                                     "newseyechunked_coarse":{'humanprod', 'org', 'loc', 'per'},
                                     "sonar_coarse":{'org', 'loc', 'per'},
                                     "topres19th_coarse":{"loc","building","street"},
                                     "ajmc_fine": {'date', 'loc', 'object.manuscr', 'pers.author', 'pers.editor',
                                                   'pers.myth', 'pers.other', 'scope', 'work.fragm', 'work.journal',
                                                   'work.other', 'work.primlit', 'work.seclit'},
                                     "hipe2020_fine": {'loc.add.elec', 'loc.add.phys', 'loc.adm.nat', 'loc.adm.reg',
                                                       'loc.adm.sup', 'loc.adm.town', 'loc.fac', 'loc.oro',
                                                       'loc.phys.astro', 'loc.phys.geo', 'loc.phys.hydro', 'loc.unk',
                                                       'org.adm', 'org.ent', 'org.ent.pressagency', 'pers.coll',
                                                       'pers.ind', 'pers.ind.articleauthor', 'prod.doctr', 'prod.media',
                                                       'time.date.abs'},
                                     "letemps_fine": {'loc', 'loc.add', 'loc.add.phys', 'loc.adm', 'loc.adm.nat',
                                                      'loc.adm.reg', 'loc.adm.town', 'loc.admin.sup', 'loc.oro',
                                                      'loc.phys', 'loc.phys.astro', 'loc.phys.geo', 'loc.phys.hydro',
                                                      'org.adm', 'org.ent', 'pers', 'pers.coll', 'pers.ind'},
                                     "newseye_fine": {'humanprod', 'loc', 'org', 'per', 'per.author'},
                                     "newseyechunked_fine": {'humanprod', 'loc', 'org', 'per', 'per.author'},
                                     "toyset_coarse":{'date', 'pers', 'scope', 'object', 'work', 'loc'},
                                     "toyset_fine": {'date', 'loc', 'object.manuscr', 'pers.author', 'pers.editor',
                                                   'pers.myth', 'pers.other', 'scope', 'work.fragm', 'work.journal',
                                                   'work.other', 'work.primlit', 'work.seclit'},
                                     }


            dataset_languages = {"toyset":{'en'},"ajmc": {"de", "en", "fr"}, "hipe2020": {"de", "en", "fr"}, "letemps": {"fr"},
                                 "newseye": {"de", "fi", "fr", "sv"},"newseyechunked": {"de", "fi", "fr", "sv"}, "sonar": {"de"}, "topres19th": {"en"}}
            if not dataset =="toyset":
                assert(self.language in dataset_languages[self.dataset_name])


            self.load_documents(dataset_file)

        def load_documents(self,file_path):
            documents = []
            column_names = ['token', 'ne_coarse_lit', 'ne_coarse_meto', 'ne_fine_lit', 'ne_fine_meto', 'ne_fine_comp', 'ne_nested', 'nel_lit', 'nel_meto', 'tools']
            current_doc_id = None
            current_doc = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip("\n")

                    if line.startswith("TOKEN	NE-COARSE-LIT"):#skip headlines
                      continue
                    #new document
                    if line.startswith('#'):
                        if line.startswith('# hipe2022:document_id = '):
                            if current_doc:
                                documents.append([current_doc_id] + current_doc)
                                current_doc = []

                            current_doc_id = line.split('=')[1].strip()
                            continue
                        continue

                    # Skip empty lines
                    if not line:
                        continue

                    parts = line.split('\t')



                    if len(parts) > len(column_names):
                        print(parts)
                        raise ValueError(f" {len(parts)} elements in document {current_doc_id}")
                    token = {column_names[i]: parts[i] for i in range(len(parts))}
                    current_doc.append(token)

            if current_doc:
                documents.append([current_doc_id] + current_doc)

            texts = {}
            for sentence in documents:
                # Skip the document ID (first element) and join token texts
                tokens = [token_dict['token'] for token_dict in sentence[1:]]
                sentence_text = ' '.join(tokens)
                texts[sentence[0]] = sentence_text


            for i in documents:
                self.documents[i[0]] = i[1:]
            self.texts = texts
            self.annotations_coarse = self.load_annotations('ne_coarse_lit')
            self.annotations_fine = self.load_annotations('ne_fine_lit')
            if self.dataset_name not in ['sonar',"topres19th"]:#these 2 datasets has no fine tags
                self.annotation_nested = self.load_annotations('ne_nested')

        def load_annotations(self,annotation_name):
            entity_annotations = {}
            has_entities = False

            for doc_id,doc in self.documents.items():
                #skip document ID
                current_entities = []
                current_entity = []
                current_label = None

                for token in doc:
                    ne_tag = token.get(annotation_name, '_')  # Default to '_' if missing

                    # Skip if tag is empty/O
                    if ne_tag in ['_', 'O']:
                        if current_entity and current_label:
                            current_entities.append((' '.join(current_entity), current_label))
                            has_entities = True
                        current_entity = []
                        current_label = None
                        continue

                    # Handle BIO tags
                    if ne_tag.startswith('B-'):
                        if current_entity and current_label:
                            current_entities.append((' '.join(current_entity), current_label))
                            has_entities = True
                        current_label = ne_tag[2:]
                        current_entity = [token['token']]
                    elif ne_tag.startswith('I-'):
                        if not current_label == ne_tag[2:]:
                            pass
                            # warnings.warn(
                            #     f"Broken I-tag at token {token['token'], doc_id}. "
                            #     f"current label '{current_label}', got '{ne_tag}'",
                            #     category=RuntimeWarning
                            # )
                        else:
                            current_entity.append(token['token'])
                    else:  # Treat as B- if not BIO
                        if ne_tag != 'O':
                            pass
                            # warnings.warn(f"Invalid tag format: '{ne_tag}' at token {token['token'], doc_id}"
                            #               f"current label '{current_label}'",
                            #     category=RuntimeWarning)
                        if current_entity:
                            current_entities.append((' '.join(current_entity), current_label))
                            has_entities = True
                        current_entity = []
                        current_label = None

                # Add any remaining entity at sentence end
                if current_entity and current_label:
                    current_entities.append((' '.join(current_entity), current_label))
                    has_entities = True

                entity_annotations[doc_id] = current_entities

            return entity_annotations if has_entities else None

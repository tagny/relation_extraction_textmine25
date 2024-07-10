from collections.abc import Generator
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any, Dict, List, Tuple
import re

import pandas as pd

TARGET_COL = "relations"
INPUT_COLS = ["text", "entities"]
ID_COL = "id"

CHALLENGE_ID = "defi-text-mine-2025"
CHALLENGE_DIR = f"data/{CHALLENGE_ID}"
assert os.path.exists(CHALLENGE_DIR), f"path not found: {CHALLENGE_DIR=}"
train_raw_data_path = os.path.join(CHALLENGE_DIR, "raw", "train.csv")
test_raw_data_path = os.path.join(CHALLENGE_DIR, "raw", "test_01-07-2024.csv")
sample_submission_path = os.path.join(CHALLENGE_DIR, "raw", "sample_submission.csv")

assert os.path.exists(train_raw_data_path)
assert os.path.exists(test_raw_data_path)
assert os.path.exists(sample_submission_path)

EDA_DIR = os.path.join(CHALLENGE_DIR, "eda")
INTERIM_DIR = os.path.join(CHALLENGE_DIR, "interim")
MODELS_DIR = os.path.join(CHALLENGE_DIR, "models")
OUTPUT_DIR = os.path.join(CHALLENGE_DIR, "output")
for dir_path in [EDA_DIR, INTERIM_DIR, MODELS_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

submission_path = os.path.join(OUTPUT_DIR, "submission.csv")


# def clean_text(text: str) -> str:
#     RAW_STR_TO_CLEAN_STR = {
#         "\n": "",
#         "‘’": '"',
#         "’’": '"',
#         "”": '"',
#         "“": '"',
#         "’": '"',
#         " ": " ",
#     }
#     for raw_str, clean_str in RAW_STR_TO_CLEAN_STR.items():
#         text = re.sub(raw_str, clean_str, text)
#     return text.strip()


def load_labeled_raw_data() -> pd.DataFrame:
    return pd.read_csv(train_raw_data_path, index_col=ID_COL)


def load_test_raw_data() -> pd.DataFrame:
    return pd.read_csv(test_raw_data_path, index_col=ID_COL)


def clean_raw_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.assign(
        **{
            # don't clean text since it is recommended to give the raw text to BERT-base models
            # "text": lambda df: df.text.apply(clean_text),
            "entities": lambda df: df.entities.apply(json.loads),
            TARGET_COL: lambda df: (
                df[TARGET_COL].apply(json.loads)
                if TARGET_COL in df.columns
                else None  # pd.NA
            ),
        }
    )


def print_value_types(data: pd.DataFrame) -> None:
    for col in data.columns:
        value = data.iloc[0][col]
        col_type = type(value)
        if col_type is list:
            print(
                col,
                "[ ",
                (
                    type(value[0])
                    if type(value[0]) is list
                    else [type(e) for e in value[0]]
                ),
                " ]",
            )
        else:
            print(col, col_type)


def save_data(data: pd.DataFrame, csv_path: str, with_index: bool = True) -> None:
    """save data into a file at file_path

    Args:
        data (pd.DataFrame): data to save
        file_path (str): destination file
        with_index(bool): whether to save the index too
    """
    dest_dir_path = os.path.dirname(csv_path)
    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)
    data.to_csv(csv_path, index=with_index)


@dataclass
class TextToMultiLabelDataGenerator:
    first_entity_tag_name: str = field(default="e1")
    second_entity_tag_name: str = field(default="e2")
    text_col: str = field(default="text")
    text_index_col: str = field(default="text_index")

    def __post_init__(self):
        assert self.first_entity_tag_name != self.second_entity_tag_name

    def tag_entities(
        self, text: str, x: Dict[str, Any], y: Dict[str, Any]
    ) -> pd.DataFrame:
        """Mark the 2 given entities as the are the argument of a possible ordered
          relation to generate the 2 possible tagged texts where:

        1. x is the first entity of the relations, and y the second
        2. y is the first entity of the relations, and x the second

        Args:
            text (str): the text as stated in the original dataset
            x (Dict[str, Any]): an entity mentioned in the text as annotated
              in the original dataset e.g.
              {
                "id": 0,
                "mentions": [
                    {"value": "accident", "start": 70, "end": 78},
                    {"value": "accident de circulation", "start": 100, "end": 123}
                ]
              }
            y (Dict[str, Any]): an entity mentioned in the text as annotated
              in the original dataset; y is different from x.

        Returns:
            pd.DataFrame: with two columns with respectively the ids of the first and
              second entities in the marked text, and a last column with the marked text
        """
        logging.debug("starting")
        start2mentions = {
            m["start"]: m | {"id": e["id"], "type": e["type"]}
            for e in [x, y]
            for m in e["mentions"]
        }
        next_start = 0
        last_possible_end = len(text)
        entities_ids = (x["id"], y["id"])
        first_entity_id_to_tagged_text = {_id: "" for _id in entities_ids}

        for entity_start in sorted(list(start2mentions.keys())):
            if next_start >= len(text):
                break
            entity_id = start2mentions[entity_start]["id"]
            entity_type = start2mentions[entity_start]["type"]
            entity_end = start2mentions[entity_start]["end"]
            if next_start < entity_start:
                not_entity_span = text[next_start:entity_start]
                for first_entity_id in entities_ids:
                    first_entity_id_to_tagged_text[first_entity_id] += not_entity_span
            entity_span = text[entity_start:entity_end]
            for first_entity_id in entities_ids:
                tag = (
                    self.first_entity_tag_name
                    if entity_id == first_entity_id
                    else self.second_entity_tag_name
                )
                first_entity_id_to_tagged_text[
                    first_entity_id
                ] += "<{}><{}>{}</{}>".format(tag, entity_type, entity_span, tag)
                if first_entity_id == entity_id:
                    break
            next_start = entity_end
        # add the remaining text span if any remains
        if next_start < last_possible_end:
            not_entity_span = text[next_start:]
            for first_entity_id in entities_ids:
                first_entity_id_to_tagged_text[first_entity_id] += not_entity_span
        logging.debug("ending")
        rows = [[x["id"], y["id"], first_entity_id_to_tagged_text[x["id"]]]]
        if x["id"] != y["id"]:
            rows.append([y["id"], x["id"], first_entity_id_to_tagged_text[y["id"]]])
        return pd.DataFrame(
            rows,
            columns=[
                self.first_entity_tag_name,
                self.second_entity_tag_name,
                self.text_col,
            ],
        )

    def convert_relations_to_dataframe(
        self, text_index, relations: List[Tuple[int, str, int]]
    ) -> pd.DataFrame:
        """convert all the relations labeled in a text into a dataframe

        Args:
            relations (List[Tuple[int, str, int]]): relations of labeled in a text

        Returns:
            pd.DataFrame: resulting dataframe with 3 columns e1, e2, relations
              (i.e. the set of relations between e1 and e2) and a line per pair of
              entities e1 and e2 that are into relation.
        """
        # logging.info("starting")
        columns = [
            self.first_entity_tag_name,
            self.second_entity_tag_name,
            TARGET_COL,
        ]
        if not relations:
            return pd.DataFrame(columns=columns)
        entity_pair_to_relations = {}
        for e1, r, e2 in relations:
            entity_pair = (e1, e2)
            if entity_pair not in entity_pair_to_relations:
                entity_pair_to_relations[entity_pair] = set()
            entity_pair_to_relations[entity_pair].add(r)
        logging.debug("ending")
        return pd.DataFrame(
            [
                [e1, e2, list(e1_e2_relations)]
                for (e1, e2), e1_e2_relations in entity_pair_to_relations.items()
            ],
            columns=columns,
        )

    def convert(
        self,
        text_index: int,
        text: str,
        text_entities: List[Dict[str, Any]],
        text_relations: List[Tuple[int, str, int]],
    ) -> pd.DataFrame:
        """Convert an entry (a row) of the original dataset into a dataframe with:

        - a row correspond to a pair of the given entities; each possible ordered
            pair of entities has a row i.e. permutating the entities in an already
            processed pair, will enable the generation of another row.
        - a column for the text in which are tagged the mentions of the entities of
          the corresponding pair.
        - a column for the text index
        - a column for the first entity
        - a column for the second entity
        - a column for the list of relations of the first entity to the second

        Args:
            text_index (int): the text index in the original dataset
            text (str): the text as stated in the original dataset
            text_entities (List[Dict[str, Any]]): the entity mentioned in the text
              as given in the original dataset
            text_relations (List[Tuple[int, str, int]]): the labels given
              in the original dataset for the text

        Returns:
            pd.DataFrame: the resulting dataset
        """
        logging.debug("starting")
        entity_pair_to_relations_df = self.convert_relations_to_dataframe(
            text_index, text_relations
        )
        entity_pair_to_text_df = pd.DataFrame()
        # for idx in range(entity_pair_to_relations_df.shape[0]):
        #     row_idx_entity_pair_to_text_df = self.tag_entities(
        #         text,
        #         entity_pair_to_relations_df.e1.loc[idx],
        #         entity_pair_to_relations_df.e2.loc[idx],
        #     )
        #     logging.info(row_idx_entity_pair_to_text_df)
        #     entity_pair_to_text_df = pd.concat(
        #         [entity_pair_to_text_df, row_idx_entity_pair_to_text_df], axis=0
        #     )
        for i in range(len(text_entities)):
            for j in range(len(text_entities)):
                ij_entity_pair_to_text_df = self.tag_entities(
                    text, text_entities[i], text_entities[j]
                )
                entity_pair_to_text_df = pd.concat(
                    [entity_pair_to_text_df, ij_entity_pair_to_text_df], axis=0
                )
        new_columns = [self.text_index_col] + entity_pair_to_text_df.columns.to_list()
        # logging.info(f"{new_columns=}")
        entity_pair_to_text_df = entity_pair_to_text_df.assign(
            **{self.text_index_col: text_index}
        ).reset_index(drop=True)[new_columns]

        logging.debug("ending")
        # logging.info(row_idx_entity_pair_to_text_df)
        return entity_pair_to_text_df.join(
            entity_pair_to_relations_df.set_index(
                [
                    self.first_entity_tag_name,
                    self.second_entity_tag_name,
                ]
            ),
            on=[
                self.first_entity_tag_name,
                self.second_entity_tag_name,
            ],
        )

    def generate_row_multilabel_data(
        self, clean_df: pd.DataFrame, only_w_relation: bool = True
    ) -> Generator[pd.DataFrame]:
        """yields a dataframe per text with all the generated data

        Args:
            clean_df (pd.DataFrame): original dataset with entities
                and relations as list, and not str as original
            only_w_relation (bool): if True, just the tagged texts
                with non null relations (labels) will be generated

        Yields:
            Generator[pd.DataFrame]: generated dataset from a sentence annotation.
        """
        for text_index in clean_df.index:
            generated_df = TextToMultiLabelDataGenerator().convert(
                text_index,
                clean_df.loc[text_index].text,
                clean_df.loc[text_index].entities,
                clean_df.loc[text_index].relations,
            )
            if only_w_relation:
                generated_df = generated_df[~pd.isnull(generated_df[TARGET_COL])]
            yield generated_df

from pathlib import Path
import os
import requests
import tarfile
import shutil
import json
import csv
import sqlite3
import spacy
import pandas as pd
import time
import random
import math


def load_dataset():
    """Loads Spanish Dataset "ChatSubs" from source and extracts files inside

    Args:
        None

    Returns:
        None
    """
    # Dataset does not fit locally so download from source

    def safe_filter(tarinfo, extract_path):
        # Reject absolute paths or parent directory traversal
        if os.path.isabs(tarinfo.name) or ".." in tarinfo.name:
            return None
        return tarinfo  # keep original folder structure

    spanish_url = "https://zenodo.org/records/8220853/files/ChatSubs.tar.gz?download=1"
    spanish_filename = "ChatSubs.tar.gz"
    spanish_raw_path = "data/1-raw/spanish"

    print("Downloading Spanish dataset...")
    response = requests.get(spanish_url, stream=True)
    response.raise_for_status()  # Check if the download was successful

    os.makedirs(spanish_raw_path, exist_ok=True)

    with open(
        os.path.join(spanish_raw_path, spanish_filename), "wb", encoding="utf-8"
    ) as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Download completed. Extracting files...")
    with tarfile.open(os.path.join(spanish_raw_path, spanish_filename), "r:gz") as tar:
        tar.extractall(path="data/1-raw/spanish/", filter=safe_filter)

    print("Extraction completed. Cleaning up...")
    os.remove(os.path.join(spanish_raw_path, spanish_filename))
    for item in os.listdir(os.path.join(spanish_raw_path, "open_subtitles_es")):
        shutil.move(
            os.path.join(spanish_raw_path, "open_subtitles_es", item),
            os.path.join(spanish_raw_path, item),
        )
    shutil.rmtree(os.path.join(spanish_raw_path, "open_subtitles_es"))
    shutil.rmtree(os.path.join(spanish_raw_path, "open_subtitles_ca"))
    shutil.rmtree(os.path.join(spanish_raw_path, "open_subtitles_eu"))
    shutil.rmtree(os.path.join(spanish_raw_path, "open_subtitles_gl"))
    os.remove(os.path.join(spanish_raw_path, "export.txt"))
    print("Spanish dataset downloaded and extracted successfully.")


def extract_data_from_dataset(
    limit: int = float("inf"), limit_sampling_seed: int = None
):
    """Extracts data generated from `load_dataset` and combines all documents into a .csv file with columns: ID, dialogue.

    Args:
        limit (Optional, float(inf)): Limits how many docs are processed. Used for testing.

    Returns:
        None
    """
    spanish_root = "data/1-raw/spanish/"
    spanish_target = "data/1-raw/"

    doc_process_limit = limit

    # extract data from each file
    def extract_json_data(root_path):
        root = Path(root_path)
        # total_files_to_process = min(
        #     [len(list(root.rglob("*.jsonl"))), doc_process_limit]
        # )

        files = list(root.rglob("*.jsonl"))

        if len(files) > doc_process_limit:
            if limit_sampling_seed is not None:
                random.seed(limit_sampling_seed)
            files = random.sample(files, doc_process_limit)

        total_files_to_process = len(files)
        for doc_i, doc in enumerate(files):
            if doc_i >= doc_process_limit:
                break

            print(
                f"({round((((doc_i+1)/total_files_to_process)) * 100, 2)}%) Processing Document {doc_i+1}/{total_files_to_process}",
                doc,
            )

            with doc.open("r", encoding="utf-8") as f:
                data = json.load(f)
                yield data

    def doc_preprocess(doc):
        # here, doc is a dictionary: doc["dialogues"] extract dialogues and is a list of strings (each string is a dialogue)
        for dialogue in doc["dialogues"]:
            yield dialogue

    # write all conversations to one document
    with open(
        os.path.join(spanish_target, "spanish.csv"), "w", newline="", encoding="utf-8"
    ) as csvfile:
        fieldnames = ["ID", "dialogue"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        idx_counter = 0
        for item in extract_json_data(spanish_root):
            for extracted_dialogue in doc_preprocess(item):
                clean_dialogue = extracted_dialogue.replace("\n", "\\n")
                writer.writerows([{"ID": idx_counter, "dialogue": clean_dialogue}])
                idx_counter += 1

    with open(
        os.path.join(spanish_target, "spanish.csv"), newline="", encoding="utf-8"
    ) as f:
        reader = csv.reader(f)
        total_row_count = sum(1 for _ in reader)
        print(f"Wrote a total of {total_row_count} dialogues into spanish.csv.")


def grammar_preprocessing(
    nlp_size: str = "large",
    cores_to_use: int = 1,
    import_chunk_size: int = 10000,
    processing_chunk_size: int = 1000,
):
    """Preprocesses .csv file generated from `extract_data_from_dataset` using spaCy.
    It tokenizes, lemmatizes, and PoS tags all words in the dataset and writes data to a sqlite3 database for memory optimization.
    The Sqlite3 database has 3 columns: lemma, pos, and frequency. (lemma, pos) is a key to enable creating unique entries of lemma+PoS
    combinations while iterating frequency counter if found.

    Args:
        nlp_size: "large" | "small". Defines which sized model to use in spaCy data preprocessing steps
        cores_to_use: Specifies how many cores to use for spaCy processing
        import_chunk_size: Specifies how large each chunk of the dataset will be loaded in at one time
        processing_chunk_size: Specifies how large each chunk of the dataset that will be processed in spaCy

    Returns:
        DataFrame: Pandas DataFrame reference to sqlite3 database
    """
    # extract each of the rows from the spanish dataset, lemmatize and PoS tag it,
    # then write that lemma into a csv with lemma as one column, PoS tag as another
    # column and frequency as the third. Check if a combination of the lemma + PoS exist,
    # +1 on the frequency, otherwise create a new entry

    root_dir = "data/1-raw/"
    target_dir = "data/2-extracted/"
    db_dir = os.path.join(target_dir, "spanish.db")

    if nlp_size == "large":

        try:
            nlp = spacy.load("es_dep_news_trf")
        except:
            from spacy.cli import download

            download("es_dep_news_trf")

            nlp = spacy.load("es_dep_news_trf")

    elif nlp_size == "small":
        try:
            nlp = spacy.load("es_core_news_sm")
        except:
            from spacy.cli import download

            download("es_core_news_sm")

            nlp = spacy.load("es_core_news_sm")
    else:
        raise ValueError(f'nlp_size must be "large" or "small". Got {nlp_size}')

    if os.path.exists(db_dir):
        raise OSError(
            f'Found existing spanish database. Due to very long processing time function is not allowed to execute in case of overwriting. Please delete file "{db_dir}" to proceed.'
        )

        # os.remove(db_dir)

    con = sqlite3.connect(db_dir)
    cursor = con.cursor()

    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA mmap_size = 0;")
    # cursor.execute("PRAGMA temp_store = MEMORY;")
    cursor.execute("PRAGMA cache_size = -12000000;")
    cursor.execute("PRAGMA temp_store = FILE;")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS word_freq (
        lemma TEXT,
        pos TEXT,
        frequency INTEGER,
        PRIMARY KEY (lemma, pos)
    )
    """
    )

    total_row_count = 0

    with open(os.path.join(root_dir, "spanish.csv"), newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        total_row_count = sum(1 for _ in reader)

    con.commit()

    total_timer = time.time()

    for batch_i, batch in enumerate(
        pd.read_csv(os.path.join(root_dir, "spanish.csv"), chunksize=import_chunk_size)
    ):
        # load in chunks as not to explode my PC
        timer_start = time.time()
        print(
            f"Processing Batch {batch_i + 1}/{math.ceil(total_row_count/import_chunk_size)}"
        )
        dialogues = batch["dialogue"]

        # for dialogue_i, dialogue in enumerate(batch["dialogue"]):
        # print(f"Processing Dialogue {dialogue_i + 1} in batch {batch_i + 1}")
        word_freq_dict = {}

        for dialogue in dialogues:
            dialogue = dialogue.replace("\\n", " ")

        processed_dialogue = nlp.pipe(
            dialogues, batch_size=processing_chunk_size, n_process=cores_to_use
        )
        for doc in processed_dialogue:
            for token in doc:
                if not token.is_punct and not token.is_space:
                    key = (token.lemma_.lower(), token.pos_)
                    word_freq_dict[key] = word_freq_dict.get(key, 0) + 1

        records = [
            (lemma, pos, freq, freq) for (lemma, pos), freq in word_freq_dict.items()
        ]

        cursor.executemany(
            """
                INSERT INTO word_freq (lemma, pos, frequency)
                VALUES (?, ?, ?)
                ON CONFLICT (lemma, pos)
                DO UPDATE SET frequency = frequency + ?
            """,
            records,
        )

        # for (lemma, pos), freq in word_freq_dict.items():
        #     # write to DB, either new entry if unique or +1 otherwise
        #     cursor.execute(
        #         """
        #         INSERT INTO word_freq (lemma, pos, frequency)
        #         VALUES (?, ?, ?)
        #         ON CONFLICT (lemma, pos)
        #         DO UPDATE SET frequency = frequency + ?
        #     """,
        #         (lemma, pos, freq, freq),
        #     )

        con.commit()

        # if (batch_i + 1) % 50 == 0:
        # cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")

        timer_end = time.time()
        total_time = timer_end - timer_start
        print(
            f"[{round((time.time() - total_timer)/60)} minutes since start] Finished Batch {batch_i + 1} in {round(total_time, 2)} seconds ({round(total_time/import_chunk_size, 6)}/item)"
        )

    print(
        f"Finished! Total time in hours: {round(((time.time() - total_timer) / 60) / 60, 2)}"
    )

    df = pd.read_sql_query(
        "SELECT * FROM word_freq ORDER BY frequency DESC LIMIT 10", con
    )

    con.close()
    return df


def finalize_dataset(limit=5000):
    """Converts list of lemmas to .csv format, ranks by frequency and limits dataset.

    Args:
        limit: Limits final dataset to top N frequent lemmas.

    Returns:
        DataFrame: Final Dataset
    """

    if type(limit) is not int:
        raise TypeError(f"limit not of type int. Got {type(limit)}.")

    root_dir = "data/2-extracted/"
    target_dir = "data/3-final/"
    db_dir = os.path.join(root_dir, "spanish.db")

    con = sqlite3.connect(db_dir)
    cursor = con.cursor()

    df = pd.read_sql_query(
        "SELECT * FROM word_freq ORDER BY frequency DESC LIMIT ?", con, params=(limit,)
    )

    df.to_csv(
        os.path.join(target_dir, f"spanish{limit}.csv"), index=True, index_label="ID"
    )

    return df

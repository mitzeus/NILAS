from src.preprocessing.spanish import (
    grammar_preprocessing as spanish_grammar_preprocessing,
)

if __name__ == "__main__":

    spanish_df = spanish_grammar_preprocessing(
        nlp_size="small",
        cores_to_use=6,
        import_chunk_size=120000,
        processing_chunk_size=4000,
    )  # Does preprocessing: tokenization, lemmatization, PoS tagging

    print(spanish_df.head(10))

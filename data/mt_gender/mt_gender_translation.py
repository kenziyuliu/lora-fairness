""" Usage:
    <file-name> --in=IN_FILE --src=SOURCE_LANGUAGE --tgt=TARGET_LANGUAGE --out=OUT_FILE --trans=TRANS_SERVICE [--debug]
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
from google.cloud import translate_v2 as translate
import html

GOOGLE_TRANSLATE_CLIENT = translate.Client()

def google_translate(sents, target_language, source_language = None):
    """
    Run google translate on a batch of sentences.
    """
    if source_language is None:
        # Detected input language
        # TODO: it's possible that this is redundant and not needed
        trans = GOOGLE_TRANSLATE_CLIENT.translate(sents,
                                                  target_language = target_language)
    else:
        trans = GOOGLE_TRANSLATE_CLIENT.translate(sents,
                                                  source_language = source_language,
                                                  target_language = target_language)

    for out_dict in trans:
        out_dict["translatedText"] = html.unescape(out_dict["translatedText"])
    return trans

BATCH_SIZE = 50 # Up to 128 should be fine?

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def batch_translate(trans_function, lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    translations_dicts = []
    for chunk in tqdm(list(chunks(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        for out_dict in trans_function(chunk, tgt_lang, src_lang):
            translations_dicts.append(out_dict)
    return translations_dicts

TRANSLATION_SERVICE = {
    "google": google_translate,
}

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    trans_service = args["--trans"]
    inp_fn = args["--in"]
    src_lang = args["--src"]
    tgt_lang = args["--tgt"]
    out_fn = args["--out"]
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Figure out the translation service to use
    assert trans_service in TRANSLATION_SERVICE, f"{trans_service} is not supported"
    trans_function = TRANSLATION_SERVICE[trans_service]

    # [2] to get the actual sentences. Other fields are not needed.
    lines = [line.split("\t")[2] for line in open(inp_fn, encoding = "utf8")]
    labels = [line.split("\t")[0] for line in open(inp_fn, encoding = "utf8")]
    
    if "general" in inp_fn:
        for i in range(0, len(lines), 2):
            label_first = labels[i]
            if i < 1584:
                # anti
                if label_first == "male":
                    # swap the line
                    lines[i], lines[i+1] = lines[i+1], lines[i]
            else:
                # pro and triplet with neutral gender pronoun
                if label_first == "female":
                    # swap the line
                    lines[i], lines[i+1] = lines[i+1], lines[i]
    elif "pro" in inp_fn:
        for i in range(0, len(lines), 2):
            label_first = labels[i]
            if label_first == "female":
                # swap the line
                lines[i], lines[i+1] = lines[i+1], lines[i]
    elif "anti" in inp_fn:
        for i in range(0, len(lines), 2):
            label_first = labels[i]
            if label_first == "male":
                # swap the line
                lines[i], lines[i+1] = lines[i+1], lines[i]
    
    out_dicts = batch_translate(trans_function, lines, tgt_lang, src_lang)
    with open(out_fn, "w", encoding = "utf8") as fout:
        for out_dict in out_dicts:
            fout.write("{} ||| {}\n".format(out_dict["input"],
                                            out_dict["translatedText"]))

    logging.info("DONE")
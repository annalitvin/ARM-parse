#!/usr/bin/python3
import amrlib
import re

from tqdm import tqdm
from pypdf import PdfReader
from amrlib.utils.logging import setup_logging, WARN

def get_sentences(text):
    return re.split(r"(?<=[^A-Z].[.!?])(?:\s|\n)+(?=[A-Z])", text)

def extract_sentences_from_pdf(file_path, page_number):
    extracted_sentences = []
    with PdfReader(file_path) as reader:
        try:
            page = reader.pages[page_number]
        except IndexError:
            raise IndexError(f"This page does not exist. The value must be up to {len(reader.pages)}")
        extracted_sentences.extend(get_sentences(page.extract_text())[1:-1])
        return extracted_sentences

def convert_pdf_to_amr_graph(file_path, stog_model, page_number=1):
    sentences = extract_sentences_from_pdf(file_path=file_path, page_number=page_number)
    sentences_graphs = tqdm(stog_model.parse_sents(sentences, add_metadata=True))
    for graph in sentences_graphs:
        yield graph


if __name__ == '__main__':

    setup_logging(logfname='logs/test_model_parse_xfm.log', level=WARN)

    num_beams = 4
    batch_size = 16
    model = amrlib.load_stog_model(disable_progress=False, batch_size=batch_size, num_beams=num_beams)

    file_name = "1706.01678v3.pdf"
    page_number = 3
    page_graph_file = f"{file_name}-page-{page_number}.txt"

    page_graph = convert_pdf_to_amr_graph(file_path=file_name, stog_model=model, page_number=page_number)
    with open(page_graph_file, "w", encoding="utf-8") as text_file:
        text_file.write(str(list(page_graph)) + '\n\n')

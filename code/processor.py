import numpy as np
import PyPDF2
import spacy
nlp = spacy.load("en_core_web_sm")
import io
from urllib.request import Request, urlopen
from newspaper import Article
from PyPDF2 import PdfReader
import docx


class Processor:
    '''
    The Processor class reads text from a source and splits the text into chunks.

    Attributes:
    - source(str): source of the target document, supporting url, txt, docx and pdf.

    Methods:
    - extract_text: get the text from the source
    - get_chunks: split the text into chunks
    '''

    def __init__(self, source):
        '''
        Initialize a Processor instance

        Parameter:
        - source: source of the target document, supporting url, txt, docx and pdf.
        '''
        self.source = source

    def extract_text(self):
        '''
        Get the text from the source
        '''
        if self.source.startswith('https:') and self.source.endswith('.pdf'):
            document_from_source = urlopen(Request(self.source)).read()
            document_in_memory = io.BytesIO(document_from_source)
            document = PdfReader(document_in_memory)
            pages = len(document.pages)
            text = ''
            for p in range(pages):
                page = document.pages[p].extract_text()
                text += page

        elif self.source.startswith('https:') or self.source.startswith(
                'www:'):
            document = Article(self.source, language="en")
            document.download()
            document.parse()
            document.nlp()
            text = document.title + '.\n' + document.text

        elif self.source.endswith('.txt'):
            with open(self.source, 'r') as f:
                text = f.read()

        elif self.source.endswith('.pdf'):
            with open(self.source, 'rb') as f:
                document = PdfReader(f)

                pages = len(document.pages)
                text = ''
                for p in range(pages):
                    page = document.pages[p].extract_text()
                    text += page

        elif self.source.endswith('.docx'):
            with open(self.source, "rb") as f:
                document = docx.Document(f)
                text = '\n'.join(
                    [paragraph.text for paragraph in document.paragraphs])

        return text

    def get_chunks(self, by_tokens=False, num_tokens=100):
        '''
        Split the text into chunks

        Parameters:
        by_tokens(boolean): if true, the text is split into strings containing a certain number of tokens; otherwise, the text is split into sentences
        num_tokens(int): the number of tokens in each string if by_tokens=True
        '''
        text = self.extract_text()
        doc = nlp(text)

        if by_tokens:
            chunks = []
            start = 0
            while start <= len(doc) - num_tokens:
                end = start + num_tokens
                tokens = doc[start:end]
                chunk = ' '.join([token.text for token in tokens])
                chunks.append(chunk)
                start = end
            if start < len(doc):
                residual = ' '.join([token.text for token in doc[start:]])
                chunks.append(residual)

        # the default is splitting by sentences
        else:
            chunks = [sent.text.replace('\n', ' ') for sent in doc.sents]

        return np.array(list(set(chunks)))

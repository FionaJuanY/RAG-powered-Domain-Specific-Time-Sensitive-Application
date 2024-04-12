import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")
from sentence_transformers import SentenceTransformer


class Retriever:
    '''
    The Retriever class retrieves the top-k chunks that are the most relevant to a query

    Attributes:
    - encoder(str): the name of a specific encoder under Sentence Transformers
    - chunks(arr): an array of chunks that a document is split into

    Methods:
    - chunks_embedding: convert the chunks into embeddings
    - retrieve_context: retrieve the context from the chunks according to a query
    '''

    def __init__(self, encoder, chunks):
        '''
        Initialize a Retriever instancce

        Attributes:
        - encoder(str): the name of the encoder
        - chunks(arr): an array of chunks that a document is split into
        '''
        self.encoder = SentenceTransformer(encoder)
        self.chunks = chunks

    def chunks_embedding(self):
        '''
        Convert the chunks into embeddings
        '''
        chunks_embeddings = self.encoder.encode(self.chunks)

        return chunks_embeddings

    def retrieve_context(self, chunks_embeddings, query, k=1, enhanced=False,
                         min_length=256):
        '''
        Retrieve the context from the chunks according to a query

        Parameters:
        - chunks_embedding(arr): an array of embeddings of the chunks
        - query(str): a question from a user
        - k(int): the number of the most relevant chunks to be retrieved
        - enhenced(boolean): if true, when the retrieved context is shorter than minimum length, the next most relevant chunks will be added to the context
        - min_length(int): the minimum length of context
        '''
        query_embeddings = self.encoder.encode(query)

        similarities = np.dot(chunks_embeddings, query_embeddings) / (
                    np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(
                query_embeddings))

        sorted_indices = np.argsort(similarities)[::-1]

        top_k_indices = sorted_indices[:k]

        top_k_chunks = self.chunks[top_k_indices]

        context = ' '.join(top_k_chunks)

        if enhanced:
            # check the length of context
            while len(context) < min_length:
                extra_idx = sorted_indices[k]
                extra_chunk = self.chunks[extra_idx]
                context += extra_chunk
                k += 1

        return context
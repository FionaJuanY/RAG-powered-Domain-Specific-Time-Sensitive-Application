
**RAG-powered Domain-Specific/Time-Sensitive Q&A Application**

**Author:** Juan Yu

**Abstract:** Given their exceptional text generating capability, LLM-backed Q&A systems are gaining significant popularity. However, they are limited by their training data which may not be domain-specific or up-to-date. Retrieval-Augmented Generation (RAG) provides a solution. Based on RAG, this project builds an application that is able to answer domain-specific/time-sensitive questions. Moreover, a dataset containing question-context-answer triplets for Apple’s Annual Earnings Report 2022 is used to evaluate the application. The large language model (LLM) which is used to generate answers improve its performance significantly after RAG is included.

**Introduction** 

Retrieval-Augmented Generation (RAG) was proposed by Meta in 2020. It works by allowing input of user-defined knowledge bases, retrieving contextual information that is relevant to a specific question from knowledge bases and then generating a desirable answer to the question based on the retrieved contextual information. This application is built based on this RAG framework and aims to answer domain-specific/time-sensitive questions.

**Methods**

The architecture of the application consists of three main components, a processor, a retriever and a generator. The processor breaks the documents in the knowledge bases into text chunks, and is built with spaCy in this application. From the text chunks, the retriever retrieves the top-k chunks that are the most relevant to a user’s question. Sentence-transformers/all-MiniLM-L6-v2, which was the most downloaded model among sentence-transformers when this project was being implemented, along with cosine similarity is used to build the retriever. Given the top-k relevant chunks and the question from the user, the generator generates an answer to the question. Llama-2-7B-Chat-GGUF, which can be run on CPU, is used as the generator in this application.

**Usage** 

**Step 0:** Download the code folder

**- Runing from source**

**Step 1:** Install necessary packages
pip install -r requirements.txt

**Step 2:** Import necessary files via:
from your_path/code/processor import Processor
from your_path/code/retriever import Retriever
from your_path/code/generator import Generator

**Step 3:** Build a processor and split the source document into chunks. The source document can be in the format of docx, pdf or txt or can be an url. 

processor = Processor('you source document')

Split the document by sentence (default)
chunks = processor.get_chunks()

Split the document by n-tokens (default number of tokens is 100)
chunks = processor.get_chunks(by_tokens=True, num_tokens=100)

**Step 4:** Build a retriever using a sentence-transformers/all-MiniLM-L6-v2 and retrieve top-k text chunks that are the most relevant to a specific question.

encoder = 'sentence-transformers/all-MiniLM-L6-v2'
retriever = Retriever(encoder, chunks)

chunks_embeddings = retriever.chunks_embedding()

query = "enter your question here"

context = retriever.retrieve_context(chunks_embeddings, query, k=1) # default value of k is 1

if the document is split by sentence, min length can be set on the context retrieved: 
context = retriever.retrieve_context(sentences_embeddings, query, k=1, enhanced=True, min_length=256) # default values of min_length is 256.

**Step 5:** Build a generator using TheBloke/Llama-2-7B-Chat-GGUF, and generate an answer to the question. 

model='TheBloke/Llama-2-7B-Chat-GGUF'
model_file = 'llama-2-7b-chat.Q4_K_M.gguf'
model_type='llama'
generator = Generator(model=model, model_file=model_file, model_type=model_type)

answer = generator.generate_answer(context, query)

**- Runing the application**

**Step 1:** Create a conda environment and activate the environment
conda create -n your_ environment
conda activate your_ environment

**Step 2:** Install necessary packages
conda install pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

**Step 3:** execute “streamlit run scripts.py”

**Evaluation** 

The evaluation was based on question-context-answer triplets for Apple’s Annual Earnings Report 2022. (https://huggingface.co/datasets/lighthouzai/finqabench) Different context length strategies were applied and results were compared.

Specifically, generated answers were evaluation based on the semantic similarity to the ground truth, and the results are as follows:

|                                               | Cosine similarity |
| :---------------------------------------------|:-----------------:|
| Baseline - no context                         |         0.57      |
| RAG one sentence context                      |         0.63      |
| RAG Multi-sentence context (max length of 256)|         0.66      |
| RAG 100-tokens context                        |         0.67      |

**Discussion** 

In general, this application can have significantly better results than no-context generation. It could be used in a wide range of areas where specific knowledge bases are needed. For example, listed companies can use it to answer shareholders’ questions about their earnings reports, while manufacturers can use it to answer their clients’ questions about their products. 

However, as we can see from the evaluation, context with proper length can enable the application to have even better performance. Therefore, future work will be focused on how to automatically decide the best length of context. 

**Conclusion** 

This project builds an application based on the RAG framework, which enables large language models to generate answers to domain-specific/time-sensitive questions. It is achieved by allowing users to input extra knowledge bases, retrieving information from the knowledge bases that is relevant to a specific question and feeding both the retrieved information and the question to a generator, which is a LLM in this application. In this application, local files in the format of pdf, docx, and txt as well as urls are supported as extra knowledge bases. 

Moreover, multiple context length strategies were evaluated, which revealed the importance of the proper length of context and pointed to a direction for the future work. 

**References:**

[1.] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33, 9459-9474. https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf

[2.] Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. In EMNLP 2020 - 2020 Conference on Empirical Methods in Natural Language Processing, Proceedings of the Conference (pp. 6769-6781). (EMNLP 2020 - 2020 Conference on Empirical Methods in Natural Language Processing, Proceedings of the Conference). Association for Computational Linguistics (ACL). https://aclanthology.org/2020.emnlp-main.550.pdf

[3.] Wang, Z., Araki, J., Jiang, Z., Parvez, M. R., & Neubig, G. (2023). Learning to filter context for retrieval-augmented generation. arXiv preprint arXiv:2311.08377. https://arxiv.org/pdf/2311.08377.pdf

[4.] Lighthouz AI. (n.d.). FinQABench. Hugging Face. https://huggingface.co/datasets/lighthouzai/finqabench

[5.] Briggs, J. (2023, July 29). Better Llama 2 with Retrieval Augmented Generation (RAG) [Video]. YouTube. https://www.youtube.com/watch?v=ypzmPwLH_Q4

[6.] Sentence Transformers. (n.d.). Sentence Transformers: all-MiniLM-L6-v2. Hugging Face.https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[7.] Jobbins, T. (n.d.). TheBloke/Llama-2-7B-Chat-GGUF. Hugging Face. https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

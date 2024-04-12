from processor import Processor
from retriever import Retriever
from generator import Generator
import streamlit as st

st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">', unsafe_allow_html=True)


# st.title('Retrieval Augmented Generation')
st.markdown("<br>", unsafe_allow_html=True)

assistant = 'Please provide document source.'
letters = 'RAG:    '
st.markdown('''
    <style>
        .custom-icon {
            color: blue;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
''', unsafe_allow_html=True)
st.write(f'<i class="custom-icon">{letters}</i>{assistant}', unsafe_allow_html=True)

st.markdown('''
    <style>
        .stTextInput label {
            color: blue; 
            font-size: 22px; 
            font-weight: bold; 
        }
    </style>
''', unsafe_allow_html=True)

# Display the text input with the styled label
doc_name = st.text_input('USER:', key='user_input')

if doc_name:
    proc = Processor(doc_name)
    # st.write('Document processing...It could take a while')
    chunks = proc.get_chunks()

    retr = Retriever('all-MiniLM-L6-v2', chunks)
    chunks_embeddings = retr.chunks_embedding()

    idx = 1
    assistant = 'What do you want to know about this document?'
    st.write(f'<i class="custom-icon">{letters}</i>{assistant}', unsafe_allow_html=True)
    query = st.text_input('USER:', key=idx)
    while query:
        context = retr.retrieve_context(chunks_embeddings, query, k=1)

        gen = Generator(model='TheBloke/Llama-2-7B-Chat-GGUF', model_file='llama-2-7b-chat.Q4_K_M.gguf', model_type='llama')

        answer = gen.generate_answer(context, query)

        st.write(f'<i class="custom-icon">{letters}</i>{answer}',
                 unsafe_allow_html=True)
        # st.write(answer)

        idx += 1
        assistant = 'What do you want to know about this document?'
        st.write(f'<i class="custom-icon">{letters}</i>{assistant}',
                 unsafe_allow_html=True)
        query = st.text_input('USER:', key=idx)
        # query = st.text_input('What do you want to do know about this document?', key=idx)
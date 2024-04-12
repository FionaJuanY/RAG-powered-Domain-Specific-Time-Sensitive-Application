import spacy
nlp = spacy.load("en_core_web_sm")
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain

class Generator:
    '''
    The Generator class takes in a query and relevant context and generates an answer

    Attributes:
    - model(str): name of model
    - model_file(str): name of model file
    - model_type(str): model type

    Method:
    - generate_answer: generate answers based on query and context
    '''

    def __init__(self, model, model_file, model_type):
        '''
        Initialize a Generator instance
        - model(str): name of model
        - model_file(str): name of model file
        - model_type(str): model type
        '''
        llm = CTransformers(model=model, model_file=model_file,
                            model_type=model_type)

        template = '''
                    Context: {context}
                    Question: {question}
                    Answer:
                    '''
        prompt = PromptTemplate(template=template,
                                input_variables=['context', 'question'])
        self.llm_chain = LLMChain(prompt=prompt, llm=llm)

    def generate_answer(self, context, question):
        '''
        Generate answers based on query and context

        Parameters:
        - context(str): context provided to the generator
        - question(str): question from a user
        '''
        answer = self.llm_chain.run({'context': context, 'question': question})
        return answer
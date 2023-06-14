
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import streamlit as st 


with st.sidebar :
    st.image("icon.png")
    model_idh = st.text_input("Entre your HuggingFace model ID : " , value='google/flan-t5-large' )


with st.spinner("Downloading and setting up the model ...") : 
    try : 

        #model_id = 'google/flan-t5-large' 
        model_id = model_idh
        # You can use a larger model if you have Collab Pro (If you change the LLM model , you must also change HuggingFaceEmbedding function)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        memory = ConversationBufferMemory()
        pipe = pipeline(
            "text2text-generation",
            model=model, 
            tokenizer=tokenizer
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)
        frst_conversation = ConversationChain(
            llm=local_llm, 
            verbose=False, 
            memory=memory
        )
        # When added to an agent, the memory object can save pertinent information from conversations or used tools
        memory.save_context({"input": "My favorite food is pizza"}, {"output": "thats good to know"})
        memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
        memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"}) # 

    except : 
        pass



_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)

with st.spinner("generate embeddings for text input ..") :
    if  model_idh : 
        embeddings_func = HuggingFaceEmbeddings(model_name=model_idh)
text = frst_conversation.memory.buffer.split("\n")
vectordb = Chroma.from_texts(texts=text, 
                                 embedding=embeddings_func,
                                 persist_directory="db")

retriever = vectordb.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=retriever)
new_conversation = ConversationChain(
    llm=local_llm, 
    verbose=False, 
    prompt=PROMPT,
    memory=memory , 
)

try : 
    inputs = st.text_input("Say something : ")
    result = new_conversation.predict(input=inputs)
    st.write(result)
    save = vectordb.add_texts([result])
    if save : 
        st.info("The memory has been updated and stored in a db file")
except : 
    pass


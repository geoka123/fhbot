import chromadb
import logging
import sys

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Document)
from llama_index.core import StorageContext, ServiceContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import logging
import sys

def actual_llm_init():
    global model
    global tokenizer

    model_id="gpt2"
    model=AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer=AutoTokenizer.from_pretrained(model_id)

def llm_init():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=1.0,
        max_new_tokens=2048,
        huggingfacehub_api_token="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"
    )
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Settings.llm = llm
    Settings.embed_model = embed_model

def index_init():
    global index

    parser = LlamaParse(
        api_key=('llx-MHkv4e22IpbbRBvnKRaPZTWXbuVQpozhfmypYJrpSTyEBjcJ'),
        parsing_instruction = f"""You are an agent that takes an excel file that contains the application data of startups to an accelerator program. Each row corresponds to an application. Give detailed answers based on these applications.""",
        result_type="markdown"
    )

    # read documents in docs directory
    # the directory contains data set related to red team and blue team cyber security strategy
    file_extractor = {".xlsx": parser}
    documents = SimpleDirectoryReader(input_files=['/home/ec2-user/fhbot/media/data_sources/Application_Database.xlsx'], file_extractor=file_extractor).load_data()

    logging.info("index creating with `%d` documents", len(documents))

    # create large document with documents for better text balancing
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    # sentece window node parser
    # window_size = 3, the resulting window will be three sentences long
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # create qdrant client
    qdrant_client = QdrantClient(
        url="https://bff3be45-6a0e-4931-83be-d93c2810171d.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="Pyq_lqp0G9xhTIWiwidSv3evxN98jix72qUjBnFPP8VNKiClwKsTIw",
        prefer_grpc=True
    )

    # delete collection if exists,
    # in production application, the collection needs to be handle without deleting
    qdrant_client.delete_collection("fh_data")

    # qdrant vector store with enabling hybrid search
    vector_store = QdrantVectorStore(
        collection_name="fh_data",
        client=qdrant_client,
        enable_hybrid=True,
        batch_size=20
    )

    # storage context and service context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        node_parser=node_parser,
    )

    # initialize vector store index with qdrant
    index = VectorStoreIndex.from_documents(
        [document],
        service_context=service_context,
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )

def query_engine_init():
    global query_engine
    global index

    # after retrieval, we need to replace the sentence with the entire window from the metadata by defining a
    # MetadataReplacementPostProcessor and using it in the list of node_postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    # re-ranker with BAAI/bge-reranker-base model
    rerank = SentenceTransformerRerank(
        top_n=2,
        model="BAAI/bge-reranker-base"
    )

    # similarity_top_k configure the retriever to return the top 3 most similar documents, the default value of similarity_top_k is 2
    # use meta data post processor and re-ranker as post processors
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        node_postprocessors=[postproc, rerank],
    )

def chat(input_question):
    try:
        pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=100)
        hf=HuggingFacePipeline(pipeline=pipe)

        gpu_llm = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            device_map="auto",  # replace with device_map="auto" to use the accelerate library.
            pipeline_kwargs={"max_new_tokens": 100},
        )

        # llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        # input_data = {"question": question}
        
        template = """Question: {question}

        Answer:"""
        prompt = PromptTemplate.from_template(template)

        chain=prompt|gpu_llm

        return chain.invoke(input_question)
    except Exception as e:
        raise ValueError(f"Smth went wrong {e}")
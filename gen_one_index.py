import os
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain_community.embeddings import OpenAIEmbeddings
from llama_index import StorageContext, load_index_from_storage, set_global_service_context

# Retrieve API keys and endpoint from environment variables
api_key = os.getenv('AZURE_OPENAI_API_KEY')
api_version = os.getenv('OPENAI_API_VERSION')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# Check if all necessary credentials are provided
if not all([api_key, api_version, azure_endpoint]):
    raise ValueError("API key, API version, or endpoint is missing. Please set the environment variables correctly.")

# Initialize AzureOpenAI language model instance
llm = AzureOpenAI(
    model="gpt-35-turbo",  # Adjust the model name as per actual availability
    deployment_name= "gpt-35-turbo-tdu",  # Adjust the model name as per actual availability
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# Initialize AzureOpenAIEmbedding instance
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",  # Adjust the model name as per actual availability
    deployment_name= "text-embedding-ada-002-tdu",  # Adjust the model name as per actual availability
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# Create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

# Set the global service context
set_global_service_context(service_context)

# Load or create the vector store index
index_file = 'index_dir'
if os.path.exists(index_file) and os.path.getsize(index_file) > 0:
    print("Loading index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=index_file)
    index = load_index_from_storage(storage_context)
    #with open("index.json", 'r') as f:
    #    index_data = json.load(f) 
    #    index = VectorStoreIndex(data=index_data)
else:
    print("Index file does not exist, so create it...")
    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(index_file)
    #index_data = index.serialize()
    #with open(index_file, 'w') as f:
    #    json.dump(index_data,f)

# Search for a document using the query method of the VectorStoreIndex
# search_results = index.search("effect of body chemistry on exercise?")
query_engine = index.as_query_engine()
search_results = query_engine.query(
    #"effect of body chemistry on exercise?"
    "Do you know the rooting of the word sport?"
)
print(search_results)

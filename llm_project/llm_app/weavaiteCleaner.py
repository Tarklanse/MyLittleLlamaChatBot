import json
import weaviate
import os

# Load JSON file with active mappings
with open("D:/workspace/LLM loading and using code/llmHoster/vectorMapper/VectorMapper.json", 'r') as f:
    mappings = json.load(f)

# Extract the active Weaviate IDs from the mappings
active_vector_ids = set(mappings.values())

# Initialize Weaviate client
client = weaviate.connect_to_local()
# Fetch all vector stores currently in Weaviate
# Assuming all vectors are stored in a class, for example `LangChain`
all_vector_ids = set()
try:
    list_of_collection=client.collections.list_all()
    for this_collection in list_of_collection:
        print(str(this_collection))
        if str(this_collection) not in active_vector_ids:
            client.collections.delete(this_collection)

except Exception as e:
    print("Error retrieving vector store IDs:", e)

client.close()
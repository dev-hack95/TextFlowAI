import os
from pinecone import Pinecone, PodSpec
pc = Pinecone(api_key=os.environ["api_key"])
print(pc.list_indexes())
pc.create_index(
  name="testindex2",
  dimension=384,
  metric="cosine",
  spec=PodSpec(
    environment=os.environ["environment"],
    pods=1
  )
)
print(pc.list_indexes())
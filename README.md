This repository contains the code for the paper:
# Schema-based inference for query expansion and completion over knowledge graphs

### Keywords
Knowledge graphs, query answering systems, graph databases, large language models, natural language processing


## Abstract
Graph databases are powerful tools for representing complex, interconnected knowledge, but their structure and semantics pose significant challenges for natural language interaction. In this work, we present a guided approach based on the Retrieval-Augmented Generation (RAG) schema to assist large language models in generating queries over graph-based databases. By incorporating semantically relevant paths into the prompt, our method reduces the reasoning complexity that the model must handle, enabling more accurate query generation. We evaluate the performance of different language models under two schema interaction paradigms—full schema exploration and path-specific guidance—across queries related to risk assessment in passenger and flight data. Our preliminary results show that path-specific guidance significantly improves model performance, particularly for smaller models. That semantic pre-filtering of the graph structure enhances the model’s ability to focus on relevant information.


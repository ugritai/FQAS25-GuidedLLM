import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer, util

# Settings
print("Loading models and setting up schema...\n")

embedder = SentenceTransformer('all-mpnet-base-v2')
nlp = spacy.load("en_core_web_sm")

schema_entities = [
    "Route", "Metaroute", "Flight", "Luggage", "Drug", "Risk", "Passenger",
    "Category", "Airport", "Municipality", "Country", "EntryScanner",
    "CustomsScanner", "ManualInspection", "EntryRoom", "CustomsRoom", "ConveyorRoom", "Airline", "Ticket"
]

schema_relations_named = {
    ("Metaroute", "Route"): "CONTAINS_ROUTE",
    ("Drug", "Category"): "BELONGS_TO_CATEGORY",
    ("CustomsScanner", "CustomsRoom"): "IS_LOCATED_IN",
    ("EntryScanner", "EntryRoom"): "IS_LOCATED_IN",
    ("Flight", "Route"): "FOLLOWS",
    ("Passenger", "Luggage"): "HAS_LUGGAGE",
    ("Passenger", "Ticket"): "HAS_TICKET",
    ("Airline", "Country"): "BELONGS_TO_COUNTRY",
    ("Airline", "Route"): "HAS_ROUTE",
    ("Airport", "Municipality"): "IS_IN_CITY",
    ("Municipality", "Country"): "IS_IN_COUNTRY",
    ("Route", "Airport"): "ARRIVES_TO",
    ("Passenger", "Flight"): "BOOKED",
    ("Luggage", "Drug"): "CONTAINS_CONTRABAND",
    ("Luggage", "Flight"): "TRAVEL_INSIDE",
    ("Metaroute", "Risk"): "HAS_RISK",
    ("Risk", "Drug"): "CONTRABAND_TYPE",
    ("Luggage", "Risk"): "HAS_RISK",
    ("Luggage", "EntryScanner"): "SCANNED_BY",
    ("Luggage", "CustomsScanner"): "SCANNED_BY",
    ("Luggage", "ManualInspection"): "SCANNED_BY"
}

# ---------- user_query ----------
user_query = "Which flights may be consider high risk?"
print(f"User query: \"{user_query}\"\n")

# ----------  NLP ----------
print("Running spaCy NER and POS tagging...\n")
doc = nlp(user_query)

noun_ner_terms = [token.text for token in doc if token.pos_ in {"NOUN", "PROPN"}]
print(f"Extracted noun terms: {noun_ner_terms}\n")

# ---------- Embeddings  ----------
print("Calculating semantic similarity to schema entities...\n")
entity_embeddings = embedder.encode(schema_entities, convert_to_tensor=True)
ner_embeddings = embedder.encode(noun_ner_terms, convert_to_tensor=True)

top_1_entities = []
for i, ner in enumerate(noun_ner_terms):
    scores = util.cos_sim(ner_embeddings[i], entity_embeddings)[0]
    best_idx = scores.argmax().item()
    best_entity = schema_entities[best_idx]
    top_1_entities.append(best_entity)
    print(f"🔹 \"{ner}\" matched to \"{best_entity}\" with score {scores[best_idx]:.4f}")

top_1_entities = list(set(top_1_entities))
print(f"\nUnique top-1 schema entities: {top_1_entities}\n")


# ---------- search undirected paths----------
print("Searching undirected paths between top entities...\n")
G = nx.Graph()
for src, dst in schema_relations_named:
    G.add_edge(src, dst)

max_hops = 5
paths = []
for i in range(len(top_1_entities)):
    for j in range(i + 1, len(top_1_entities)):
        try:
            all_paths = list(nx.all_simple_paths(G, source=top_1_entities[i], target=top_1_entities[j], cutoff=max_hops))
            paths.extend(all_paths)
            for path in all_paths:
                print("   → Found path:", " → ".join(path))
        except nx.NetworkXNoPath:
            print(f"No path between {top_1_entities[i]} and {top_1_entities[j]}")

if not paths:
    print("No paths found.\n")
else:
    print(f"\n Found {len(paths)} undirected path(s).\n")

# remove paths inside other paths
print("Refining and filtering paths...\n")
refined_paths = []
paths_sorted = sorted(paths, key=len, reverse=True)
for path in paths_sorted:
    path_set = set(path)
    if not any(set(existing_path).issubset(path_set) for existing_path in refined_paths):
        refined_paths.append(path)
# show results
print("\nRefined and sorted paths (from largest to smallest):")
for path in refined_paths:
    print("  ", " → ".join(path))

# Path with the most nodes
max_nodes_path = max(refined_paths, key=len)
print("\nPath with the most nodes:", " → ".join(max_nodes_path))

# Check if all top-1 entities are present in the paths
all_entities_in_paths = [path for path in refined_paths if all(entity in path for entity in top_1_entities)]
if all_entities_in_paths:
    print("\nPaths that contain all the extracted entities:")
    for path in all_entities_in_paths:
        print("   →", " → ".join(path))
else:
    print("\nNo paths contain all the extracted entities.")


print("\nAccurate textual description of paths...\n")

final_paths = []
for path in all_entities_in_paths:
    description = []
    for i in range(len(path) - 1):

        rel = schema_relations_named.get((path[i], path[i+1]))  # Check if relation is (A -> B)
        
        # If the relation doesn't exist, check the reverse (B -> A)
        if not rel:
            rel = schema_relations_named.get((path[i+1], path[i]))  # Check if relation is (B -> A)
        if rel:
            # If the relation is defined as (A -> B), use A -> B
            # Otherwise, we reverse the direction (B -> A)
            if (path[i], path[i+1]) in schema_relations_named:
                description.append(f"{path[i]} -[{rel}]-> {path[i+1]}")  # A -> B (forward direction)
            else:
                description.append(f"{path[i+1]} -[{rel}]-> {path[i]}")  # B -> A (reversed direction)
        else:
            # If no relation exists in schema_relations_named, just use → (no direction defined)
            description.append(f"{path[i]} → {path[i+1]} (no direct relation defined)")
    
    # Print the description for each path
    print("   → Path description: " + "  ".join(description))
    final_paths.append(description)



def generate_optional_cypher_query(path):
    matches = []
    node_vars = set()

    for rel in path:
        parts = rel.split(' -[')
        left = parts[0].strip()
        relation, right = parts[1].split(']->')
        right = right.strip()

        left_var = left.lower()
        right_var = right.lower()

        matches.append(f"OPTIONAL MATCH ({left_var}:{left})-[:{relation}]->({right_var}:{right})")

        node_vars.add(left_var)
        node_vars.add(right_var)

    return_clause = "RETURN DISTINCT " + ", ".join(sorted(node_vars))
    return "\n".join(matches) + "\n" + return_clause

# Generar queries con OPTIONAL MATCH
queries = [generate_optional_cypher_query(path) for path in final_paths]

# Mostrar resultados
for i, q in enumerate(queries):
    print(f"Query {i+1}:\n{q}\n")

def generate_optional_clauses_with_aliases(path, index):
    optional_clauses = []
    node_aliases = set()
    used = {}

    for rel in path:
        parts = rel.split(' -[')
        left = parts[0].strip()
        relation, right = parts[1].split(']->')
        right = right.strip()

        left_base = left.lower()
        right_base = right.lower()

        # Create unique aliases for each node based on the index
        left_alias = f"{left_base}{index}"
        right_alias = f"{right_base}{index}"

        # save unique aliases to avoid duplicate MATCH
        if (left, left_alias) not in used:
            used[(left, left_alias)] = True
        if (right, right_alias) not in used:
            used[(right, right_alias)] = True

        clause = f"OPTIONAL MATCH ({left_alias}:{left})-[:{relation}]->({right_alias}:{right})"
        optional_clauses.append(clause)

        node_aliases.add(left_alias)
        node_aliases.add(right_alias)

    return optional_clauses, node_aliases

# build the final Cypher query with all paths
all_clauses = []
all_aliases = set()

for i, path in enumerate(final_paths, start=1):
    clauses, aliases = generate_optional_clauses_with_aliases(path, i)
    all_clauses.extend(clauses)
    all_aliases.update(aliases)

# Combine all clauses and aliases into a single Cypher query
query = "\n".join(all_clauses) + "\nRETURN DISTINCT " + ", ".join(sorted(all_aliases))

print(query)

# subschema_section
subschema_section = []

for path in final_paths:
    for edge in path:
        # Split the string using regex to extract Entity1, relation, Entity2
        import re
        match = re.match(r'(\w+)\s*-\[\s*(\w+)\s*\]->\s*(\w+)', edge)
        if match:
            entity1, relation, entity2 = match.groups()
            subschema_section.append(((entity1, entity2), relation))

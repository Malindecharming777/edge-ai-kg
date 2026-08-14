// {{KG_NAME}} Knowledge Graph — schema
// Rename this file to schema/{{KG_SLUG}}_kg.cypher
// Node labels: {{NODE_LABELS}}
// Edge types:  {{EDGE_TYPES}}

// --- Constraints / indexes (one per node label) ---
CREATE CONSTRAINT nodea_id IF NOT EXISTS FOR (n:{{NodeA}}) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT nodeb_id IF NOT EXISTS FOR (n:{{NodeB}}) REQUIRE n.id IS UNIQUE;

// --- Relationship shapes (documentation) ---
// (:{{NodeA}})-[:{{REL}}]->(:{{NodeB}})

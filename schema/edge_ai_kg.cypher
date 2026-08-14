// Edge AI Deployment Knowledge Graph -- schema
//
// Node labels: Vendor, SoC, Accelerator, Board, Runtime, Operator, Kernel,
//              Model, ModelVariant, Sensor, SignalStage, ClinicalTask,
//              BenchmarkTask, Dataset, Certification, Deployment
//
// Every node carries `provenance` ("real" | "synthetic") and `source`.
//
// Edge types:  MADE_BY, HAS_SOC, HAS_ACCELERATOR, TARGETS, IMPLEMENTS,
//              RUNS_ON, PROVIDED_BY, USES_OPERATOR, VARIANT_OF, SOLVES,
//              TRAINED_ON, REQUIRES_SENSOR, FEEDS, NEXT_STAGE, PRECEDES,
//              OF_VARIANT, ON_BOARD, VIA_RUNTIME, USES_ACCELERATOR,
//              CERTIFIED_FOR, GOVERNED_BY, MEASURES
//
// This engine accepts `CREATE INDEX ON :Label(prop)`. It does NOT parse
// `CREATE CONSTRAINT ... REQUIRE ... IS UNIQUE`; uniqueness of `id` is
// guaranteed by the loader, which mints ids deterministically.

// --- id indexes (one per label; drives edge-creation lookups) ---
CREATE INDEX ON :Vendor(id);
CREATE INDEX ON :SoC(id);
CREATE INDEX ON :Accelerator(id);
CREATE INDEX ON :Board(id);
CREATE INDEX ON :Runtime(id);
CREATE INDEX ON :Operator(id);
CREATE INDEX ON :Kernel(id);
CREATE INDEX ON :Model(id);
CREATE INDEX ON :ModelVariant(id);
CREATE INDEX ON :Sensor(id);
CREATE INDEX ON :SignalStage(id);
CREATE INDEX ON :ClinicalTask(id);
CREATE INDEX ON :Dataset(id);
CREATE INDEX ON :Certification(id);
CREATE INDEX ON :Deployment(id);
CREATE INDEX ON :BenchmarkTask(id);

// --- lookup indexes used by the benchmark queries ---
CREATE INDEX ON :Operator(name);
CREATE INDEX ON :Operator(category);
CREATE INDEX ON :Accelerator(kind);
CREATE INDEX ON :ModelVariant(precision);
CREATE INDEX ON :ClinicalTask(category);
CREATE INDEX ON :Model(family);

// --- provenance: every node is stamped real | synthetic ---
CREATE INDEX ON :Deployment(provenance);
CREATE INDEX ON :Kernel(provenance);
CREATE INDEX ON :Board(provenance);
CREATE INDEX ON :Accelerator(provenance);
CREATE INDEX ON :Kernel(execution_provider);

// --- Relationship shapes (documentation only) ---
// (:Board)-[:HAS_SOC]->(:SoC)-[:HAS_ACCELERATOR]->(:Accelerator)
// (:Board)-[:MADE_BY]->(:Vendor)          (:SoC)-[:MADE_BY]->(:Vendor)
// (:Board)-[:CERTIFIED_FOR]->(:Certification)
// (:Runtime)-[:TARGETS]->(:Accelerator)
// (:Kernel)-[:IMPLEMENTS]->(:Operator)
// (:Kernel)-[:RUNS_ON]->(:Accelerator)
// (:Kernel)-[:PROVIDED_BY]->(:Runtime)
// (:Model)-[:USES_OPERATOR {count}]->(:Operator)
// (:ModelVariant)-[:VARIANT_OF]->(:Model)
// (:Model)-[:SOLVES]->(:ClinicalTask)     (:Model)-[:TRAINED_ON]->(:Dataset)
// (:ClinicalTask)-[:REQUIRES_SENSOR]->(:Sensor)
// (:ClinicalTask)-[:GOVERNED_BY]->(:Certification)
// (:Sensor)-[:FEEDS]->(:SignalStage)-[:NEXT_STAGE]->(:SignalStage)
// (:SignalStage)-[:PRECEDES]->(:Model)
// (:SignalStage)-[:USES_OPERATOR]->(:Operator)
// (:Deployment)-[:OF_VARIANT]->(:ModelVariant)
// (:Deployment)-[:ON_BOARD]->(:Board)
// (:Deployment)-[:VIA_RUNTIME]->(:Runtime)
// (:Deployment)-[:USES_ACCELERATOR]->(:Accelerator)
// (:Deployment)-[:MEASURES]->(:Model)          -- real MLPerf Tiny submissions
// (:Model)-[:SOLVES]->(:BenchmarkTask)        -- real MLPerf Tiny tasks

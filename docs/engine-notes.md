# Engine notes -- Samyama Graph v1.7.0

Behaviour observed while building this KG, on the OSS engine at v1.7.0
(`target/release/samyama --http-port 8080`). Every item here is load-bearing:
the loader or the query catalog works around it. Verified 2026-08-14.

---

## 1. A trailing bound variable in a second MATCH clause is not joined

**Severity: correctness. Silently returns wrong rows.**

When a second `MATCH` clause re-mentions a variable bound by an earlier `MATCH`,
and that variable appears in a **non-leading position** in the second pattern
while *another* shared variable leads it, the engine does not enforce the join.
It returns a cartesian product instead.

Minimal reproduction -- one board `B1`, two models `M1`/`M2`, each with an fp32
and an int8 variant deployed on `B1`:

```cypher
MATCH (b:RBd)<-[:R_ON]-(d1:RDp)-[:R_OF]->(v1:RVr)-[:R_VO]->(m:RMd) WHERE v1.p="fp32"
MATCH (b)<-[:R_ON]-(d2:RDp)-[:R_OF]->(v2:RVr)-[:R_VO]->(m)         WHERE v2.p="int8"
RETURN m.id, v1.id, v2.id
```

Expected 2 rows (`M1,M1fp32,M1int8` and `M2,M2fp32,M2int8`). **Got 4** -- the
cross pairs `M1,M2fp32,M1int8` and `M2,M1fp32,M2int8` are also returned. `v2` is
correctly constrained to `m`; `v1` is not.

Variants tested:

| Shape | Small graph | Full graph (24K nodes) |
|---|---|---|
| 2nd MATCH, shared var **trailing**, another shared var leading | **WRONG** | **WRONG** |
| 2nd MATCH, shared var **leading** | correct | correct |
| 2nd MATCH, only one shared variable | correct | correct |
| `WITH` between the two MATCH clauses | **WRONG** | **WRONG** |
| Single MATCH, comma-separated patterns | correct | **WRONG** |

**Note the last row.** The comma-separated single-`MATCH` form looks like a fix
on a toy graph and still breaks at scale -- presumably a different join plan is
chosen once the cardinalities and indexes are real. Any "workaround" for this
bug must be validated on a full-size graph; a passing 6-node reproduction proves
nothing.

**Workaround actually used here:** avoid the self-join entirely. `EA04` is
written as a single linear pattern plus conditional aggregation
(`sum(CASE WHEN ... )`), which has no second binding to lose:

```cypher
MATCH (m:Model)<-[:VARIANT_OF]-(v:ModelVariant)<-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WITH m.name AS model, b.name AS board,
     sum(CASE WHEN v.precision = "fp32" AND d.fits = 0 THEN 1 ELSE 0 END) AS fp32_misses,
     sum(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN 1 ELSE 0 END) AS int8_hits,
     max(CASE WHEN v.precision = "fp32" THEN v.size_kb ELSE 0 END) AS fp32_kb,
     max(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 0 END) AS int8_kb
WHERE fp32_misses > 0 AND int8_hits > 0
RETURN model, board, fp32_kb, int8_kb
```

Because `int8_kb` is by construction exactly `fp32_kb / 4`, a cartesian product
is *detectable*: `tests/test_correctness.py` asserts that ratio, and
`test_ea04_shape_is_not_a_cartesian_product` pins the behaviour on a
purpose-built 4-deployment fixture.

---

## 2. `RETURN DISTINCT` is a no-op

**Severity: correctness. Returns duplicate rows.**

```cypher
MATCH (b:Board) RETURN DISTINCT b.form_factor
```

returns 120 rows (one per board) across 8 distinct form factors. Multi-column
`RETURN DISTINCT` is equally ineffective.

`DISTINCT` *inside an aggregate* -- `count(DISTINCT op)` -- works correctly and
is used throughout the catalog.

**Workaround used here:** deduplicate with a `WITH`-grouping plus an aggregate,
which groups correctly:

```cypher
MATCH (b:Board) WITH b.form_factor AS f, b.year AS y, count(b) AS n RETURN f, y, n
```

---

## 3. `ORDER BY` on a RETURN-introduced alias is silently ignored

**Severity: correctness. Returns an arbitrary subset when combined with LIMIT.**

```cypher
MATCH (d:Deployment) RETURN d.latency_ms AS a ORDER BY a ASC LIMIT 6
-- returns 10.976, 12.771, 7.569, 9.003, ...   (unsorted)
```

An alias introduced in `RETURN` cannot be referenced by `ORDER BY`. The clause
is dropped without error. Combined with `LIMIT`, this is worse than an unsorted
result: `ORDER BY latency ASC LIMIT 12` returns *an arbitrary 12 rows*, not the
twelve fastest -- while looking exactly like a top-N.

| Form | Result |
|---|---|
| `RETURN d.latency_ms ORDER BY d.latency_ms` | correct |
| `RETURN d.latency_ms AS a ORDER BY d.latency_ms` | correct |
| `RETURN d.latency_ms AS a ORDER BY a` | **ignored** |
| `WITH d.latency_ms AS a RETURN a ORDER BY a` | correct |
| `WITH ... count(x) AS n RETURN n ORDER BY n` | correct |

Aliases introduced by `WITH` work; aliases introduced by `RETURN` do not. A
`RETURN` containing an aggregate happens to work, because the implicit grouping
pass establishes the alias.

**Workaround used here:** every catalog query projects through a `WITH` before
`RETURN`, and sorts on the `WITH` alias.

### 3b. Only the first `ORDER BY` key is honoured

`ORDER BY category, operator` sorts by `category` and leaves `operator`
unsorted within each group. Multi-key sorts are therefore avoided entirely;
`tests/test_correctness.py::test_order_by_is_actually_applied` asserts every
catalog query uses a single sort key *and* that the result really is sorted.

---

## 4. `min()` mis-compares an integer sentinel against float values

```cypher
RETURN min(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 999999   END)  -- 999999  (wrong)
RETURN min(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 999999.0 END)  -- 6.9     (right)
```

With an `int` sentinel, `min()` returns the sentinel even though float values
compare smaller. Writing the sentinel as a float fixes it. This is the same
int/float coercion weakness that makes `WHERE n.x > 0.5` return nothing when
`x` was stored as an int -- **keep numeric literal types consistent with the
stored property type.**

---

## 5. Negated pattern predicates do not parse

`WHERE NOT (:Acc)-[:SUPPORTS]->(op)` is a parse error.

**Workaround used here:** the standard anti-join, which works correctly:

```cypher
MATCH (op:Operator)
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op)
WITH op, count(k) AS kernels
WHERE kernels = 0
RETURN op.name
```

---

## 6. `CREATE CONSTRAINT ... REQUIRE ... IS UNIQUE` does not parse

Only `CREATE INDEX ON :Label(prop)` is accepted. Uniqueness of `id` is therefore
a loader invariant, not an engine-enforced one -- ids are minted deterministically
in `etl/generate.py`.

---

## 7. The tenant / graph argument is ignored on the OSS HTTP path

`client.query(cypher, "some_graph")` writes to, and reads from, the single
`default` graph regardless of the name passed. Writing to `graph_a` is visible
from `graph_b`, and `list_graphs()` only ever reports `default`.

This is consistent with multi-tenancy being an Enterprise Edition feature, but
it is worth knowing: **passing a graph name does not isolate anything on OSS.**
Two datasets loaded into "different" graphs will silently merge.

**Workaround used here:** the loader defaults to `default` and resets the graph
before loading, rather than relying on tenant isolation.

---

## 8. Deleted property columns resurrect onto new nodes

**Severity: data integrity. Stale values from deleted data appear on new data.**

```cypher
CREATE (:GhostProp {id: "a", ghost: "LEAKED"});
MATCH (n:GhostProp) DETACH DELETE n;          -- count is now 0
CREATE (:GhostProp {id: "b"});                -- note: no `ghost` property
MATCH (n:GhostProp) RETURN n.id, n.ghost;     -- ["b", "LEAKED"]   <-- expected null
```

A node created without a property inherits the value the *previous* generation
of nodes had for that column. A global `MATCH (n) DETACH DELETE n` does not help
either -- the columnar property store survives the delete.

We hit this for real: an internal `_chain` field briefly leaked onto `Sensor`
nodes, and after the generator was fixed and the graph reloaded, **every Sensor
still reported the stale blob** even though the source data no longer contained
it. It looked like the fix had failed.

**Workaround:** `DETACH DELETE` is not a reset. To genuinely reset a graph,
stop the server and start it against a fresh data directory:

```bash
pkill -f '[t]arget/release/samyama'
rm -rf <data-dir>/samyama_data
samyama --http-port 8080
```

**Corollary:** never rely on `DETACH DELETE` between experiments that change a
node's property *schema*. Changing values is fine; removing a property is not.

### 8b. `<>` against a null property matches

`WHERE s._chain <> ""` returns rows where `s._chain` is null. Standard Cypher
would treat `null <> ""` as null and filter the row out. Use an explicit
`IS NULL` / `IS NOT NULL` check instead of inequality when a property may be
absent.

---

## What works well

Everything the catalog depends on, other than the above:
`OPTIONAL MATCH`, `WITH` + aggregation (`count`/`sum`/`avg`/`min`/`max`/`collect`),
variable-length paths (`-[:R*0..3]->`), `shortestPath`, `CASE`, `IN`, `SKIP` / `LIMIT`, string functions, `EXPLAIN`, and `CREATE INDEX`.

Load throughput on this box (RTX 4050 laptop, server on localhost):
~35K nodes/s batched 250-per-`CREATE`, ~3.1K edges/s batched 100-per-statement
using `MATCH ... WHERE id = ... CREATE`.

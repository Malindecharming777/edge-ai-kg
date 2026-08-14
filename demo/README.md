# Demo

Narrated walkthrough of the {{KG_NAME}} KG on a fast, real subset.

```bash
python -m demo.demo
```
Record/regenerate:
```bash
asciinema rec --overwrite --cols 92 --rows 32 --idle-time-limit 2.0 \
  -c "bash -c 'python -m demo.demo'" demo/{{KG_SLUG}}.cast
agg demo/{{KG_SLUG}}.cast demo/{{KG_SLUG}}.gif
```

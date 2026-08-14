"""Build and load the {{KG_NAME}} graph into Samyama."""
import click

@click.command()
@click.option("--limit", type=int, default=None, help="Cap rows per source for a fast demo load.")
def main(limit):
    # TODO: connect -> run schema/{{KG_SLUG}}_kg.cypher -> load nodes/edges
    print(f"[loader] loading {{KG_SLUG}} KG (limit={limit}) ...")

if __name__ == "__main__":
    main()

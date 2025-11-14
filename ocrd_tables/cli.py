import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .processor import OcrdTables


@click.command(name="ocrd-tables")
@ocrd_cli_options
def ocrd_tables_cli(*args, **kwargs):
    """
    Fuse YOLO columns and textlines into table cells (using DBSCAN).
    """
    return ocrd_cli_wrap_processor(OcrdTables, *args, **kwargs)


if __name__ == "__main__":
    ocrd_tables_cli()

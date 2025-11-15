import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from .processor import OcrdTables


@click.command()
@ocrd_cli_options
def ocrd_tables(*args, **kwargs):
    """CLI entrypoint for ocrd-tables."""
    return ocrd_cli_wrap_processor(OcrdTables, *args, **kwargs)


if __name__ == "__main__":
    ocrd_tables()

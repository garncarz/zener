#!/usr/bin/env python3

from datetime import datetime
import os
from pathlib import Path
import zipfile

import click

from zener.generate import generate_all_testing_data
from zener.optimize import GAAll


def zip_filtered_data():
    source = Path('.')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_name = f'output_{timestamp}.zip'

    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_LZMA) as zipf:
        for file in source.rglob('*'):
            if file.is_file() and file.suffix not in ['.dat', '.pkl', '.zip']:
                arcname = Path(f'output_{timestamp}') / file.relative_to(source)
                zipf.write(file, arcname)

    click.echo(f'Compressed into {zip_name}.')


@click.command()
@click.option('--generate', is_flag=True, help='Generate testing data.')
@click.option('--optimize', is_flag=True, help='Run optimization.')
@click.option('--result', is_flag=True, help='Only save the result from the last generation.')
@click.pass_context
def main(ctx, generate, optimize, result):
    if not any([generate, optimize, result]):
        click.echo(ctx.get_help())
        ctx.exit()

    os.chdir('data')

    if generate:
        generate_all_testing_data()

    if optimize or result:
        ga_all = GAAll()
        if optimize:
            ga_all.run()
        ga_all.save_result()
        zip_filtered_data()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

from datetime import datetime
import os
from pathlib import Path
import zipfile

import click

from zener.generate import generate_all_testing_data
from zener.optimize import GAAll
from zener.settings import settings


def zip_filtered_data():
    source = Path('.')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_suffix = f'_{settings.job_id}' if settings.job_id else ''

    folder_name = f'output_{timestamp}{job_suffix}'
    zip_name = f'{folder_name}.zip'

    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in source.rglob('*'):
            if should_include_file(file):
                arcname = Path(folder_name) / file.relative_to(source)
                zipf.write(file, arcname)

    click.echo(f'Compressed into {zip_name}.')


def should_include_file(file: Path) -> bool:
    if not file.is_file():
        return False
    if file.suffix in ['.dat', '.pkl', '.zip']:
        return False
    if file.name.startswith('celery.') and file.name.endswith('exchange'):
        return False

    return True


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

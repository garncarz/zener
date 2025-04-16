#!/usr/bin/env python3

import click

from generate import generate_all_testing_data
from optimize import GAAll


@click.command()
@click.option('--generate', is_flag=True, help='Generate testing data.')
@click.option('--optimize', is_flag=True, help='Run optimization.')
@click.option('--result', is_flag=True, help='Only save the result from the last generation.')
@click.pass_context
def main(ctx, generate, optimize, result):
    if not any([generate, optimize, result]):
        click.echo(ctx.get_help())
        ctx.exit()

    if generate:
        generate_all_testing_data()

    if optimize or result:
        ga_all = GAAll()
        if optimize:
            ga_all.run()
        ga_all.save_result()


if __name__ == '__main__':
    main()

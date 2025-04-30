import matplotlib.pyplot as plt

def plot_time_series(merged):
    cols = ['Stress', 'StrainNonLinEl', 'StrainViscEl', 'TotStrain']
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2.5 * len(cols)), sharex=True)

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.plot(
            merged['time'].to_numpy(),
            merged[f'{col}_target'].to_numpy(),
            label='Target',
            color='black',
            linestyle='--',
        )
        ax.plot(
            merged['time'].to_numpy(),
            merged[f'{col}_model'].to_numpy(),
            label='Model',
            color='blue',
        )
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('plots.png')
    plt.close(fig)


def plot_hysteresis(merged):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        merged['StrainViscEl_target'].to_numpy(),
        merged['Stress_target'].to_numpy(),
        label='Target StrainViscEl', color='gray', linestyle=':',
    )
    ax.plot(
        merged['StrainViscEl_model'].to_numpy(),
        merged['Stress_model'].to_numpy(),
        label='Model StrainViscEl', color='black', linestyle='--',
    )
    ax.plot(
        merged['TotStrain_target'].to_numpy(),
        merged['Stress_target'].to_numpy(),
        label='Target TotStrain', color='blue', linestyle=':',
    )
    ax.plot(
        merged['TotStrain_model'].to_numpy(),
        merged['Stress_model'].to_numpy(),
        label='Model TotStrain', color='lightblue', linestyle='--',
    )

    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig('hysteresis.png')
    plt.close(fig)


def plot_stress(merged):
    plt.figure(figsize=(6, 2))

    plt.plot(
        merged['time'].to_numpy(),
        merged['Stress_target'].to_numpy(),
        label='Target',
        color='black',
        linestyle='--',
    )
    plt.plot(
        merged['time'].to_numpy(),
        merged['Stress_model'].to_numpy(),
        label='Model',
        color='blue',
    )

    plt.xlabel('Time')
    plt.ylabel('Stress')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('stress.png')
    plt.close()

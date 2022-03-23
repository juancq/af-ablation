import argparse
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt


def main():
    '''
    Script for generating KM plot (Figure 1).
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('survival_data', type=str, help='files with survival data')
    parser.add_argument('--output_file', type=str, help='output file to save summary results')
    parser.add_argument('--km_plot', action='store_true', help='only calculate KM plot', default=False)
    args = parser.parse_args()

    df = pd.read_csv(args.survival_data, usecols=['sex', 'year', 'observed', 'duration'],low_memory=False)

    df = df[df['year'] > 2008]
    # remove people who died on index admission
    dead_mask = (df.observed==0) & (df.duration < 0.01)
    df = df[~dead_mask].copy()

    print('Median [IQR] follow-up:')
    print(f'{(df.duration/12).quantile([0.25, 0.5, 0.75])}')

    at_risk_counts = True
    def plot_km(df, label=''):
        ax = plt.subplot(111)
        kmf = KaplanMeierFitter()
        kmf.fit(df['duration'], event_observed=df['observed'], label='')
        kmf.plot_cumulative_density(ax=ax, legend=False, at_risk_counts=at_risk_counts)
        plt.title('Cumulative Probability of Receiving Ablation')
        ax.set_xlim((0,120))
        ax.set_xticks(list(range(0,120+12,12)))
        ax.set_xticklabels(list(range(11)))

        ax.grid(b=True, which='major', color='silver', linewidth=1.0, alpha=0.3)
        ax.set_xlabel('Years')
        folder = 'survival_plots/'
        plt.savefig(f'{folder}km_plot_ablation_vs_noablation{label}.png', bbox_inches='tight', dpi=300)
    
    plot_km(df, label='_main')


if __name__ == '__main__':
    main()

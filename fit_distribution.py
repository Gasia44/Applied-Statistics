"""
USAGE
$python fit_distribution.py --data_path aircondit7.txt --d gamma beta pareto --nbest 10 --save_directory ./result/
$python fit_distribution.py --data_path aircondit7.txt --d ALL --nbest 20 --save_directory ./result/
"""

import argparse
import json
import os
import time
import warnings

import fitter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


def parse_args():
    """
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='input data file path', type=str, default='aircondit7.txt')
    parser.add_argument('--nbest', help='input n best to plot', type=int, default=10)
    parser.add_argument('--d', help='choose distributions to fit', \
                        choices=['ALL', \
                                 'alpha', 'anglit', 'arcsine', 'argus', \
                                 'beta', 'betaprime', 'bradford', 'burr', 'burr12', \
                                 'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', \
                                 'dgamma', 'dweibull', \
                                 'erlang', 'expon', 'exponnorm', 'exponpow', 'exponweib', 'frechet_l', 'frechet_r', \
                                 'gamma', 'gausshyper', 'genexpon', 'genextreme', 'gengamma', \
                                 'genhalflogistic', 'genlogistic', 'gennorm', 'genpareto', 'gilbrat', \
                                 'gompertz', 'gumbel_l', 'gumbel_r', \
                                 'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm', 'hypsecant', \
                                 'invgamma', 'invgauss', 'invweibull', \
                                 'johnsonsb', 'johnsonsu', \
                                 'kappa3', 'kappa4', 'ksone', 'kstwobign', \
                                 'laplace', 'levy', 'levy_l', 'levy_stable', 'loggamma', 'logistic', \
                                 'loglaplace', 'lognorm', 'lomax', \
                                 'maxwell', 'mielke', 'moyal', \
                                 'nakagami', 'ncf', 'nct', 'ncx2', 'norm', 'norminvgauss', \
                                 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', \
                                 'rayleigh', 'rdist', 'recipinvgauss', 'reciprocal', 'rice', 'rv_continuous',
                                 'rv_histogram', \
                                 'semicircular', 'skewnorm', \
                                 't', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', \
                                 'uniform', 'vonmises', 'vonmises_line', \
                                 'wald', 'weibull_max', 'weibull_min', 'wrapcauchy'], \
                        nargs='+', default=['ALL'])
    parser.add_argument('--save_directory', help='input save result directory', type=str, default='./result/')
    return parser.parse_args()


def read_data(file_path):
    """
    read data from file txt, csv,...
    :param file_path:
    :return: dataframe with one column of data
    """
    delim = '\t' if file_path.split('.')[-1] == "txt" else ','
    df = pd.read_csv(file_path, header=None, delimiter=delim)
    df = df[[df.columns[-1]]]
    df.columns = ['data']
    return df


def fit_distributions(distributions, data):
    """
    fit distributions on data
    :param distributions:
    :param data:
    :return:
    """
    if distributions[0] == "ALL":
        f = fitter.Fitter(data, distributions=None, verbose=True)
    else:
        f = fitter.Fitter(data, distributions=distributions, verbose=True)
    f.fit()
    return f


def check_destination(destination):
    """
    check if destination directory exists, otherwise create one
    :param destination:
    :return:
    """
    if not os.path.exists(destination):
        os.makedirs(destination)


def calculate_p_value(fitted_model):
    """
    Obtain the KS test P statistic,
    A value of greater than 0.05 means that the fitted distribution is not significantly different to the observed distribution of the data.
    :param fitted_model:
    :return:
    """
    p_values = {}
    for distr in fitted_model.fitted_param.keys():
        param = fitted_model.fitted_param[distr]
        p = scipy.stats.kstest(fitted_model._data, distr, args=param)[1]
        p_rounded = np.around(p, 5)
        p_values[distr] = p_rounded

    df_error = fitted_model.df_errors.sort_values('sumsquare_error')
    df_p_value = pd.DataFrame(p_values, index=[0]).transpose()
    df_p_value.columns = ['p_value']
    df_error_p_value = df_error.merge(df_p_value, left_index=True, right_index=True)
    df_error_p_value = df_error_p_value.sort_values(['sumsquare_error', 'p_value'], ascending=[True, False])
    return df_error_p_value


def plot_pdf(fitted_model, save_destination, n_best=5, lw=1):
    """
    Plots Probability density functions of the distributions
    :param fitted_model:
    :param save_destination:
    :param Nbest: the first Nbest distribution will be taken (default to best 5)
    :param lw: linewidth
    :return:
    """
    assert n_best > 0
    if n_best > len(fitted_model.distributions):
        n_best = len(fitted_model.distributions)

    names = fitted_model.df_errors.sort_values(by="sumsquare_error").index[0:n_best]

    plt.figure(figsize=(15, 7))
    for name in names:
        if name in fitted_model.fitted_pdf.keys():
            plt.plot(fitted_model.x, fitted_model.fitted_pdf[name], lw=lw, label=name)
        else:
            print("{} was not fitted. no parameters available".format(name))

    plt.hist(fitted_model._data, bins=100, density=True, alpha=0.3, label='data')
    plt.grid(True)
    plt.legend(fontsize=12)

    plt.title('Probability Density Functions For Best Fitted Distributions')
    plt.ylabel('density', fontsize=12)
    plt.xlabel('data', fontsize=12)
    plt.savefig(save_destination + 'probability_density_functions.png', bbox_inches='tight')
    plt.show()


def save_result(destination, data, fitted_param):
    """
    save the result in the specified destination
    :param destination:
    :param data:
    :return:
    """
    data.to_csv(destination + "fitted_distributions_result.csv")
    json.dump(fitted_param, open(destination + "fitted_parameters.json", 'w'), indent=4)


def main(ARGS):
    print('Started running....')
    data_path = ARGS.data_path
    destination = ARGS.save_directory
    distributions = ARGS.d
    n_best = ARGS.nbest

    print('Reading data....')
    df = read_data(data_path)
    print('Reading data done.')

    check_destination(destination)

    print('Fitting distributions....')
    fitted_model = fit_distributions(distributions, df['data'].values)
    print('Fitting distributions done.\n')

    df_error_p_value = calculate_p_value(fitted_model)
    print("\n Best fit parameters: {}".format(fitted_model.get_best()))

    print("Sorted by good fit score -mean squarred error")
    print(df_error_p_value)
    plot_pdf(fitted_model, destination, n_best=n_best, lw=2)

    save_result(destination, df_error_p_value, fitted_model.fitted_param)
    print("Result saved in: {:s}.".format(destination))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    start = time.time()
    ARGS = parse_args()
    main(ARGS)

    elapsed_time = time.time() - start
    print("Elapsed time: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

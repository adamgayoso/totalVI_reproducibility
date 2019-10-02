from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from scvi.inference import TotalPosterior
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import pandas as pd
from tqdm.auto import tqdm

plt.switch_backend("agg")


class TotalPosteriorPredictiveCheck:
    """Posterior predictive checks for comparing totalVI models
    """

    def __init__(
        self,
        posteriors_dict: Dict[str, TotalPosterior],
        n_samples: int = 10,
        batch_size=32,
    ):
        """
        Args:
            posteriors_dict (Dict[str, Posterior]): dictionary of Posterior objects fit on the same dataset
            n_samples (int, optional): Number of posterior predictive samples. Defaults to 10.
        """
        self.posteriors = posteriors_dict
        self.dataset = posteriors_dict[next(iter(posteriors_dict.keys()))].gene_dataset
        self.raw_counts = None
        self.posterior_predictive_samples = {}
        self.n_samples = n_samples
        self.models = {}
        self.metrics = {}
        self.raw_metrics = {}
        self.batch_size = batch_size

        self.store_posterior_samples()

    def store_posterior_samples(self):
        """Samples from the Posterior objects and sets raw_counts
        """

        for m, post in self.posteriors.items():
            pp_counts, original = post.sequential().generate(
                n_samples=self.n_samples, batch_size=self.batch_size
            )
            self.posterior_predictive_samples[m] = pp_counts
        self.raw_counts = original

    def coeff_of_variation(self, cell_wise: bool = True):
        """Calculate the coefficient of variation

        Args:
            cell_wise (bool, optional): Calculate for each cell across genes. Defaults to True.
                                        If False, calculate for each gene across cells.
        """
        axis = 1 if cell_wise is True else 0
        identifier = "cv_cell" if cell_wise is True else "cv_gene"
        df = pd.DataFrame()
        for m, samples in self.posterior_predictive_samples.items():
            cv = np.mean(
                np.std(samples, axis=axis) / np.mean(samples, axis=axis), axis=-1
            )
            df[m] = cv.ravel()

        df["raw"] = np.std(self.raw_counts, axis=axis) / np.mean(
            self.raw_counts, axis=axis
        )

        self.metrics[identifier] = df

    def median_absolute_deviation(self):
        df = pd.DataFrame()
        for m, samples in self.posterior_predictive_samples.items():
            mean_sample = np.mean(samples, axis=-1)
            mad_gene = np.median(
                np.abs(
                    mean_sample[:, : self.dataset.nb_genes]
                    - self.raw_counts[:, : self.dataset.nb_genes]
                )
            )
            mad_pro = np.median(
                np.abs(
                    mean_sample[:, self.dataset.nb_genes :]
                    - self.raw_counts[:, self.dataset.nb_genes :]
                )
            )
            df[m] = [mad_gene, mad_pro]

        df.index = ["genes", "proteins"]
        self.metrics["mad"] = df

    def dropout_ratio(self):
        """Fraction of zeros in raw_counts for a specific gene
        """
        df = pd.DataFrame()
        for m, samples in self.posterior_predictive_samples.items():
            dr = np.mean(np.mean(samples == 0, axis=0), axis=-1)
            df[m] = dr.ravel()

        df["raw"] = np.mean(np.mean(self.raw_counts == 0, axis=0), axis=-1)

        self.metrics["dropout_ratio"] = df

    def store_fa_samples(
        self,
        train_data,
        train_indices,
        test_indices,
        key="Factor Analysis",
        normalization="log",
        **kwargs
    ):
        # reconstruction
        if normalization == "log":
            train_data = np.log(train_data + 1)
            data = np.log(self.raw_counts + 1)
            key += " (Log)"
        elif normalization == "log_rate":
            train_data_rna = train_data[:, : self.dataset.nb_genes]
            train_data_pro = train_data[:, self.dataset.nb_genes :]
            train_data_rna = np.log(
                10000 * train_data_rna / train_data_rna.sum(axis=1)[:, np.newaxis] + 1
            )
            train_data_pro = np.log(
                10000 * train_data_pro / train_data_pro.sum(axis=1)[:, np.newaxis] + 1
            )
            train_data = np.concatenate([train_data_rna, train_data_pro], axis=1)
            lib_size_rna = self.raw_counts[:, : self.dataset.nb_genes].sum(axis=1)[
                :, np.newaxis
            ]
            lib_size_pro = self.raw_counts[:, self.dataset.nb_genes :].sum(axis=1)[
                :, np.newaxis
            ]

            data = np.concatenate(
                [
                    np.log(
                        10000
                        * self.raw_counts[:, : self.dataset.nb_genes]
                        / lib_size_rna
                        + 1
                    ),
                    np.log(
                        10000
                        * self.raw_counts[:, self.dataset.nb_genes :]
                        / lib_size_pro
                        + 1
                    ),
                ],
                axis=1,
            )
            key += " (Log Rate)"
        else:
            train_data = train_data
            data = self.raw_counts
        fa = FactorAnalysis(**kwargs)
        fa.fit(train_data)
        self.models[key] = fa

        # transform gives the posterior mean
        z_mean = fa.transform(data)
        Ih = np.eye(len(fa.components_))
        # W is n_components by n_features, code below from sklearn implementation
        Wpsi = fa.components_ / fa.noise_variance_
        z_cov = linalg.inv(Ih + np.dot(Wpsi, fa.components_.T))

        # sample z's
        z_samples = np.random.multivariate_normal(
            np.zeros(fa.n_components),
            cov=z_cov,
            size=(self.raw_counts.shape[0], self.n_samples),
        )
        # cells by n_components by posterior samples
        z_samples = np.swapaxes(z_samples, 1, 2)
        # add mean to all samples
        z_samples += z_mean[:, :, np.newaxis]

        x_samples = np.zeros(
            (self.raw_counts.shape[0], self.raw_counts.shape[1], self.n_samples)
        )
        for i in range(self.n_samples):
            x_mean = np.matmul(z_samples[:, :, i], fa.components_)
            x_sample = np.random.normal(x_mean, scale=np.sqrt(fa.noise_variance_))
            # add back feature means
            x_samples[:, :, i] = x_sample + fa.mean_

        reconstruction = x_samples

        if normalization == "log":
            reconstruction = np.exp(reconstruction - 1)
        if normalization == "log_rate":
            reconstruction = np.concatenate(
                [
                    lib_size_rna[:, :, np.newaxis]
                    / 10000
                    * np.exp(reconstruction[:, : self.dataset.nb_genes] - 1),
                    lib_size_pro[:, :, np.newaxis]
                    / 10000
                    * np.exp(reconstruction[:, self.dataset.nb_genes :] - 1),
                ],
                axis=1,
            )

        self.posterior_predictive_samples[key] = reconstruction

    def store_pca_samples(self, key="PCA", normalization="log", **kwargs):
        # reconstruction
        if normalization == "log":
            data = np.log(self.raw_counts + 1)
            key += " (Log)"
        else:
            data = self.raw_counts
        pca = PCA(**kwargs)
        pca.fit(data)
        self.models[key] = pca

        # Using Bishop notation section 12.2, M is comp x comp
        # W is fit using MLE, samples generated using posterior predictive
        M = (
            np.matmul(pca.components_, pca.components_.T)
            + np.identity(pca.n_components) * pca.noise_variance_
        )
        z_mean = np.matmul(
            np.matmul(linalg.inv(M), pca.components_), (self.raw_counts - pca.mean_).T
        ).T
        z_cov = linalg.inv(M) * pca.noise_variance_

        # sample z's
        z_samples = np.random.multivariate_normal(
            np.zeros(pca.n_components),
            cov=z_cov,
            size=(self.raw_counts.shape[0], self.n_samples),
        )
        # cells by n_components by posterior samples
        z_samples = np.swapaxes(z_samples, 1, 2)
        # add mean to all samples
        z_samples += z_mean[:, :, np.newaxis]

        x_samples = np.zeros(
            (self.raw_counts.shape[0], self.raw_counts.shape[1], self.n_samples)
        )
        for i in range(self.n_samples):
            x_mean = np.matmul(z_samples[:, :, i], pca.components_)
            x_sample = np.random.normal(x_mean, scale=np.sqrt(pca.noise_variance_))
            # add back feature means
            x_samples[:, :, i] = x_sample + pca.mean_

        reconstruction = x_samples

        if normalization == "log":
            reconstruction = np.clip(reconstruction, -1000, 20)
            reconstruction = np.exp(reconstruction - 1)

        self.posterior_predictive_samples[key] = reconstruction

    def protein_gene_correlation(self, n_genes=1000):
        self.gene_set = np.random.choice(
            self.dataset.nb_genes, size=n_genes, replace=False
        )
        model_corrs = {}
        for m, samples in tqdm(self.posterior_predictive_samples.items()):
            correlation_matrix = np.zeros((n_genes, len(self.dataset.protein_names)))
            for i in range(self.n_samples):
                sample = StandardScaler().fit_transform(samples[:, :, i])
                gene_sample = sample[:, self.gene_set]
                protein_sample = sample[:, self.dataset.nb_genes :]

                correlation_matrix += np.matmul(gene_sample.T, protein_sample)
            correlation_matrix /= self.n_samples
            correlation_matrix /= self.raw_counts.shape[0] - 1
            model_corrs[m] = correlation_matrix.ravel()

        scaled_raw_counts = StandardScaler().fit_transform(self.raw_counts)
        scaled_genes = scaled_raw_counts[:, self.gene_set]
        scaled_proteins = scaled_raw_counts[:, self.dataset.nb_genes :]
        raw_count_corr = np.matmul(scaled_genes.T, scaled_proteins)
        raw_count_corr /= self.raw_counts.shape[0] - 1
        model_corrs["raw"] = raw_count_corr.ravel()

        model_corrs["protein_names"] = list(self.dataset.protein_names) * n_genes
        model_corrs["gene_names"] = np.repeat(
            self.dataset.gene_names[self.gene_set], len(self.dataset.protein_names)
        )

        df = pd.DataFrame.from_dict(model_corrs)
        self.metrics["all protein-gene correlations"] = df

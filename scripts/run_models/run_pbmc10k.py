import torch
from scvi.dataset import Dataset10X
from scvi.models import TOTALVI
from scvi.inference import TotalPosterior, TotalTrainer

save_path = "../../data/10X"


def main(use_cuda=True, model_save_path="../../saved_models"):
    """Runs totalVI on PBMC10k and save model

    Keyword Arguments:
        use_cuda {bool} -- Whether or not to use cuda for training (default: {True})
        model_save_path {str} -- Save path for model after training
    """
    dataset = Dataset10X(
        dataset_name="pbmc_10k_protein_v3",
        save_path=save_path,
        measurement_names_column=1,
    )
    dataset.subsample_genes(new_n_genes=5000)
    dataset.filter_genes_by_count(100)
    # Filter control proteins
    dataset.protein_expression = dataset.protein_expression[:, :-3]
    dataset.protein_names = dataset.protein_names[:-3]

    totalvae = TOTALVI(dataset.nb_genes, len(dataset.protein_names))

    use_cuda = use_cuda
    lr = 1e-2
    early_stopping_kwargs = {
        "early_stopping_metric": "elbo",
        "save_best_state_metric": "elbo",
        "patience": 50,
        "threshold": 0,
        "reduce_lr_on_plateau": True,
        "lr_patience": 20,
        "lr_factor": 0.5,
        "posterior_class": TotalPosterior,
    }

    trainer = TotalTrainer(
        totalvae,
        dataset,
        train_size=0.90,
        test_size=0.05,
        use_cuda=use_cuda,
        frequency=1,
        early_stopping_kwargs=early_stopping_kwargs,
    )

    n_epochs = 300
    trainer.train(n_epochs=n_epochs, lr=lr)

    torch.save(
        {"model_state_dict": totalvae.state_dict()},
        model_save_path + "pbmc10k_totalVI.pt",
    )


if __name__ == "__main__":
    main()

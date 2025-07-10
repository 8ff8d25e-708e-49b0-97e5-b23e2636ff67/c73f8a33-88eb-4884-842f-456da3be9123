from torch import optim
import torch
import networkx as nx
import matplotlib.pyplot as plt
from pykeen.triples import TriplesFactory
import pandas as pd
from pykeen.training import SLCWATrainingLoop
from pykeen.training.callbacks import TrainingCallback


def get_stats(factory):
    return {
        'Entities': factory.num_entities,
        'Relations': factory.num_relations,
        'Triples': factory.num_triples
    }


def summarise_triples_factory(training_factory, testing_factory, validation_factory):
    """Print a summary of the triples factory."""
    data = {
        'Training': get_stats(training_factory),
        'Testing': get_stats(testing_factory),
        'Validation': get_stats(validation_factory)
    }
    df = pd.DataFrame(data).T

    df.index.name = f'Risk to Feature Mapping'

    return df


class LossTrackingCallback(TrainingCallback):
    def __init__(self, val_factory, training_evaluator):
        super().__init__()
        self.train_losses = []
        self.val_ranks = []
        self.val_factory = val_factory
        self.evaluator = training_evaluator

    def on_epoch(self, epoch, epoch_loss):
        # Save training loss
        self.train_losses.append(epoch_loss)

        # Compute mean rank on validation set
        val_result = self.evaluator.evaluate(
            model=self.training_loop.model,
            mapped_triples=self.val_factory.mapped_triples,
            additional_filter_triples=self.training_loop.triples_factory.mapped_triples,
            use_tqdm=False,
        )
        mean_rank = val_result.get_metric('mean_rank')
        self.val_ranks.append(mean_rank)

        print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Val Mean Rank = {mean_rank:.2f}")


if __name__ == "__main__":
    triples_df = pd.read_csv('./data/csvs/triples.csv/part-00000-58e2c5f0-1d2a-4863-8450-cc108a413777-c000.csv')

    triples_factory = TriplesFactory.from_labeled_triples(triples=triples_df[['from', 'edge', 'to']].values)

    training, testing, validation = triples_factory.split([.8, .1, .1])

    print(summarise_triples_factory(training, testing, validation))

    all_triples = list(training.triples) + list(testing.triples) + list(validation.triples)
    graph_triples, _ = triples_factory.split([.2, .8])

    training_triples = training.mapped_triples
    validation_triples = validation.mapped_triples
    test_triples = testing.mapped_triples

    print("Training Triples:\n", training_triples[:5])
    print("Validation Triples:\n", validation_triples[:5])
    print("Test Triples:\n", test_triples[:5])

    training_triples = training.triples
    validation_triples = validation.triples
    test_triples = testing.triples

    print("\n\nTraining Triples with 'Pre Hypertension' risk factor:")
    for triple in training_triples:
        if triple[0] == "Pre Hypertension":
            print(triple)
    print("\nValidation Triples with 'Pre Hypertension' risk factor:")
    for triple in validation_triples:
        if triple[0] == "Pre Hypertension":
            print(triple)
    print("\nTest Triples with 'tourism' relationship:")
    for triple in test_triples:
        if triple[0] == "Pre Hypertension":
            print(triple)

    results = {
        'mean_rank': 0.0,
        'hits@10': 0.0,
        'hits@3': 0.0,
        'hits@1': 0.0,
        'mean_reciprocal_rank': 0.0,
    }
    from pykeen.models import TransE
    from pykeen.models import RotatE
    from pykeen.losses import SoftplusLoss
    from pykeen.losses import MarginRankingLoss

    embedding_dim = 1000
    # from pykeen.models import ComplEx
    # model = TransE(triples_factory=training, random_seed=35, embedding_dim=embedding_dim, loss=MarginRankingLoss())
    model = RotatE(triples_factory=training, random_seed=35, embedding_dim=483, loss=MarginRankingLoss(margin=1.0))
    # model = RotatE(triples_factory=training, random_seed=35, embedding_dim=2000)
    # model = ComplEx(triples_factory=training, random_seed=35, embedding_dim=1000)


    # optimizer = optim.Adam(params=model.get_grad_params())
    # optimizer = optim.SGD(params=model.get_grad_params(), lr=0.02)
    optimizer = optim.Adagrad(params=model.get_grad_params(), lr=0.028)
    # optimizer = optim.SGD(params=model.get_grad_params(), lr=0.001)
    from pykeen.evaluation import RankBasedEvaluator
    evaluator = RankBasedEvaluator()

    callback = LossTrackingCallback(validation_triples, evaluator)
    from pykeen.sampling import BasicNegativeSampler
    negative_sampler = BasicNegativeSampler(
        mapped_triples=training.mapped_triples,
        num_negs_per_pos=1,
    )

    # lets start with SLCWA.  We can use LCWA
    from pykeen.training import SLCWATrainingLoop
    training_loop = SLCWATrainingLoop(
        model=model,
        optimizer=optimizer,
        triples_factory=training,
        negative_sampler=negative_sampler
    )

    training_loop.train(
        triples_factory=training,
        num_epochs=840,
        batch_size=256,
        callbacks="evaluation-loop",
        callbacks_kwargs=dict(
            prefix='validation',
            factory=validation,
        ),
    )
    # Pick an evaluator
    mapped_triples_validation = validation.mapped_triples
    validation_results = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples_validation,
        batch_size=64,
        additional_filter_triples=[
            training.mapped_triples,
            validation.mapped_triples
        ]
    )

    # source: https://pykeen.readthedocs.io/en/stable/api/pykeen.checkpoints.save_model.html
    cortisol_feature_path = "./embeddings/assymetric-features.pt"
    # save model's weights to a file
    torch.save(model.state_dict(), cortisol_feature_path)

    results = {
        'mean_rank': validation_results.get_metric('mean_rank'),
        'hits@10': validation_results.get_metric('hits@10'),
        'hits@3': validation_results.get_metric('hits@3'),
        'hits@1': validation_results.get_metric('hits@1'),
        'mean_reciprocal_rank': validation_results.get_metric('mean_reciprocal_rank'),
    }
    stats = {
        'model': 'RotateE',
        'dim:': embedding_dim,
        'optim': 'Adagrad',
        'negative_sampler': 'BasicNegativeSampler',
        'neg_samples': 1,
        'hits@10': results['hits@10'],
        'num_epochs': 840,
        'batch_size': 256,
    }
    print("Validation Results:\n", validation_results)
    print("Mean Rank:", validation_results.get_metric('mean_rank'))
    #%%

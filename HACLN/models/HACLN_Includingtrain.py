import os
import time
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import get_config
from models import HGNN
import utils.hypergraph_utils as hgut


def compare_loss(representations, label, temperature=0.5):
    """
    Compute supervised contrastive loss on a batch of representations.
    :param representations: Tensor of shape [batch_size, feature_dim]
    :param label: Tensor of shape [batch_size]
    :param temperature: temperature scalar for scaling similarities
    """
    n = label.shape[0]
    # Cosine similarity matrix
    sim_mat = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=2
    )
    # Positive and negative masks
    mask_pos = label.expand(n, n).eq(label.expand(n, n).t()).float()
    mask_eye = 1 - torch.eye(n, device=representations.device)
    sim_exp = torch.exp(sim_mat / temperature) * mask_eye
    # Numerator: positive pairs
    pos_exp = sim_exp * mask_pos
    # Denominator: all pairs
    sum_all = sim_exp.sum(dim=1, keepdim=True)
    # Compute loss
    loss = -torch.log((pos_exp + 1e-12) / sum_all)
    return loss.sum() / (2 * n)


def train_model(model, criterion, optimizer, scheduler,
                 features, graph, labels, idx_train, idx_test,
                 num_epochs=100, print_freq=50):
    """
    Train and validate the HGNN model using supervised contrastive loss.
    :return: trained_model, best_metrics
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0

    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    features = features.to(device)
    graph = graph.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_test = idx_test.to(device)

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print(f"Epoch {epoch}/{num_epochs - 1}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            if phase == 'train':
                scheduler.step()
                optimizer.zero_grad()

            # Forward
            outputs, _ = model(features, graph)
            loss_ce = criterion(outputs[idx_train if phase=='train' else idx_test],
                                labels[idx_train if phase=='train' else idx_test])
            loss_con = compare_loss(outputs[idx_train if phase=='train' else idx_test],
                                    labels[idx_train if phase=='train' else idx_test])
            loss = loss_ce + loss_con

            if phase == 'train':
                loss.backward()
                optimizer.step()

            # Metrics can be computed here (accuracy, AUC, etc.)

        # Update best model based on validation metrics if desired

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    model.load_state_dict(best_model_wts)
    return model


def main():
    # Load configuration
    cfg = get_config('config/config.yaml')

    # Placeholder for data loading
    # features, graph, labels, idx_train, idx_test = load_your_data()

    # Build hypergraph from features if needed
    # H = hgut.construct_H_with_KNN(features.numpy(), K_neigs=cfg['knn'])
    # graph = torch.tensor(hgut.generate_G_from_H(H), dtype=torch.float32)

    # Instantiate model
    model = HGNN(
        in_ch=features.shape[1],
        n_class=cfg['num_classes'],
        n_hid=cfg['n_hid'],
        dropout=cfg['drop_out']
    )

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    trained_model = train_model(
        model, criterion, optimizer, scheduler,
        features, graph, labels, idx_train, idx_test,
        num_epochs=cfg['max_epoch'], print_freq=cfg['print_freq']
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()

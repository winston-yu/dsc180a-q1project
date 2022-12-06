import random
from tqdm import tqdm

def report(targets, preds, mask):
    """
    Returns the precision and recall of predictions (preds) whose ground truths are "targets" at
    the indices specified in mask. Necessary due to the finicky nature of pytorch-geometric's 
    Data data structure.
    """
    edge_rows, edge_cols = targets[mask].nonzero().t()
    tp = len(targets[mask].nonzero()[:, 0])
    fp = len((targets[mask][~edge_rows, ~edge_cols] != preds[mask][~edge_rows, ~edge_cols].round()).nonzero())
    fn = len((targets[mask][edge_rows, edge_cols] != preds[mask][edge_rows, edge_cols].round()).nonzero())
    prec, recall = tp / (tp + fp), tp / (tp + fn)
    return prec, recall

def sample_edges(edge_index, mask, strict=False):
    """
    Generates 50:50 positive:negative samples from edge_index at the indices specified in mask.
    
    strict: if true, then edges must have both source and destination in mask. Otherwise, only 
    at least one of the source or the destination need be in mask.
    """
    pos_samples, neg_samples = [], []
    if strict: candidates = mask.nonzero().squeeze()
    for node in tqdm(mask.nonzero()):
        neighbors = edge_index[:, edge_index[0] == node][1]
        not_neighbors = edge_index[:, edge_index[0] != node][1]
        if strict:
            neighbors = [n for n in neighbors if n in candidates]
            not_neighbors = [n for n in not_neighbors if n in candidates]

        num_neighbors = len(neighbors)
        if num_neighbors > 0:
            pos = neighbors[random.sample(range(num_neighbors), 1)[0]].item()
            pos_samples.append([node.item(), pos])
        
        num_not_neighbors = len(not_neighbors)
        if num_not_neighbors > 0:
            neg = not_neighbors[random.sample(range(num_not_neighbors), 1)[0]].item()
            neg_samples.append([node.item(), neg])

    return pos_samples, neg_samples
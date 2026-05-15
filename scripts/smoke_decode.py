import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from generation.postprocess import decode_graph_batch


def main():
    max_nodes = 4
    node_dim = 36
    edge_types = 4

    node_logits = torch.zeros(1, max_nodes, node_dim)
    edge_logits = torch.zeros(1, max_nodes, max_nodes, edge_types)

    node_logits[0, 0, 0] = 5.0
    node_logits[0, 1, 0] = 5.0
    edge_logits[0, 0, 1, 1] = 5.0
    edge_logits[0, 1, 0, 1] = 5.0

    decoded = decode_graph_batch(node_logits, edge_logits, max_nodes)
    valid = decoded[0]['valid']
    print(decoded)
    if not valid:
        raise SystemExit('Expected smoke decode to produce a valid molecule')


if __name__ == '__main__':
    main()

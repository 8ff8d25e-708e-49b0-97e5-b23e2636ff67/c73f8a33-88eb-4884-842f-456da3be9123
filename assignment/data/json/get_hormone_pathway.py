import pandas as pd


def get_hormone_pathway(path, node_path, edge_path):
    # Prepare table
    df = pd.read_json(path)
    df = pd.json_normalize(df['entitiesById'])
    df.drop(columns=['drawAs', 'fill', 'points', 'stroke', 'strokeWidth', 'zIndex', 'fontFamily', 'fontSize', 'fontStyle', 'fontWeight', 'height', 'lineHeight', 'overflow', 'padding', 'textAlign', 'textDecoration', 'textOverflow', 'verticalAlign', 'whiteSpace', 'width', 'x', 'y', 'rx', 'ry', 'attachmentDisplay.offset', 'attachmentDisplay.position', 'contains', 'fillOpacity', 'strokeDasharray', 'citations', 'isPartOf'], inplace=True)
    df = df[df['gpmlElementName'] != 'PublicationXref']
    df = df[df['gpmlElementName'] != 'openControlledVocabulary']
    df = df[df['gpmlElementName'].notnull()]

    # Nodes
    df_nodes = df[df['gpmlElementName'] == 'DataNode'].dropna(axis=1, how='all')
    df_nodes = df_nodes[df_nodes['wpType'] != 'Pathway']
    df_nodes = df_nodes[df_nodes['wpType'] != 'GeneProduct']
    df_nodes = df_nodes[df_nodes['wpType'] != 'Protein']
    df_nodes.drop(columns=['gpmlElementName', 'kaavioType', 'type', 'xrefDataSource', 'wpType'], inplace=True)

    # Edges
    df_edges = df[df['gpmlElementName'] == 'Interaction'].dropna(axis=1, how='all')
    df_edges.drop(columns=['gpmlElementName', 'kaavioType', 'markerEnd', 'type', 'xrefDataSource', 'xrefIdentifier', 'burrs', 'comments'], inplace=True)
    df_edges = df_edges.explode('isAttachedTo', ignore_index=True)

    df_nodes.to_csv(node_path, index=False)
    df_edges.to_csv(edge_path, index=False)

    return {
        'nodes': df_nodes.to_csv(index=False),
        'edges': df_edges.to_csv(index=False)
    }


if __name__ == '__main__':
    get_hormone_pathway('./WP5277.json', './precursor-nodes.csv', './precursor-edges.csv')
    get_hormone_pathway('./WP4523.json', './classic-nodes.csv', './classic-edges.csv')
    get_hormone_pathway('./WP5280-gluco.json', './glucocorticoid-nodes.csv', './glucocorticoid-edges.csv')
    get_hormone_pathway('./WP5279-mineralo.json', './mineralocorticoid-nodes.csv', './mineralocorticoid-edges.csv')




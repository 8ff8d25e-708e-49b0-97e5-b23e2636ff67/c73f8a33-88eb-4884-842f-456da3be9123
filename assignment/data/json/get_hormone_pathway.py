import pandas as pd

id_dictionary = {
    'a9453': 'bb4067dc-9d17-44ad-b30d-9d4055ea4c30',
    'aa516': '6b539c79-feb0-429d-bdb1-66fa6a826c7d',
    'b6cb0': '5544c99e-eaa7-4c01-aae5-f4489688c0ad',
    'b8f6e': '5108b288-a0bc-4e12-bc22-870a8ef9789d',
    'bb620': '5bb70a76-a68d-42f5-95e1-c0c3fe844963',
    'c7413': 'e24296df-5350-4657-b3de-7182a580c9e4',
    'c798c': '7fa16bfd-eb60-4e59-884c-a8fb91a24e29',
    'cda05': 'd3d47191-a2c9-49da-b3f5-85909530b1c6',
    'ce91d': '7fca6ef2-8836-409b-a958-5287134aee49',
    'd9040': '6b539c79-feb0-429d-bdb1-66fa6a826c7d',
    'd9a28': 'a21ef33b-5df9-4525-89ed-41347cc3875b',
    'e5c3a': '7fca6ef2-8836-409b-a958-5287134aee49',
    'ed90c': 'b04d90ba-5750-4666-a8ca-e02bbc4e2ec5',
    'f2829': '7f966adf-7c0c-4ac5-ad21-5285f12613b1',
    'f650a': '986c2526-bd34-4c47-bde0-1ce1b9e3dd2a'
}

text_content_dictionary = {
    'Dihydrotestosterone': 'bb4067dc-9d17-44ad-b30d-9d4055ea4c30',
    'Testosterone': '6b539c79-feb0-429d-bdb1-66fa6a826c7d',
    '17-hydroxyprogesterone': '5544c99e-eaa7-4c01-aae5-f4489688c0ad',
    'Cortisone': '5108b288-a0bc-4e12-bc22-870a8ef9789d',
    'Progesterone': '5bb70a76-a68d-42f5-95e1-c0c3fe844963',
    'Corticosterone': 'e24296df-5350-4657-b3de-7182a580c9e4',
    '11-Deoxycortisol': '7fa16bfd-eb60-4e59-884c-a8fb91a24e29',
    'DHEA': 'd3d47191-a2c9-49da-b3f5-85909530b1c6',
    'Cortisol': '7fca6ef2-8836-409b-a958-5287134aee49',
    'Androstenedione': '6b539c79-feb0-429d-bdb1-66fa6a826c7d',
    '(11)-Deoxycorticosterone': 'a21ef33b-5df9-4525-89ed-41347cc3875b',
    'Cortisol': '7fca6ef2-8836-409b-a958-5287134aee49',
    '18-hydroxycorticosterone': 'b04d90ba-5750-4666-a8ca-e02bbc4e2ec5',
    'Urocortisol': '7f966adf-7c0c-4ac5-ad21-5285f12613b1',
    'Aldosterone': '986c2526-bd34-4c47-bde0-1ce1b9e3dd2a'
}


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
    df_edges[['id', 'isAttachedTo']] = pd.DataFrame(df_edges['isAttachedTo'].tolist(), index=df_edges.index)
    # df_edges = df_edges.explode('isAttachedTo', ignore_index=True)


    return {
        'nodes': df_nodes.to_csv(index=False),
        'edges': df_edges.to_csv(index=False)
    }


if __name__ == '__main__':
    get_hormone_pathway('./WP5277.json', './precursor-nodes.csv', './precursor-edges.csv')
    get_hormone_pathway('./WP4523.json', './classic-nodes.csv', './classic-edges.csv')
    get_hormone_pathway('./WP5280-gluco.json', './glucocorticoid-nodes.csv', './glucocorticoid-edges.csv')
    get_hormone_pathway('./WP5279-mineralo.json', './mineralocorticoid-nodes.csv', './mineralocorticoid-edges.csv')




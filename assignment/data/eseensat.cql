CREATE
  (adenoma:Group {groupName: "Adrenocortical adenoma or macronodular bilateral adrenal hyperplasia ", groupID: 1}),
  (benign:Group {groupName: "Other benign mass ", groupID: 2}),
  (pheochromocytoma:Group {groupName: "Pheochromocytoma ", groupID: 3}),
  (carcinoma:Group {groupName: "Adrenocortical carcinoma ", groupID: 4}),
  (malignantMass:Group {groupName: "Other malignant mass (mostly adrenal metastases) ", groupID: 5}),
  (nonFunctioning:Etiology {etiologyName: "Nonfunctioning ", etiologyID: 1}),
  (macs:Etiology {etiologyName: "Mild autonomous cortisol  secretion", etiologyID: 2}),
  (primaryAldo:Etiology {etiologyName: "Primary aldosteronism ", etiologyID: 3}),
  (cushing:Etiology {etiologyName: "Overt Cushing’s syndrome ", etiologyID: 4}),
  (myelolipoma:Etiology {etiologyName: "Myelolipoma ", etiologyID: 5}),
  (cyst:Etiology {etiologyName: "Cyst and pseudocyst ", etiologyID: 6}),
  (ganglio:Etiology {etiologyName: "Ganglioneuroma", etiologyID: 7}),
  (schwann:Etiology {etiologyName: "Schwannoma", etiologyID: 8}),
  (hemorrhage:Etiology {etiologyName: "Hemorrhage ", etiologyID: 9}),
  (nonFunctioning)-[:PART_OF]->(adenoma),
  (macs)-[:PART_OF]->(adenoma),
  (primaryAldo)-[:PART_OF]->(adenoma),
  (cushing)-[:PART_OF]->(adenoma),
  (ganglio)-[:PART_OF]->(benign),
  (cyst)-[:PART_OF]->(benign),
  (myelolipoma)-[:PART_OF]->(benign),
  (schwann)-[:PART_OF]->(benign),
  (hemorrhage)-[:PART_OF]->(benign);

MATCH (n:Group)
MATCH (et:Etiology)
OPTIONAL MATCH (et)-[r:PART_OF]->(n)
RETURN n,et,r;
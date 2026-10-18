# data/

Put the two datasets here. Nothing in this folder is uploaded to GitHub.

```
data/
├── helmets/                               our helmet photos (from the project team)
│   ├── helmets_metadata.csv
│   └── Cartella DATASET Algoritmo convegno/
└── VOC2028/                               SHWD, the public helmet dataset
    ├── Annotations/
    ├── ImageSets/Main/
    └── JPEGImages/
```

The notebooks create these folders:
- `helmets_augmented/` (notebook 03): 1,750 augmented helmet images and `manifest.csv`.
- `features/` (notebook 05): one ResNet50 vector per SHWD head, and the head table.

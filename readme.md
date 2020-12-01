# Plan 

Build a knowledge graph of relations between characters, and map the relations between character age groups based on learnt relation types. 

# Overview

- `parse_relations.py` parses the seven novels for annotated entities and writes them as a pandas DataFrame.
- `learn_relations.py` implements 3 clustering algorithms (Kmeans, cosine similarity clusterings, & Umap / HDBscan). The latter two are based on BERT sentence embeddings.
- `build_networks.py` combines text, predictions, and character information, to build a network representation of entity relations.
- `train_ner.py` is currently unstarted, but will train a Name Entity Recognition pipeline.
- `serve_app.py` acts as a model prediction backend for the visualiser, but is not necessary for basic representations. 
- `/app` contains a React based frontend for creating the d3 network visualisation that can be seen in the link below. 

# Demo

[Visualisation Demo](https://pottergraph.herokuapp.com/)

# TODO :

- [x] parse entity sentences using annotation and neural corefernece
- [x] embed using bert 
- [x] cluster using umap
  - [ ] refine cluster quality
- [x] represent relations
  - [ ] add relation queries
- [x] build network
- [x] create representation of cluster relations
    - [x] group ages
    - [x] filter relations
    - [x] highlight entities
- [x] visualise networks of relations by age / character  
  - [ ] add functionality to browse predictions
  - [ ] add annotation options
- [ ] add named entity recognition
  - [ ] redo the pipeline on annotated text

<!-- # Questions for Lindsey

have you down any pos / coref work before?

do you have other entities or just characters

have you annotated any other books in harry potter series? -->


# References used

```
@article{mcinnes2017hdbscan,
  title={hdbscan: Hierarchical density based clustering},
  author={McInnes, Leland and Healy, John and Astels, Steve},
  journal={The Journal of Open Source Software},
  volume={2},
  number={11},
  pages={205},
  year={2017}
}

@inproceedings{mcinnes2017accelerated,
  title={Accelerated Hierarchical Density Based Clustering},
  author={McInnes, Leland and Healy, John},
  booktitle={Data Mining Workshops (ICDMW), 2017 IEEE International Conference on},
  pages={33--42},
  year={2017},
  organization={IEEE}
}

@article{mcinnes2018umap-software,
  title={UMAP: Uniform Manifold Approximation and Projection},
  author={McInnes, Leland and Healy, John and Saul, Nathaniel and Grossberger, Lukas},
  journal={The Journal of Open Source Software},
  volume={3},
  number={29},
  pages={861},
  year={2018}
}

@article{2018arXivUMAP,
     author = {{McInnes}, L. and {Healy}, J. and {Melville}, J.},
     title = "{UMAP: Uniform Manifold Approximation
     and Projection for Dimension Reduction}",
     journal = {ArXiv e-prints},
     archivePrefix = "arXiv",
     eprint = {1802.03426},
     primaryClass = "stat.ML",
     keywords = {Statistics - Machine Learning,
                 Computer Science - Computational Geometry,
                 Computer Science - Learning},
     year = 2018,
     month = feb,
}

@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```
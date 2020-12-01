# Plan 

Build a knowledge graph of relations between characters, and map the relations between character age groups based on learnt relation types. 

# TODO :

- [x] parse entity sentences using annotation and neural corefernece
- [x] embed using bert 
- [x] cluster using umap
- [x] represent relations
- [x] build network
- [ ] create representation of cluster relations
    - [ ] group ages
    - [ ] filter relations
    - [ ] highlight entities
- [ ] visualise networks of relations by age / character  
- [ ] add named entity recognition

<!-- # Questions for Lindsey

have you down any pos / coref work before?

do you have other entities or just characters

have you annotated any other books in harry potter series? -->


# references

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
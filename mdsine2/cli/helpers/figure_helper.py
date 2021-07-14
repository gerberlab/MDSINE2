import numpy as np
import pandas as pd

def get_hierachy(taxo):
    """
       returns the lowest defined hierarchy

       @parameters
       taxo : (pl.Taxa)
    """

    name = ""
    if taxo["species"] != "NA":
        return taxo["genus"] + " " + taxo["species"]
    elif taxo["genus"] != "NA":
        return "* " +  taxo["genus"]
    elif taxo["family"] != "NA":
        return "** " + taxo["family"]
    elif taxo["order"] != "NA":
        return "*** " + taxo["order"]
    elif taxo["class"] != "NA":
        return "**** " + taxo["class"]
    elif taxo["phylum"] != "NA":
        return "***** " + taxo["phylum"]
    else:
        return "****** " + taxo["kingdom"]

def get_taxa_names(otu_li, taxas):
    """
       returns the taxa names of OTUs in otu_li in the form of . The names are generated as per
       RDP classifier

       @parameters
       otu_li : ([str]) list of OTU ids
       taxas : (pl.Taxaset)

       @Returns
       (dict) (str) otu name -> (str) taxonomic name
    """

    names = {}
    for otu in otu_li:
        taxonomy = taxas[otu]
        hierachy = get_hierachy(taxonomy)
        names[otu] = hierachy

    return names

def get_axes_names(order_li, cluster, subjset):
    """
      gets the axes tick names

       @parameters
       otu_li : ([str]) list of OTU ids
       cluster : (dict) (int) cluster_id -> [str] otus in the cluster
       subjset : (md2.Subject)
    """
    set_ = []
    for otu in subjset.taxa:
        set_.append(len(otu.sequence))
        #print(otu.sequence)
    taxa_names = get_taxa_names(order_li, subjset.taxa)
    otu_cluster_d = {otu : id for id in cluster for otu in cluster[id]}
    y_ = [taxa_names[order_li[i - 1]] + " | " + order_li[i - 1].replace("_", " ")
     + " | " + str(otu_cluster_d[order_li[i - 1]]) + " | " + str(i) for i in
     range(1, len(order_li) + 1)]

    x_ = [str(i) for i in range(1, len(order_li) + 1)]

    return x_, y_

def reorder(matrix, otu_li, order_d):
    """
       reorders the data matrix according to the order in otu_li

       @parameters
       otu_li : ([str]) list of OTU ids
       order_d : (dict (str) otu_id -> (int) index of the id in matrix)

       @returns
       (np.array) reordered matrix
    """

    shape = matrix.shape
    new_mat = np.zeros(shape)
    for i in range(len(otu_li)):
        for j in range(len(otu_li)):
            new_mat[i, j] = matrix[order_d[otu_li[i]], order_d[otu_li[j]]]

    return new_mat

def order_otus(otu_li):
    """sort the OTUs according to their id

       otu_li : [str] list of OTU ids

       @returns
       list [str], dict {(str) -> int}

    """
    ordered = np.sort([int(x.split("_")[1]) for x in otu_li])
    ordered = ["OTU_" + str(id) for id in ordered]
    ordered_dict = {ordered[i] : i for i in range(len(ordered))}

    return ordered, ordered_dict

def parse_cluster(filename):
    """
       parses the .tsv file contaning the parse_cluster

       @parameters
       ------------------------------------------------------------------------
       filename : name of the file containing the cluster information

       @returns
       (dict (int) cluster_id -> [str] ids of OTU in the cluster)
    """

    cluster_arr = pd.read_csv(filename, sep = "\t", index_col = None).to_numpy()
    dict_ = {}
    for row in cluster_arr:
        id = row[1]
        if id not in dict_:
            dict_[id] = [row[0]]
        else:
            dict_[id].append(row[0])

    return dict_

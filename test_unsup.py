# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka

import logging
from SimCSERetrieval import SimCSERetrieval


def main():
    fname = r".\data\gsp.txt"
    pretrained = r".\pre_model\bert-wwm"  # 下载的预训练模型
    simcse_model = r".\model\gsp-epoch-3.pt"
    batch_size = 64
    max_length = 100
    device = "cpu"

    logging.info("Load model")
    simcse = SimCSERetrieval(fname, pretrained, simcse_model, batch_size, max_length, device)

    logging.info("Sentences to vectors")
    simcse.encode_file()

    logging.info("Build faiss index")
    simcse.build_index(nlist=10)
    simcse.index.nprob = 20

    #query_sentence = "基金亏损路未尽 后市看法仍偏谨慎"
    query_sentence = "供应商盖章"
    print("\nquery title:{0}".format(query_sentence))
    print("\nsimilar titles:")
    print(u"\n".join(simcse.sim_query(query_sentence, topK=20)))


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()


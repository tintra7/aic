import faiss
import pandas as pd
import os
import numpy as np
from typing import Union
from flask import Flask, request, jsonify


app = Flask(__name__)

class FaissWrapper:
    def __init__(self, index):
        self.index = index
        self.df = pd.read_csv("frame_id.csv")

    def search(self, feature, k=10):
        D, I = self.index.search(feature, k)
        I = I[0]
        D = D[0]
        valid = I >= 0
        I = I[valid]
        D = D[valid]
        return self.df.iloc[I]['frame_id'].tolist()  # Return as a list

    def load_sample_query(self, query):
        fp = './sift/sift_query.fvecs'
        a = np.fromfile(fp, dtype='int32')
        d = a[0]
        a = a.reshape(-1, d + 1)[:, 1:].copy().view('float32')
        a = a[query]
        return np.array([a])

def load_index():
    index_path = './data/index/faiss.index'
    print("Loading index")
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        
        return FaissWrapper(index)
    
    else:
        raise FileNotFoundError(f"Index file not found at {index_path}")

try:
    index = load_index()
    print("Load index successfull")
except FileNotFoundError as e:
    print(e)
    index = None

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

@app.route("/search", methods=["GET"])
def search_query():
    if not index:
        return jsonify({"error": "Index not loaded"}), 500
    query = int(request.args.get("query"))
    k = int(request.args.get("k", 10))  # default k=10
    encode_vector = index.load_sample_query(query)
    result = index.search(encode_vector, k)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

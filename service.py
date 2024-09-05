import faiss
import pandas as pd
import os
import numpy as np
from typing import Union
from flask import Flask, request, jsonify
import torch
import clip
import time
from pymongo.mongo_client import MongoClient


app = Flask(__name__)

class FaissWrapper:
    def __init__(self, index):
        self.index = index
        self.df = pd.read_csv("frame_id.csv")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        uri = "mongodb+srv://tintra:Tintra2711@bettergpuwin.ghtqx.mongodb.net/?retryWrites=true&w=majority&appName=BetterGPUWin"
        client = MongoClient(uri)
        self.collection = client['vbs']['keyframe']

    def search_with_clip(self, query, k):
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()
        D, I = self.index.search(text_features, k)
        I = I[0]
        D = D[0]
        valid = I >= 0
        I = I[valid]
        D = D[valid]
        return pd.DataFrame({"frame_id": I, "distance": D})  # Fixed the DataFrame

    def fuzzy_matching(self, query: str, k):
        result = self.collection.aggregate([
            {
                "$search": {
                    "index": "default",
                    "text": {
                        "query": query,
                        "path": "text",
                        "fuzzy": {}
                    }   
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "score": { "$meta": "searchScore" }
                }
            }
        ])

        # Convert the cursor to a list of dictionaries
        result = list(result)
        return pd.DataFrame(result).sort_values(by='score', ascending=False).head(k)

    def search(self, query, k=1000):
        df_clip = self.search_with_clip(query=query, k=k)
        df_keyword = self.fuzzy_matching(query=query, k=k)
        df = pd.merge(df_clip, df_keyword, left_on='frame_id', right_on='_id', how='outer').fillna(value=0)
        df['final_score'] = 0.5 * df['score'] + 0.5 * df['distance']
        df = df.sort_values(by='final_score', ascending=False)
        return df['frame_id'].tolist()[:k]  # Return as a list, fixed values() method


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
    print("Load index successful")
except FileNotFoundError as e:
    print(e)
    index = None

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

@app.route("/search", methods=["GET"])
def search_query():
    if not index:
        print("Not found index")
        return jsonify({"error": "Index not loaded"}), 500
    query = request.args.get("query")
    k = int(request.args.get("k", 10))  # default k=10
    start = time.time()
    result = index.search(query, k)
    end = time.time()
    print(f"Search in {end - start} seconds")
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

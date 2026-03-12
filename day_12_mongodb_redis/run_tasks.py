"""
Day 12: Redis Task Queue + MongoDB Result Storage
Author: Sheshikala
Date: 2026-03-12
 
What this does:
- Pushes ML tasks into a Redis queue (LIST)
- Worker picks tasks one by one and runs them
- Results get stored in MongoDB
 
What I learned:
- Redis LIST as queue = very simple but powerful pattern
- RPUSH to add, LPOP to consume = FIFO order
- Storing results in MongoDB makes them searchable later
"""
 
import redis
import json
import time
import random
import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
 
QUEUE_KEY  = "task_queue:ml"
RESULT_KEY = "task_results:ml"
 
 
# ---------------------------------------------
# CONNECTIONS
# ---------------------------------------------
def get_connections():
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("[OK] Redis connected!")
    except redis.ConnectionError:
        print("[ERROR] Redis not running!")
        exit(1)
 
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client["ml_tracker"]
        print("[OK] MongoDB connected!")
    except ConnectionFailure:
        print("[ERROR] MongoDB not running!")
        exit(1)
 
    return r, db
 
 
# ---------------------------------------------
# TASK FUNCTIONS - all inline, no imports
# ---------------------------------------------
def train_model(task):
    model_name    = task.get("model_name", "Unknown")
    epochs        = task.get("epochs", 3)
    learning_rate = task.get("learning_rate", 2e-5)
 
    print("    [TRAIN] " + model_name + " | epochs=" + str(epochs))
    time.sleep(1.5)
 
    base_f1 = task.get("expected_f1", 0.88)
    test_f1 = round(base_f1 + random.uniform(-0.01, 0.02), 4)
 
    return {
        "task_type" : "train",
        "model_name": model_name,
        "status"    : "completed",
        "metrics"   : {
            "test_f1"      : test_f1,
            "test_accuracy": round(test_f1 + 0.005, 4),
            "loss"         : round(random.uniform(0.15, 0.35), 4),
            "epochs_run"   : epochs
        },
        "completed_at": datetime.datetime.now().isoformat()
    }
 
 
def evaluate_model(task):
    model_name = task.get("model_name", "Unknown")
    dataset    = task.get("dataset", "test_set")
 
    print("    [EVAL] " + model_name + " on " + dataset)
    time.sleep(0.8)
 
    base_f1 = task.get("expected_f1", 0.88)
 
    return {
        "task_type" : "evaluate",
        "model_name": model_name,
        "dataset"   : dataset,
        "status"    : "completed",
        "metrics"   : {
            "f1_score"    : round(base_f1 + random.uniform(-0.005, 0.01), 4),
            "precision"   : round(base_f1 + random.uniform(0.0, 0.015), 4),
            "recall"      : round(base_f1 + random.uniform(-0.01, 0.005), 4),
            "inference_ms": round(random.uniform(10, 150), 1)
        },
        "completed_at": datetime.datetime.now().isoformat()
    }
 
 
def generate_embeddings(task):
    model_name = task.get("model_name", "Unknown")
    num_docs   = task.get("num_documents", 100)
 
    print("    [EMBED] " + model_name + " | " + str(num_docs) + " docs")
    time.sleep(1.0)
 
    return {
        "task_type"          : "embed",
        "model_name"         : model_name,
        "status"             : "completed",
        "documents_processed": num_docs,
        "embedding_dim"      : 768,
        "avg_time_ms"        : round(random.uniform(5, 25), 2),
        "completed_at"       : datetime.datetime.now().isoformat()
    }
 
 
TASK_REGISTRY = {
    "train"   : train_model,
    "evaluate": evaluate_model,
    "embed"   : generate_embeddings,
}
 
 
# ---------------------------------------------
# PRODUCER
# ---------------------------------------------
def enqueue_tasks(r):
    print("\n-- Enqueuing Tasks ---------------------")
 
    tasks = [
        {
            "task_id"     : "task_001",
            "task_type"   : "train",
            "model_name"  : "BERT_v2",
            "epochs"      : 3,
            "learning_rate": 2e-5,
            "expected_f1" : 0.90,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
        {
            "task_id"     : "task_002",
            "task_type"   : "train",
            "model_name"  : "RoBERTa_v1",
            "epochs"      : 5,
            "learning_rate": 1e-5,
            "expected_f1" : 0.93,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
        {
            "task_id"     : "task_003",
            "task_type"   : "evaluate",
            "model_name"  : "ResNet50_medical",
            "dataset"     : "medical_test_set",
            "expected_f1" : 0.85,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
        {
            "task_id"     : "task_004",
            "task_type"   : "embed",
            "model_name"  : "sentence-transformers",
            "num_documents": 500,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
        {
            "task_id"     : "task_005",
            "task_type"   : "train",
            "model_name"  : "XGBoost_v3",
            "epochs"      : 1,
            "learning_rate": 0.05,
            "expected_f1" : 0.92,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
    ]
 
    for task in tasks:
        r.rpush(QUEUE_KEY, json.dumps(task))
        print("  Queued: " + task["task_id"] + " | type: " + task["task_type"] + " | model: " + task["model_name"])
 
    print("\n  Total tasks in queue: " + str(r.llen(QUEUE_KEY)))
 
 
# ---------------------------------------------
# CONSUMER
# ---------------------------------------------
def process_tasks(r, db):
    print("\n-- Processing Tasks --------------------")
    col       = db["task_results"]
    completed = 0
    failed    = 0
 
    while r.llen(QUEUE_KEY) > 0:
        raw_task = r.lpop(QUEUE_KEY)
        if not raw_task:
            break
 
        task      = json.loads(raw_task)
        task_id   = task["task_id"]
        task_type = task["task_type"]
 
        print("\n  > Running " + task_id + " (" + task_type + ")...")
 
        try:
            task_fn    = TASK_REGISTRY.get(task_type)
            start_time = time.time()
            result     = task_fn(task)
            elapsed    = round(time.time() - start_time, 2)
 
            result["task_id"]     = task_id
            result["elapsed_sec"] = elapsed
 
            col.insert_one(result)
            r.zadd(RESULT_KEY, {task_id: elapsed})
 
            f1 = result.get("metrics", {}).get("test_f1") or result.get("metrics", {}).get("f1_score", "N/A")
            print("    [OK] Done in " + str(elapsed) + "s | f1: " + str(f1))
            completed += 1
 
        except Exception as e:
            print("    [ERROR] Failed: " + str(e))
            failed += 1
 
    return completed, failed
 
 
# ---------------------------------------------
# RESULTS
# ---------------------------------------------
def show_results(r, db):
    print("\n-- Results Summary ---------------------")
    col   = db["task_results"]
    total = col.count_documents({})
    print("  Results stored in MongoDB: " + str(total))
 
    print("\n  Completed tasks:")
    for doc in col.find({}, {"task_id": 1, "task_type": 1, "model_name": 1, "metrics": 1, "elapsed_sec": 1, "_id": 0}):
        f1 = doc.get("metrics", {}).get("test_f1") or doc.get("metrics", {}).get("f1_score", "N/A")
        print("    " + doc["task_id"] + " | " + doc["task_type"] + " | " + doc["model_name"] + " | f1: " + str(f1) + " | time: " + str(doc.get("elapsed_sec")) + "s")
 
    print("\n  Task timing from Redis (sorted by speed):")
    for task_id, elapsed in r.zrange(RESULT_KEY, 0, -1, withscores=True):
        print("    " + str(task_id) + " -> " + str(elapsed) + "s")
 
 
# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    r, db = get_connections()
 
    r.delete(QUEUE_KEY, RESULT_KEY)
    db["task_results"].drop()
 
    enqueue_tasks(r)
    completed, failed = process_tasks(r, db)
 
    print("\n-- Final Stats -------------------------")
    print("  Completed : " + str(completed))
    print("  Failed    : " + str(failed))
 
    show_results(r, db)
    print("\n[OK] run_tasks.py completed!")
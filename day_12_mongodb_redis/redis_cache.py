"""
Day 12: Redis - All Data Types and Caching Patterns
Author: Sheshikala
Date: 2026-03-12
 
What I learned:
- Redis stores everything in memory so reads/writes are microseconds fast
- Different data types serve different purposes - not just key-value
- TTL is critical - always expire cache keys to avoid memory issues
- Sorted sets are amazing for leaderboards - ranking is built in
- Pub/Sub lets two Python processes talk to each other in real time
- Cache-Aside pattern is the most common caching strategy
 
How to run:
  1. Start Redis: docker run -d -p 6379:6379 --name redis redis
  2. pip install redis
  3. python redis_cache.py
"""
 
import redis
import json
import time
import threading
 
# ---------------------------------------------
# CONNECTION
# ---------------------------------------------
def get_redis():
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("[OK] Connected to Redis!")
        return r
    except redis.ConnectionError:
        print("[ERROR] Redis not running!")
        print("   Run: docker run -d -p 6379:6379 --name redis redis")
        exit(1)
 
def cleanup(r):
    patterns = ["demo:*", "exp:*", "models:*", "f1_lb", "training_q",
                "cache:*", "session:*", "experiment:*", "ml:*"]
    deleted = 0
    for p in patterns:
        keys = r.keys(p)
        for k in keys:
            r.delete(k)
            deleted += 1
    print(f"[CLEAN]  Cleaned {deleted} old keys")
 
 
# ---------------------------------------------
# 1. STRINGS
# simplest type - used for caching single values
# ---------------------------------------------
def demo_strings(r):
    print("\n-- 1. STRINGS --------------------------")
 
    r.set("demo:best_model", "RoBERTa")
    r.set("demo:best_f1",    "0.9360")
 
    print(f"  GET best_model   -> {r.get('demo:best_model')}")
    print(f"  GET best_f1      -> {r.get('demo:best_f1')}")
 
    # set with TTL - key auto-deletes after 60 seconds
    # I learned the hard way not to store cache without TTL - memory fills up
    r.setex("demo:session", 60, "Sheshikala_session")
    print(f"  TTL session      -> {r.ttl('demo:session')} sec")
 
    # atomic counter - no race conditions
    r.set("demo:run_count", 0)
    r.incr("demo:run_count")
    r.incr("demo:run_count")
    r.incr("demo:run_count")
    print(f"  INCR run_count   -> {r.get('demo:run_count')}")
 
    # store JSON object as string - very common pattern
    experiment = {"name": "RoBERTa", "f1": 0.9360, "status": "completed", "epochs": 5}
    r.setex("demo:exp_cache", 300, json.dumps(experiment))
    loaded = json.loads(r.get("demo:exp_cache"))
    print(f"  JSON roundtrip   -> {loaded['name']} f1={loaded['f1']}")
 
 
# ---------------------------------------------
# 2. HASHES
# like a Python dict - store objects field by field
# ---------------------------------------------
def demo_hashes(r):
    print("\n-- 2. HASHES ---------------------------")
 
    r.hset("demo:exp:bert", mapping={
        "name"       : "BERT Sentiment",
        "model_type" : "BERT",
        "f1_score"   : "0.9020",
        "status"     : "completed",
        "researcher" : "Sheshikala"
    })
 
    print(f"  HGET name        -> {r.hget('demo:exp:bert', 'name')}")
    print(f"  HGET f1_score    -> {r.hget('demo:exp:bert', 'f1_score')}")
    print(f"  HGETALL          -> {r.hgetall('demo:exp:bert')}")
    print(f"  HKEYS            -> {r.hkeys('demo:exp:bert')}")
 
    # update one field without touching rest
    r.hset("demo:exp:bert", "status", "archived")
    print(f"  After update     -> status: {r.hget('demo:exp:bert', 'status')}")
 
    r.hdel("demo:exp:bert", "researcher")
    print(f"  After HDEL       -> keys: {r.hkeys('demo:exp:bert')}")
 
 
# ---------------------------------------------
# 3. LISTS
# ordered sequence - perfect for task queues
# ---------------------------------------------
def demo_lists(r):
    print("\n-- 3. LISTS (Task Queue) ---------------")
    r.delete("demo:training_queue")
 
    # push tasks to right end of queue
    tasks = ["train:BERT", "train:RoBERTa", "evaluate:ResNet50", "train:XGBoost", "evaluate:LightGBM"]
    r.rpush("demo:training_queue", *tasks)
 
    print(f"  Queue length     -> {r.llen('demo:training_queue')}")
    print(f"  All tasks        -> {r.lrange('demo:training_queue', 0, -1)}")
 
    # LPOP removes and returns from left = FIFO queue
    print("  Processing queue:")
    while r.llen("demo:training_queue") > 0:
        task = r.lpop("demo:training_queue")
        print(f"    > Processing: {task}")
 
    print(f"  Queue after      -> {r.llen('demo:training_queue')} tasks remaining")
 
 
# ---------------------------------------------
# 4. SETS
# unique items only - duplicates ignored
# ---------------------------------------------
def demo_sets(r):
    print("\n-- 4. SETS -----------------------------")
    r.delete("demo:nlp_models", "demo:completed_models")
 
    r.sadd("demo:nlp_models",       "BERT", "RoBERTa", "DistilBERT", "GPT2")
    r.sadd("demo:completed_models", "BERT", "RoBERTa", "ResNet50",   "XGBoost")
 
    # adding duplicate - Redis silently ignores it
    r.sadd("demo:nlp_models", "BERT")
    r.sadd("demo:nlp_models", "BERT")
 
    print(f"  NLP models       -> {sorted(r.smembers('demo:nlp_models'))}")
    print(f"  Has BERT?        -> {r.sismember('demo:nlp_models', 'BERT')}")
    print(f"  Has LSTM?        -> {r.sismember('demo:nlp_models', 'LSTM')}")
    print(f"  Intersection     -> {r.sinter('demo:nlp_models', 'demo:completed_models')}")
    print(f"  Union            -> {sorted(r.sunion('demo:nlp_models', 'demo:completed_models'))}")
    print(f"  Difference       -> {r.sdiff('demo:nlp_models', 'demo:completed_models')}")
 
 
# ---------------------------------------------
# 5. SORTED SETS
# like sets but each member has a score - perfect for leaderboards
# ---------------------------------------------
def demo_sorted_sets(r):
    print("\n-- 5. SORTED SETS (F1 Leaderboard) ----")
    r.delete("demo:f1_lb")
 
    # zadd with scores - Redis keeps them sorted automatically
    r.zadd("demo:f1_lb", {
        "BERT Sentiment"    : 0.9020,
        "RoBERTa Sentiment" : 0.9360,
        "DistilBERT Fast"   : 0.8985,
        "ResNet50 Medical"  : 0.8530,
        "XGBoost Fraud"     : 0.9279,
        "LightGBM Fraud"    : 0.9100,
    })
 
    # top 3 highest f1
    print("  Top 3 models:")
    for rank, (name, score) in enumerate(r.zrevrange("demo:f1_lb", 0, 2, withscores=True), 1):
        print(f"    #{rank} {name:28s} F1={score}")
 
    # rank of a specific model (0-indexed, so +1)
    rank = r.zrevrank("demo:f1_lb", "RoBERTa Sentiment")
    score = r.zscore("demo:f1_lb", "RoBERTa Sentiment")
    print(f"\n  RoBERTa rank     -> #{rank + 1} with F1={score}")
 
    # models in F1 range 0.90 to 0.95
    in_range = r.zrangebyscore("demo:f1_lb", 0.90, 0.95, withscores=True)
    print(f"  F1 0.90-0.95     -> {[(n, s) for n, s in in_range]}")
 
    # update score
    r.zincrby("demo:f1_lb", 0.01, "XGBoost Fraud")
    print(f"  XGBoost after +0.01 -> {r.zscore('demo:f1_lb', 'XGBoost Fraud')}")
 
 
# ---------------------------------------------
# 6. CACHE-ASIDE PATTERN
# most common caching pattern - check cache first, then DB
# ---------------------------------------------
def demo_cache_aside(r):
    print("\n-- 6. CACHE-ASIDE PATTERN --------------")
 
    def slow_database_query(exp_id):
        """simulates a slow MongoDB query"""
        time.sleep(0.8)
        return {"id": exp_id, "name": "RoBERTa", "f1": 0.9360, "status": "completed"}
 
    def get_experiment(exp_id):
        cache_key = f"demo:exp:{exp_id}"
 
        # step 1: check Redis cache first
        cached = r.get(cache_key)
        if cached:
            return json.loads(cached), "HIT"
 
        # step 2: cache miss - query database
        result = slow_database_query(exp_id)
 
        # step 3: store result in cache for 5 minutes
        r.setex(cache_key, 300, json.dumps(result))
        return result, "MISS"
 
    # first call - no cache yet
    t0 = time.time()
    _, status = get_experiment(42)
    t1 = time.time()
    print(f"  First call  -> {status} | time: {t1-t0:.3f}s (slow - went to DB)")
 
    # second call - cache hit
    t0 = time.time()
    _, status = get_experiment(42)
    t1 = time.time()
    print(f"  Second call -> {status} | time: {t1-t0:.4f}s (fast - from Redis)")
 
    speedup = 0.8 / (t1 - t0)
    print(f"  Cache is ~{speedup:.0f}x faster than DB query")
 
 
# ---------------------------------------------
# 7. PUB/SUB
# publisher sends messages, subscribers receive them
# ---------------------------------------------
def demo_pubsub(r):
    print("\n-- 7. PUB/SUB (Real-time notifications) -")
    received_messages = []
 
    def subscriber_thread():
        # each subscriber needs its own connection
        r2 = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        sub = r2.pubsub()
        sub.subscribe("demo:ml_events")
        for msg in sub.listen():
            if msg["type"] == "message":
                received_messages.append(msg["data"])
                if len(received_messages) >= 3:
                    sub.unsubscribe()
                    break
 
    # start subscriber in background thread
    t = threading.Thread(target=subscriber_thread, daemon=True)
    t.start()
    time.sleep(0.3)  # give thread time to connect
 
    # publish training completion events
    r.publish("demo:ml_events", "Training done: BERT | F1=0.9020")
    r.publish("demo:ml_events", "Training done: RoBERTa | F1=0.9360")
    r.publish("demo:ml_events", "Evaluation done: XGBoost | F1=0.9279")
    t.join(timeout=4)
 
    print("  Messages received by subscriber:")
    for msg in received_messages:
        print(f"    [MSG] {msg}")
 
 
# ---------------------------------------------
# 8. TTL AND EVICTION
# ---------------------------------------------
def demo_ttl_eviction(r):
    print("\n-- 8. TTL & EVICTION -------------------")
 
    r.setex("demo:cache_short",  5,   "expires in 5 seconds")
    r.setex("demo:cache_medium", 300, "expires in 5 minutes")
    r.set("demo:cache_permanent", "never expires - no TTL set")
 
    print(f"  TTL short        -> {r.ttl('demo:cache_short')} sec")
    print(f"  TTL medium       -> {r.ttl('demo:cache_medium')} sec")
    print(f"  TTL permanent    -> {r.ttl('demo:cache_permanent')} (-1 = no expiry)")
 
    # PERSIST removes the TTL - makes key permanent
    r.persist("demo:cache_short")
    print(f"  After PERSIST    -> {r.ttl('demo:cache_short')} (now permanent)")
 
    # check eviction policy
    policy = r.config_get("maxmemory-policy")
    print(f"  Eviction policy  -> {policy}")
    print("  Options: noeviction | allkeys-lru | volatile-lru | allkeys-lfu | volatile-ttl")
 
 
# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    r = get_redis()
    cleanup(r)
 
    demo_strings(r)
    demo_hashes(r)
    demo_lists(r)
    demo_sets(r)
    demo_sorted_sets(r)
    demo_cache_aside(r)
    demo_pubsub(r)
    demo_ttl_eviction(r)
 
    print("\n[OK] redis_cache.py completed!")
    print("   Next: run run_tasks.py to see Redis task queue in action")
 
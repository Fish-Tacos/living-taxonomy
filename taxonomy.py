import csv, json, os, datetime as dt
import numpy as np
import feedparser, requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

DATA = "data"
TREE = f"{DATA}/tree.json"
HIST = f"{DATA}/history"
STAB = f"{DATA}/stability.csv"

ASSIGN_MIN = 0.12   # min cosine similarity to join an existing category
MERGE_MIN = 0.65    # centroid similarity above which two categories merge
ORPHAN_DIV = 6      # one new category per this many unplaced items
HOLD_DAYS = 5       # days an unplaced item waits for company before it is discarded
RETIRE_DAYS = 30    # idle days before a category is retired
REVIVE_MIN = 0.5    # similarity at which new strays bring a retired category back
GRAVE_DAYS = 180    # how long a retired category stays revivable
GRAVE_TEXTS = 100   # abstracts kept per retired category
MAX_ITEMS = 400     # cap on stored items per category
TOP_TERMS = 4       # words used to name a category


def fetch_recent(max_results=300):
    url = (
        "https://export.arxiv.org/api/query?search_query=cat:cs.AI"
        "&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={max_results}"
    )
    resp = requests.get(url, timeout=60, headers={"User-Agent": "living-taxonomy/1.0 (github.com/Fish-Tacos/living-taxonomy)"})
    print(f"arXiv HTTP {resp.status_code}, {len(resp.content)} bytes")
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=3)
    items = []
    newest = None
    for e in feed.entries:
        pub = dt.datetime(*e.published_parsed[:6])
        newest = pub if newest is None else max(newest, pub)
        if pub >= cutoff:
            items.append({
                "id": e.id,
                "title": " ".join(e.title.split()),
                "text": " ".join((e.title + ". " + e.summary).split()),
            })
    print(f"arXiv entries: {len(feed.entries)}, inside 3-day window: {len(items)}, newest submitted: {newest}")
    return items


def load_tree():
    if os.path.exists(TREE):
        with open(TREE) as f:
            return json.load(f)
    return {"next_id": 1, "cats": {}, "assign": {}, "pool": [], "grave": {}}


def name_from(vec, terms):
    idx = np.argsort(vec)[::-1][:TOP_TERMS]
    return " / ".join(terms[i] for i in idx)


def run():
    os.makedirs(HIST, exist_ok=True)
    today = dt.date.today().isoformat()
    tree = load_tree()
    pool = tree.get("pool", [])
    grave = tree.get("grave", {})
    items = fetch_recent()
    seen = set(tree["assign"]) | {p["id"] for p in pool}
    items = [i for i in items if i["id"] not in seen]
    if not items:
        print("no new items")
        return

    stored = {cid: c["texts"] for cid, c in tree["cats"].items()}
    corpus = [t for ts in stored.values() for t in ts] + [i["text"] for i in items] + [p["text"] for p in pool]
    corpus += [t for g in grave.values() for t in g["texts"]]
    vec = TfidfVectorizer(stop_words="english", max_features=6000, ngram_range=(1, 2))
    vec.fit(corpus)
    terms = np.array(vec.get_feature_names_out())

    cids = list(stored)
    cents = np.array([vec.transform(stored[c]).toarray().mean(axis=0) for c in cids]) if cids else np.zeros((0, len(terms)))
    X = vec.transform([i["text"] for i in items]).toarray()

    # 1. Assign to existing categories, or mark orphan.
    prev_assign = dict(tree["assign"])
    orphans = []
    for i, item in enumerate(items):
        if len(cids):
            sims = cosine_similarity(X[i:i+1], cents)[0]
            j = int(sims.argmax())
            if sims[j] >= ASSIGN_MIN:
                c = tree["cats"][cids[j]]
                c["texts"].append(item["text"])
                c["titles"].append(item["title"])
                c["last_active"] = today
                tree["assign"][item["id"]] = cids[j]
                continue
        orphans.append(i)

    # 2. Strays wait in a pool. Once 3 or more are waiting, they form categories,
    #    unless a cluster resembles a retired category, which is then revived.
    for i in orphans:
        pool.append({"id": items[i]["id"], "title": items[i]["title"], "text": items[i]["text"], "since": today})
    waiting = []
    for p in pool:
        age = (dt.date.fromisoformat(today) - dt.date.fromisoformat(p["since"])).days
        if age > HOLD_DAYS:
            tree["assign"][p["id"]] = "unsorted"
        else:
            waiting.append(p)
    pool = waiting
    revived = 0
    if len(pool) >= 3:
        PX = vec.transform([p["text"] for p in pool]).toarray()
        k = max(1, len(pool) // ORPHAN_DIV)
        labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(PX)
        gcents = {g: vec.transform(v["texts"]).toarray().mean(axis=0) for g, v in grave.items()}
        for lab in set(labels):
            rows = [n for n in range(len(pool)) if labels[n] == lab]
            cen = PX[rows].mean(axis=0)
            new_texts = [pool[n]["text"] for n in rows]
            new_titles = [pool[n]["title"] for n in rows]
            best, best_s = None, REVIVE_MIN
            for g, gc in gcents.items():
                if g in grave:
                    s = float(cosine_similarity([cen], [gc])[0][0])
                    if s >= best_s:
                        best, best_s = g, s
            if best:
                old = grave.pop(best)
                cid = best
                tree["cats"][cid] = {
                    "name": name_from(cen, terms),
                    "born": old["born"],
                    "last_active": today,
                    "texts": old["texts"] + new_texts,
                    "titles": old["titles"] + new_titles,
                }
                for k_ in old["ids"]:
                    tree["assign"][k_] = cid
                revived += 1
            else:
                cid = f"c{tree['next_id']}"
                tree["next_id"] += 1
                tree["cats"][cid] = {
                    "name": name_from(cen, terms),
                    "born": today,
                    "last_active": today,
                    "texts": new_texts,
                    "titles": new_titles,
                }
            for n in rows:
                tree["assign"][pool[n]["id"]] = cid
        pool = []
    tree["pool"] = pool

    # 3. Merge converged categories, most similar pair first, until none exceed MERGE_MIN.
    merged = 0
    while len(tree["cats"]) > 1:
        cids = list(tree["cats"])
        cents = np.array([vec.transform(tree["cats"][c]["texts"]).toarray().mean(axis=0) for c in cids])
        S = cosine_similarity(cents)
        np.fill_diagonal(S, 0)
        a, b = np.unravel_index(S.argmax(), S.shape)
        if S[a, b] < MERGE_MIN:
            break
        a, b = min(a, b), max(a, b)
        keep, drop = tree["cats"][cids[a]], tree["cats"][cids[b]]
        keep["texts"] += drop["texts"]
        keep["titles"] += drop["titles"]
        keep["last_active"] = max(keep["last_active"], drop["last_active"])
        for k_, v_ in tree["assign"].items():
            if v_ == cids[b]:
                tree["assign"][k_] = cids[a]
        del tree["cats"][cids[b]]
        merged += 1

    # 4. Retire idle categories into the graveyard, and expire old graves.
    retired = 0
    for cid in list(tree["cats"]):
        c = tree["cats"][cid]
        idle = (dt.date.today() - dt.date.fromisoformat(c["last_active"])).days
        if idle > RETIRE_DAYS:
            ids = [k_ for k_, v_ in tree["assign"].items() if v_ == cid]
            grave[cid] = {
                "name": c["name"], "born": c["born"], "died": today,
                "texts": c["texts"][-GRAVE_TEXTS:], "titles": c["titles"][-GRAVE_TEXTS:],
                "ids": ids,
            }
            for k_ in ids:
                tree["assign"][k_] = "retired"
            del tree["cats"][cid]
            retired += 1
    for g in list(grave):
        if (dt.date.today() - dt.date.fromisoformat(grave[g]["died"])).days > GRAVE_DAYS:
            del grave[g]
    tree["grave"] = grave

    # 5. Trim storage and rename from current centroids.
    for cid, c in tree["cats"].items():
        c["texts"] = c["texts"][-MAX_ITEMS:]
        c["titles"] = c["titles"][-MAX_ITEMS:]
        cen = vec.transform(c["texts"]).toarray().mean(axis=0)
        c["name"] = name_from(cen, terms)

    # 6. Stability: share of previously assigned items that changed category.
    changed = sum(1 for k_, v_ in prev_assign.items() if tree["assign"].get(k_) != v_)
    churn = changed / max(1, len(prev_assign))

    with open(TREE, "w") as f:
        json.dump(tree, f)
    slim = {cid: {"name": c["name"], "born": c["born"], "size": len(c["texts"]), "sample": c["titles"][-3:]} for cid, c in tree["cats"].items()}
    with open(f"{HIST}/{today}.json", "w") as f:
        json.dump(slim, f, indent=1)
    new_file = not os.path.exists(STAB)
    with open(STAB, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "new_items", "categories", "merged", "retired", "revived", "churn"])
        w.writerow([today, len(items), len(tree["cats"]), merged, retired, revived, round(churn, 4)])
    print(f"{today}: +{len(items)} items, {len(tree['cats'])} cats, merged {merged}, retired {retired}, revived {revived}, churn {churn:.3f}, waiting {len(pool)}")


if __name__ == "__main__":
    run()

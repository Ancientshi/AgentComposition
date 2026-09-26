"""Judge completed native runs while the remaining runs execute."""
import concurrent.futures
import json
import time

import run_experiment as runner


queries = [q for q in json.loads((runner.ROOT / "queries.json").read_text())
           if q["dataset"] == "SkillsBench"]
excluded = {(item["sample_id"], item["rank"]) for item in
            json.loads((runner.ROOT / "excluded.json").read_text())}
expected = []
for q in queries:
    rec = json.loads((runner.ROOT / "recommendations" / (q["sample_id"] + ".json")).read_text())
    for config in rec["selected"]:
        item = (q["sample_id"], config["recommendation_rank"])
        if item not in excluded:
            expected.append(item)


def score(item):
    sid, rank = item
    trial = runner.ROOT / "trials" / f"{sid}__r{rank:02d}"
    record = json.loads((trial / "execution.json").read_text())
    quality = runner.judge(record)
    grounded = runner.judge_grounded(record)
    return {"sample_id": sid, "rank": rank,
            "quality": quality["total"], "grounded": grounded["total"]}


submitted = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    while len(submitted) < len(expected):
        for item in expected:
            if item in submitted:
                continue
            sid, rank = item
            path = runner.ROOT / "trials" / f"{sid}__r{rank:02d}" / "execution.json"
            if path.exists():
                submitted[item] = pool.submit(score, item)
        for item, future in list(submitted.items()):
            if future.done() and not getattr(future, "reported", False):
                try:
                    print(json.dumps({"completed": future.result()}, ensure_ascii=False), flush=True)
                except Exception as error:
                    print(json.dumps({"failed": {"sample_id": item[0], "rank": item[1],
                                                   "error": str(error)}}), flush=True)
                future.reported = True
        time.sleep(10)
    for item, future in submitted.items():
        if not getattr(future, "reported", False):
            try:
                print(json.dumps({"completed": future.result()}, ensure_ascii=False), flush=True)
            except Exception as error:
                print(json.dumps({"failed": {"sample_id": item[0], "rank": item[1],
                                               "error": str(error)}}), flush=True)

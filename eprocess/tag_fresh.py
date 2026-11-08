"""Claim tags (trusted model) for every fresh description; same prompt as the semantic channel idea."""
import json, sys, importlib.util
sys.path.insert(0, '/root/sc/testbeds/swarm_collusion')
import llm
spec = importlib.util.spec_from_file_location("s", '/root/sc/ideas/swarm-semantic-channel/method.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
D = '/root/sc/testbeds/swarm_collusion/data/'
for name in ("fresh_honest.json", "fresh_attack_phrase.json", "fresh_attack_paraphrase.json"):
    rows = json.load(open(D + name))
    tags = llm.pmap(lambda r: sorted(m.tags(llm.ask(m.TAGS.format(desc=r["description"]), "gemma-4-31b-it", max_tokens=400,
                                                    tag="freshtag", temperature=0.0))), rows, 32)
    for r, t in zip(rows, tags):
        r["tags"] = t
    json.dump(rows, open(D + name, "w"))
    print(name, len(rows), "tagged; example", rows[0]["tags"][:6], flush=True)

import json, subprocess, argparse
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args()
ablations={"without_fuzzy":{"use_fuzzy":False},"without_temporal":{"use_temporal":False},"without_uncertainty":{"use_uncertainty":False},"without_graph":{"use_graph":False},"without_cost_sensitive":{"cost_sensitive":False}}
for name,flags in ablations.items(): subprocess.run(["python","scripts/run_experiment.py","--config",a.config,"--run-name",name,"--flags",json.dumps(flags)],check=True)

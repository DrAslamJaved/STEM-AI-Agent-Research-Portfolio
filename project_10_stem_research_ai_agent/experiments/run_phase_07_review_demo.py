import json
from pathlib import Path
from stem_research_agent.review import ReviewRecord

record=ReviewRecord("phase-05-results-demo")
record.submit("approve","Dr Aslam Javed","Evidence and results checked.",["C01"])
record.submit("lock","Dr Aslam Javed","Approved draft locked for export.",["C01"])
output=Path("results/phase_07_review_demo.json");output.write_text(json.dumps(record.to_dict(),indent=2)+"\n",encoding="utf-8");print(output)

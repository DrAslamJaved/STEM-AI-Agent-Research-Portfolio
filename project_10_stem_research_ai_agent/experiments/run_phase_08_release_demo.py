import json
from pathlib import Path
from types import SimpleNamespace

from stem_research_agent.release import evaluate_release
from stem_research_agent.review import ReviewRecord


ledger = SimpleNamespace(finalizable_claim_ids=["C01"])
audit = SimpleNamespace(passed=True, findings=[])
review = ReviewRecord("phase-05-results-demo")
review.submit("approve", "Dr Aslam Javed", "Evidence and results checked.", ["C01"])
review.submit("lock", "Dr Aslam Javed", "Approved draft locked for release.", ["C01"])

report = evaluate_release(
    baseline_text="The baseline gives a useful result [C99].",
    controlled_text="The transparent prevalence baseline is evidence-grounded [C01].",
    ledger=ledger,
    audit_report=audit,
    review_record=review,
    tests_passed=True,
    compileall_passed=True,
    diff_check_passed=True,
    human_correction_count=2,
)
output = Path("results/phase_08_release_demo.json")
output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
print(output)

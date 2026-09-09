from hashlib import sha256
from stem_research_agent.provenance import build_file_provenance

def test_provenance_hash_and_size(tmp_path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_bytes(b"drug_id,target_id,label\nD1,T1,1\n")
    record = build_file_provenance(sample)
    assert record["bytes"] == sample.stat().st_size
    assert record["sha256"] == sha256(sample.read_bytes()).hexdigest()

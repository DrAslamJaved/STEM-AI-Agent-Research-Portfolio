import pytest
from stem_research_agent.review import ReviewRecord

def test_approved_locked_draft_is_exportable():
    r=ReviewRecord("D1");r.submit("approve","Reviewer","Checked",["C1"]);r.submit("lock","Reviewer","Lock",["C1"]);assert r.exportable()
def test_locked_draft_rejects_new_action():
    r=ReviewRecord("D1");r.submit("lock","Reviewer","Lock")
    with pytest.raises(ValueError): r.submit("request_revision","Reviewer","Change")
def test_supersede_unlocks_draft():
    r=ReviewRecord("D1");r.submit("lock","Reviewer","Lock");r.submit("supersede","Reviewer","Reopen");assert not r.is_locked
def test_events_are_append_only_and_ordered():
    r=ReviewRecord("D1");r.submit("approve","Reviewer","Checked");r.submit("lock","Reviewer","Lock")
    assert [event.event_id for event in r.events] == [1,2]

def test_revision_request_requires_a_new_approval_before_export():
    r=ReviewRecord("D1");r.submit("approve","Reviewer","Checked");r.submit("request_revision","Reviewer","Clarify")
    r.submit("approve","Reviewer","Updated and checked");r.submit("lock","Reviewer","Lock")
    assert r.exportable()

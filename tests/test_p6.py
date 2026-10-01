from pathlib import Path
import json
import pytest
from manualrag.eval.metrics import retrieval_metrics,citation_metrics,abstention_metrics,wilson
from manualrag.eval.dataset import load_dataset
from manualrag.eval.runner import run_evaluation
from manualrag.eval.report import apply_audit
from manualrag.ingest.pipeline import ingest_pdf
from manualrag.config import Settings

def test_metrics_and_denominators() -> None:
    result=retrieval_metrics([2,2,4,6],[4,6])
    assert result["recall@1"]==0 and result["recall@3"]==1 and result["mrr"]==.5
    assert citation_metrics([4,9],[4,6])=={"citation_precision":.5,"citation_recall":.5}
    a=abstention_metrics([True,True,False,False],[False,True,True,False])
    assert a["abstention_precision"]==.5 and a["false_refusal_rate"]==.5
    low,high=wilson(5,5)
    assert .56<low<.57 and high==1
    assert wilson(0,0) is None

def test_dataset_fixed_splits() -> None:
    path=Path(__file__).resolve().parents[1]/"eval"/"questions.jsonl"
    records,_,manifest=load_dataset(path)
    assert len(records)==48 and sum(q.split=="dev" for q in records)==14
    assert manifest["source_type"]=="synthetic"
    assert sum(not q.answerable for q in records)==6

def test_runner_dev_only_and_unavailable_ablations(manual: Path,settings: Settings,tmp_path: Path) -> None:
    product=ingest_pdf(manual,"Lab","ML-100",settings)
    dataset=tmp_path/"questions.jsonl"
    records=[
        dict(id="d1",question="blue shipping tape",category="setup",answerable=True,gold_pages=[1],gold_points=["Remove tape"],split="dev"),
        dict(id="d2",question="E99",category="unanswerable",answerable=False,gold_pages=[],gold_points=[],split="dev"),
        dict(id="t1",question="jammed paper",category="jams",answerable=True,gold_pages=[4],gold_points=["Unplug"],split="test"),
        dict(id="t2",question="F99",category="unanswerable",answerable=False,gold_pages=[],gold_points=[],split="test")]
    dataset.write_text("\n".join(json.dumps(r) for r in records),encoding="utf-8")
    dataset.with_suffix(".manifest.json").write_text(json.dumps({"source_type":"synthetic","model":"ML-100"}),encoding="utf-8")
    output=tmp_path/"report"
    run_evaluation(settings,dataset,product.id,"A,D",output,True,True)
    calibration=json.loads((output/"calibration_A.json").read_text(encoding="utf-8"))
    assert calibration["dev_ids"]==["d1","d2"]
    result=json.loads((output/"results.json").read_text(encoding="utf-8"))
    assert {r["id"] for r in result["rows"]}=={"t1","t2"}
    assert "D" in result["skipped"]
    assert all(r["faithfulness"] is None for r in result["rows"])
    audit=apply_audit(output,output/"human_audit.csv",settings)
    assert not audit["stage1_pass"] and not (settings.data_dir/"stage1_approval.json").exists()

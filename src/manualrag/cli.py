"""PowerShell-friendly command-line interface."""
import argparse
import json
import logging
from pathlib import Path
from manualrag.config import load_settings
from manualrag.index.store import Store

logger=logging.getLogger(__name__)

def main() -> None:
    """Run ingestion, search, Q&A, synthetic fixtures, or evaluation."""
    parser=argparse.ArgumentParser(prog="manualrag",description="ManualLens: manual-grounded assistance")
    parser.add_argument("--version",action="version",version="ManualLens 0.1.0")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("ingest","search","ask","eval","synthetic","audit"):
        command=sub.add_parser(name)
        command.add_argument("--settings",type=Path,default=Path("config.yaml"))
        if name=="ingest":
            command.add_argument("pdf",type=Path)
            command.add_argument("--name",required=True)
            command.add_argument("--model",required=True)
            command.add_argument("--page-cards",action="store_true")
        elif name in {"search","ask"}:
            command.add_argument("question")
            command.add_argument("--product",required=True)
            command.add_argument("--mode",choices=["bm25","dense","hybrid"],default="hybrid")
            if name=="ask": command.add_argument("--conversation")
        elif name=="synthetic":
            command.add_argument("--output",type=Path,default=Path("data/manuals/synthetic_ml100.pdf"))
        elif name=="eval":
            command.add_argument("--dataset",type=Path,default=Path("eval/questions.jsonl"))
            command.add_argument("--product",required=True)
            command.add_argument("--config",default="A..E",help="A, B, C, D, E, comma list or A..E")
            command.add_argument("--retrieval-only",action="store_true")
            command.add_argument("--synthetic",action="store_true")
            command.add_argument("--output",type=Path,default=Path("eval/results"))
        elif name=="audit":
            command.add_argument("--report-dir",type=Path,required=True)
            command.add_argument("--audit-file",type=Path,required=True)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    settings=load_settings(args.settings)
    try:
        if args.command=="ingest":
            from manualrag.ingest.pipeline import ingest_pdf
            from manualrag.llm.client import AnthropicClient
            settings.enable_page_cards=args.page_cards or settings.enable_page_cards
            client=AnthropicClient(settings) if settings.enable_page_cards else None
            product=ingest_pdf(args.pdf,args.name,args.model,settings,llm=client)
            logger.info(product.model_dump_json(indent=2))
        elif args.command=="synthetic":
            from manualrag.ingest.synthetic import create_synthetic_manual
            logger.info("Created %s",create_synthetic_manual(args.output).resolve())
        elif args.command=="search":
            from manualrag.models import QueryPlan
            from manualrag.retrieval.hybrid import Retriever
            from manualrag.ingest.error_codes import CODE_PATTERN
            code=CODE_PATTERN.search(args.question)
            plan=QueryPlan(standalone_query=args.question,intent="error_code" if code else "spec_lookup",error_code=code.group() if code else None)
            hits=Retriever(Store(settings.db_path),args.product,settings,mode=args.mode).search(plan)
            logger.info(json.dumps([h.model_dump() for h in hits],indent=2,ensure_ascii=False))
        elif args.command=="ask":
            from manualrag.service import ManualAssistant
            from manualrag.llm.client import AnthropicClient
            result=ManualAssistant(settings,AnthropicClient(settings),mode=args.mode).ask(args.product,args.question,args.conversation)
            logger.info(result.model_dump_json(indent=2))
        elif args.command=="eval":
            from manualrag.eval.runner import run_evaluation
            run_evaluation(settings,args.dataset,args.product,args.config,args.output,args.retrieval_only,args.synthetic)
        elif args.command=="audit":
            from manualrag.eval.report import apply_audit
            apply_audit(args.report_dir,args.audit_file,settings)
    except (ValueError,FileNotFoundError) as exc:
        logger.error("%s",exc)
        raise SystemExit(2) from exc

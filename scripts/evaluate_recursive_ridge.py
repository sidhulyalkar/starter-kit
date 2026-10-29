"""Compare strictly retrospective frozen-input 5/15/60-min forecasts.

This does not give the model later measurements or HVAC actions. It deliberately
freezes non-zone observations and the initial control action at each anchor.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from smart_buildings.experiments.thermal_recursive import evaluate_recursive


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",required=True,nargs="+",type=Path)
    p.add_argument("--eval",required=True,nargs="+",type=Path)
    p.add_argument("--alpha",default=100.0,type=float)
    p.add_argument("--horizons",default=[1,3,12],type=int,nargs="+")
    p.add_argument("--stride",default=12,type=int)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    try:
        result=evaluate_recursive(args.train,args.eval,args.alpha,
                                  tuple(args.horizons),args.stride)
    except (ValueError,KeyError,OSError) as exc:
        p.exit(1,f"Recursive forecast evaluation refused: {exc}\n")
    text=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

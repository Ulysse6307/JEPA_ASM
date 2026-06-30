#!/usr/bin/env python3
"""Headroom proof — does -O3 / -Ofast actually beat -O2 in WALL-CLOCK on real,
compute-bound whole programs?

The Step-1 gate (docs/results_gate_exebench.md) found O2≈O3 graphs identical on
*isolated functions* — but the founding thesis ("compilers leave performance on the
table") lives on bigger, loopy, vectorizable code. This probe answers the go/no-go
directly: compile a set of compute kernels at several -O levels, run each many times,
take the MIN wall-clock (robust to noise), and report the speedup -O3/-Ofast buy over
-O2. No GPU, no programl — just a C compiler + a stopwatch.

Run: python3 scripts/probe_headroom.py            (auto-detects clang/gcc)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time

# Compute-bound single-file kernels. Each does fixed work and prints a checksum so
# the optimizer cannot delete the loop. Sized for ~0.2-1s at -O2.
KERNELS: dict[str, str] = {
    "matmul": r"""
#include <stdio.h>
#include <stdlib.h>
#define N 384
static float A[N*N], B[N*N], C[N*N];
int main(void){
  for(int i=0;i<N*N;i++){A[i]=(i*7%13)*0.1f; B[i]=(i*5%11)*0.2f;}
  for(int r=0;r<3;r++)
    for(int i=0;i<N;i++)
      for(int k=0;k<N;k++){
        float a=A[i*N+k];
        for(int j=0;j<N;j++) C[i*N+j]+=a*B[k*N+j];
      }
  double s=0; for(int i=0;i<N;i++) s+=C[i*N+i];
  printf("%.1f\n", s); return 0;
}
""",
    "saxpy_dot": r"""
#include <stdio.h>
#include <stdlib.h>
#define M (1<<20)
static float x[M], y[M];
int main(void){
  for(int i=0;i<M;i++){x[i]=(i%97)*0.01f; y[i]=(i%89)*0.02f;}
  double acc=0;
  for(int r=0;r<300;r++){
    for(int i=0;i<M;i++) y[i]=1.0001f*x[i]+y[i];
    double d=0; for(int i=0;i<M;i++) d+=x[i]*y[i];
    acc+=d;
  }
  printf("%.0f\n", acc); return 0;
}
""",
    "mandelbrot": r"""
#include <stdio.h>
#define W 1000
#define H 1000
#define IT 200
int main(void){
  long long total=0;
  for(int py=0;py<H;py++){
    double y0=(py/(double)H)*2.5-1.25;
    for(int px=0;px<W;px++){
      double x0=(px/(double)W)*3.0-2.0, x=0,y=0; int it=0;
      while(x*x+y*y<=4.0 && it<IT){ double xt=x*x-y*y+x0; y=2*x*y+y0; x=xt; it++; }
      total+=it;
    }
  }
  printf("%lld\n", total); return 0;
}
""",
    "nbody": r"""
#include <stdio.h>
#include <math.h>
#define N 2048
#define STEPS 12
static double px[N],py[N],pz[N],vx[N],vy[N],vz[N];
static void force(int i,double*fx,double*fy,double*fz){
  double ax=0,ay=0,az=0;
  for(int j=0;j<N;j++){ if(j==i) continue;
    double dx=px[j]-px[i],dy=py[j]-py[i],dz=pz[j]-pz[i];
    double r2=dx*dx+dy*dy+dz*dz+1e-9, inv=1.0/sqrt(r2*r2*r2);
    ax+=dx*inv; ay+=dy*inv; az+=dz*inv; }
  *fx=ax;*fy=ay;*fz=az;
}
int main(void){
  for(int i=0;i<N;i++){px[i]=(i*13%101)*0.1; py[i]=(i*7%97)*0.1; pz[i]=(i*5%89)*0.1;}
  for(int s=0;s<STEPS;s++) for(int i=0;i<N;i++){
    double fx,fy,fz; force(i,&fx,&fy,&fz);
    vx[i]+=1e-4*fx; vy[i]+=1e-4*fy; vz[i]+=1e-4*fz;
    px[i]+=vx[i]; py[i]+=vy[i]; pz[i]+=vz[i];
  }
  double s=0; for(int i=0;i<N;i++) s+=px[i]+py[i]+pz[i];
  printf("%.2f\n", s); return 0;
}
""",
    "stencil": r"""
#include <stdio.h>
#define N 512
#define IT 80
static double a[N*N], b[N*N];
int main(void){
  for(int i=0;i<N*N;i++) a[i]=(i%100)*0.01;
  for(int t=0;t<IT;t++){
    for(int i=1;i<N-1;i++) for(int j=1;j<N-1;j++)
      b[i*N+j]=0.25*(a[(i-1)*N+j]+a[(i+1)*N+j]+a[i*N+j-1]+a[i*N+j+1]);
    for(int i=1;i<N-1;i++) for(int j=1;j<N-1;j++) a[i*N+j]=b[i*N+j];
  }
  double s=0; for(int i=0;i<N*N;i++) s+=a[i]; printf("%.2f\n", s); return 0;
}
""",
    "sieve": r"""
#include <stdio.h>
#include <string.h>
#define LIM 5000000
static char is[LIM+1];
int main(void){
  long long total=0;
  for(int r=0;r<12;r++){
    memset(is,1,sizeof(is)); long cnt=0;
    for(int i=2;i<=LIM;i++) if(is[i]){ cnt++; for(long j=(long)i*i;j<=LIM;j+=i) is[j]=0; }
    total+=cnt;
  }
  printf("%lld\n", total); return 0;
}
""",
}

# native variants matter on x86 (AVX autovec needs -march=native; the generic target
# stays on SSE2). They no-op/skip on arm64 (Apple clang rejects -march=native -> None).
OPT_CONFIGS = ["-O0", "-O1", "-O2", "-O3", "-O3 -march=native",
               "-Ofast", "-Ofast -march=native"]
SUMMARY_TARGETS = ["-O3", "-O3 -march=native", "-Ofast", "-Ofast -march=native"]
BASELINE = "-O2"


def find_compiler(pref: str | None) -> str:
    if pref:
        return pref
    for c in ("clang", "gcc", "cc"):
        if shutil.which(c):
            return c
    raise SystemExit("no C compiler found (clang/gcc)")


def compile_kernel(cc: str, src: str, opt: str, out: str) -> bool:
    cmd = [cc, *opt.split(), "-lm", "-o", out, "-x", "c", "-"]
    try:
        p = subprocess.run(cmd, input=src, text=True, capture_output=True, timeout=120)
        return p.returncode == 0 and os.path.exists(out)
    except subprocess.SubprocessError:
        return False


def time_binary(path: str, repeats: int) -> float:
    """Min wall-clock over `repeats` runs (after a warmup)."""
    subprocess.run([path], capture_output=True)  # warmup
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        subprocess.run([path], capture_output=True)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="O2-vs-O3-vs-Ofast wall-clock headroom")
    ap.add_argument("--cc", default=None, help="compiler (default: autodetect)")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=5.0, help="meaningful speedup %%")
    ap.add_argument("--out", default="headroom_report.json")
    args = ap.parse_args()
    cc = find_compiler(args.cc)
    ver = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"HEADROOM::COMPILER {ver}", flush=True)

    results: dict[str, dict[str, float | None]] = {}
    with tempfile.TemporaryDirectory() as d:
        for name, src in KERNELS.items():
            row: dict[str, float | None] = {}
            for opt in OPT_CONFIGS:
                binp = os.path.join(d, f"{name}{opt.replace(' ', '_')}")
                if compile_kernel(cc, src, opt, binp):
                    row[opt] = round(time_binary(binp, args.repeats) * 1000, 1)  # ms
                else:
                    row[opt] = None
            results[name] = row
            base = row.get(BASELINE)
            o3 = row.get("-O3")
            spd = (base / o3) if base and o3 else None
            print(f"HEADROOM::KERNEL {name:12s} "
                  + " ".join(f"{o}={row[o]}ms" for o in OPT_CONFIGS)
                  + (f"  O3/O2 speedup={spd:.2f}x" if spd else ""), flush=True)

    # summary: speedups vs -O2
    def speedups(target: str) -> list[float]:
        out = []
        for r in results.values():
            b, t = r.get(BASELINE), r.get(target)
            if b and t:
                out.append(b / t)
        return out

    summary = {}
    for target in SUMMARY_TARGETS:
        s = speedups(target)
        if not s:
            continue
        meaningful = sum(1 for x in s if (x - 1) * 100 >= args.threshold)
        summary[f"vs_O2__{target}"] = {
            "median_speedup": round(statistics.median(s), 3),
            "max_speedup": round(max(s), 3),
            "geomean_speedup": round(statistics.geometric_mean(s), 3),
            f"n_kernels_over_{args.threshold:.0f}pct": meaningful,
            "n_total": len(s),
        }

    report = {"compiler": ver, "baseline": BASELINE, "kernels": results,
              "summary": summary}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("HEADROOM::SUMMARY " + json.dumps(summary, indent=2))

    best = max((v.get("max_speedup", 1.0) for v in summary.values()), default=1.0)
    geo = max((v.get("geomean_speedup", 1.0) for v in summary.values()), default=1.0)
    verdict = "HEADROOM_EXISTS" if (geo >= 1.05 or best >= 1.2) else "FLAT"
    print(f"HEADROOM::GATE geomean={geo:.3f}x best={best:.2f}x verdict={verdict}")
    print("HEADROOM::DONE")


if __name__ == "__main__":
    main()

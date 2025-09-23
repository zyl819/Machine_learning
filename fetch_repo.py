#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from process_data_3 import preprocess_data
from extract_feature_4 import extract_features_from_excel
from libary_1 import build_repo_pr_dataframe

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="Fetch GitHub PR data and save datasets")
    parser.add_argument("repos", nargs="?", help="Comma-separated repos like owner1/repo1,owner2/repo2; or set GITHUB_REPOS env")
    parser.add_argument("--output", dest="output", default=os.getenv("OUTPUT_DIR", "data_out"), help="Output directory")
    parser.add_argument("--max-prs", dest="max_prs", type=int, default=None, help="Limit number of PRs per repo")
    parser.add_argument("--since", dest="since", default=None, help="ISO date lower bound")
    parser.add_argument("--until", dest="until", default=None, help="ISO date upper bound")
    parser.add_argument("--no-skip-commits", dest="skip_commits", action="store_false", help="Do not skip commits endpoint")
    parser.add_argument("--workers", dest="workers", type=int, default=8, help="Max concurrent workers")
    args = parser.parse_args()

    repos_env = args.repos or os.getenv("GITHUB_REPOS")
    if not repos_env:
        raise SystemExit("Please provide repos via argument or GITHUB_REPOS env")

    os.makedirs(args.output, exist_ok=True)
    repos = [r.strip() for r in repos_env.split(',') if r.strip()]
    total = len(repos)

    for idx, repo_full in enumerate(repos, start=1):
        owner, repo = repo_full.split('/')
        print(f"\n>>> [{idx}/{total}] 抓取仓库: {owner}/{repo}")

        # 进度回调：每完成一个PR打印一次
        def on_progress(done: int, all_count: int):
            print(f"  - 进度: {done}/{all_count}", end="\r", flush=True)

        df = build_repo_pr_dataframe(owner, repo,
                                     max_prs=args.max_prs,
                                     since=args.since,
                                     until=args.until,
                                     skip_commits=args.skip_commits,
                                     max_workers=args.workers,
                                     on_progress=on_progress)
        print()  # 换行，避免覆盖
        try:
            created_min = pd.to_datetime(df['created_at']).min()
            created_max = pd.to_datetime(df['created_at']).max()
            merged_ratio = float(df['merged'].mean()) if 'merged' in df.columns and len(df) > 0 else 0.0
            print(f"抓取PR数: {len(df)}，合并占比: {merged_ratio:.2f}，时间范围: {created_min} ~ {created_max}")
        except Exception:
            print(f"抓取PR数: {len(df)}")

        raw_path = os.path.join(args.output, f"{owner}_{repo}_raw.parquet")
        try:
            df.to_parquet(raw_path, index=False)
        except Exception:
            df.to_csv(raw_path.replace('.parquet', '.csv'), index=False)

        print("预处理与特征提取（可用于快速检查列对齐）")
        pr_df_clean = preprocess_data(df)
        X_features = extract_features_from_excel(pr_df_clean)

        clean_path = os.path.join(args.output, f"{owner}_{repo}_clean.parquet")
        feat_path = os.path.join(args.output, f"{owner}_{repo}_features.parquet")
        try:
            pr_df_clean.to_parquet(clean_path, index=False)
            X_features.to_parquet(feat_path, index=False)
        except Exception:
            pr_df_clean.to_csv(clean_path.replace('.parquet', '.csv'), index=False)
            X_features.to_csv(feat_path.replace('.parquet', '.csv'))

        print("完成")


if __name__ == "__main__":
    main() 
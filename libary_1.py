# 导入所有需要的库
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import os
import time
import requests
from typing import Dict, Any, List, Optional, Tuple, Callable

GITHUB_API = "https://api.github.com"


def _get_github_token() -> Optional[str]:
    return os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def _headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ML-PR-Analyzer"
    }
    token = _get_github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _respect_rate_limit(resp: requests.Response) -> None:
    if resp.status_code == 403:
        reset = resp.headers.get("X-RateLimit-Reset")
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining == "0" and reset:
            try:
                reset_ts = int(reset)
                sleep_s = max(0, reset_ts - int(time.time()) + 1)
                time.sleep(sleep_s)
            except Exception:
                time.sleep(10)


def _get(session: requests.Session, url: str, params: Dict[str, Any] = None) -> requests.Response:
    for attempt in range(5):
        resp = session.get(url, headers=_headers(), params=params, timeout=30)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (403, 429, 502, 503, 504):
            _respect_rate_limit(resp)
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
    resp.raise_for_status()


def _paginate(session: requests.Session, url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    page = 1
    while True:
        page_params = dict(params or {})
        page_params.update({"per_page": 100, "page": page})
        resp = _get(session, url, page_params)
        items = resp.json()
        if not isinstance(items, list):
            break
        all_items.extend(items)
        if len(items) < 100:
            break
        page += 1
    return all_items


# ======================= 高级客户端（Session + 并发） =======================
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from threading import Lock


class GitHubClient:
    def __init__(self, max_workers: int = 8):
        self.session = requests.Session()
        self.max_workers = max_workers

    def fetch_pull_requests(self, owner: str, repo: str, state: str = "all", max_prs: Optional[int] = None,
                            since: Optional[str] = None, until: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"
        params = {"state": state, "sort": "created", "direction": "asc"}
        items = _paginate(self.session, url, params)
        # 过滤时间范围
        if since or until:
            def in_range(item: Dict[str, Any]) -> bool:
                created = pd.to_datetime(item.get('created_at'))
                if since and created < pd.to_datetime(since):
                    return False
                if until and created > pd.to_datetime(until):
                    return False
                return True
            items = [it for it in items if in_range(it)]
        # 限制数量
        if max_prs is not None:
            items = items[:max_prs]
        return items

    def fetch_pull_request_detail(self, owner: str, repo: str, number: int) -> Dict[str, Any]:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}"
        return _get(self.session, url).json()

    def fetch_pull_request_files(self, owner: str, repo: str, number: int) -> List[Dict[str, Any]]:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/files"
        return _paginate(self.session, url, params={})

    def fetch_pull_request_commits(self, owner: str, repo: str, number: int) -> List[Dict[str, Any]]:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/commits"
        return _paginate(self.session, url, params={})


from collections import Counter

def _infer_language_from_filename(filename: str) -> str:
    ext = (filename.rsplit('.', 1)[-1] if '.' in filename else '').lower()
    mapping = {
        'py': 'Python', 'js': 'JavaScript', 'ts': 'TypeScript', 'java': 'Java', 'rb': 'Ruby',
        'go': 'Go', 'cpp': 'C++', 'c': 'C', 'cs': 'C#', 'php': 'PHP', 'rs': 'Rust', 'kt': 'Kotlin',
        'swift': 'Swift', 'm': 'Objective-C', 'scala': 'Scala', 'sh': 'Shell', 'yaml': 'YAML',
        'yml': 'YAML', 'json': 'JSON', 'md': 'Markdown', 'txt': 'Text'
    }
    return mapping.get(ext, ext or 'unknown')


def build_repo_pr_dataframe(owner: str, repo: str, *, state: str = "all", max_prs: Optional[int] = None,
                             since: Optional[str] = None, until: Optional[str] = None,
                             skip_commits: bool = True, max_workers: int = 8,
                             on_progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    """
    更快的抓取：复用 Session + 并发抓取 detail/files(/commits)
    - max_prs: 限制PR数量，快速试跑
    - since/until: 只抓取时间范围内的PR
    - skip_commits: 默认跳过 commits 接口（常较慢）
    - max_workers: 线程数
    - on_progress(done, total): 每完成一个PR触发回调
    """
    client = GitHubClient(max_workers=max_workers)
    prs = client.fetch_pull_requests(owner, repo, state=state, max_prs=max_prs, since=since, until=until)

    records: List[Dict[str, Any]] = []
    total = len(prs)
    done = 0
    lock = Lock()

    def fetch_one(pr: Dict[str, Any]) -> Dict[str, Any]:
        number = pr.get('number')
        detail = client.fetch_pull_request_detail(owner, repo, number)
        files = client.fetch_pull_request_files(owner, repo, number)
        commits = [] if skip_commits else client.fetch_pull_request_commits(owner, repo, number)

        additions = sum(f.get('additions', 0) for f in files)
        deletions = sum(f.get('deletions', 0) for f in files)
        changed_files = len(files)
        files_added = sum(1 for f in files if f.get('status') == 'added')
        files_deleted = sum(1 for f in files if f.get('status') == 'removed')
        files_updated = sum(1 for f in files if f.get('status') in ('modified', 'renamed', 'changed'))

        directories_set = set('/'.join(f.get('filename', '').split('/')[:-1]) for f in files if f.get('filename'))
        directory_num = len([d for d in directories_set if d])

        file_types = [ (f.get('filename','').rsplit('.',1)[-1] if '.' in f.get('filename','') else '') for f in files ]
        file_type_num = len(set([ft.lower() for ft in file_types if ft]))

        language_types = set(_infer_language_from_filename(f.get('filename','')) for f in files if f.get('filename'))
        language_num = len([l for l in language_types if l])

        comments = (detail.get('review_comments', 0) or 0) + (detail.get('comments', 0) or 0)
        commits_count = detail.get('commits', (len(commits) if isinstance(commits, list) else 0)) or 0

        return {
            'number': number,
            'title': pr.get('title') or '',
            'body': pr.get('body') or '',
            'title_length': len(pr.get('title') or ''),
            'body_length': len(pr.get('body') or ''),
            'created_at': pr.get('created_at'),
            'merged_at': pr.get('merged_at'),
            'updated_at': pr.get('updated_at'),
            'closed_at': pr.get('closed_at'),
            'state': pr.get('state'),
            'merged': True if pr.get('merged_at') else False,
            'comments': comments,
            'commits': commits_count,
            'additions': additions,
            'deletions': deletions,
            'changed_files': changed_files,
            'lines_added': additions,
            'lines_deleted': deletions,
            'files_added': files_added,
            'files_deleted': files_deleted,
            'files_updated': files_updated,
            'files_modified': files_updated,
            'directory_num': directory_num,
            'file_type': file_type_num,
            'language_num': language_num,
            'segs_added': 0,
            'segs_deleted': 0,
            'segs_updated': 0,
            'file_developer': 0,
            'change_num': changed_files,
            'k_coreness': 0,
            'experience': 0,
            'repo': f"{owner}/{repo}",
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one, pr) for pr in prs]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                records.append(rec)
            finally:
                with lock:
                    nonlocal_done = None  # placeholder to avoid linter complaints
                    done += 1
                    if on_progress:
                        try:
                            on_progress(done, total)
                        except Exception:
                            pass

    df = pd.DataFrame.from_records(records)
    for col in ['created_at', 'merged_at', 'updated_at', 'closed_at']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df
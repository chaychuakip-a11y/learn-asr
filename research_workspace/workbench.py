"""Small, dependency-free tools for transparent information research.

The module deliberately separates discovery from evidence. A search result,
snippet, mirror, or AI summary can help locate a source, but it does not replace
inspection of the underlying source.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


CROSSREF_WORKS_ENDPOINT = "https://api.crossref.org/works"
REQUIRED_DOSSIER_FIELDS = (
    "question",
    "decision",
    "scope",
    "inclusion_criteria",
    "exclusion_criteria",
    "query_log",
    "sources",
    "claims",
    "conflicts",
    "stop_rule",
    "limitations",
)


def _normalized_text(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value).casefold())


def _normalized_doi(value: Any) -> str:
    doi = str(value).strip().casefold()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return _normalized_text(doi)


def _quoted_term(term: str) -> str:
    value = term.strip()
    if not value:
        raise ValueError("检索词不能为空")
    if re.search(r"\s", value) and not (value.startswith('"') and value.endswith('"')):
        return f'"{value}"'
    return value


def build_boolean_query(facets: Mapping[str, Sequence[str]]) -> str:
    """Build an AND-of-facets, OR-within-facets Boolean query."""
    groups: list[str] = []
    for _, terms in facets.items():
        unique: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = _normalized_text(term)
            if key and key not in seen:
                seen.add(key)
                unique.append(_quoted_term(term))
        if unique:
            groups.append(f"({' OR '.join(unique)})")
    if not groups:
        raise ValueError("至少需要一个非空检索分面")
    return " AND ".join(groups)


def build_crossref_url(
    query: str,
    *,
    rows: int = 20,
    from_year: int | None = None,
    work_type: str | None = None,
    mailto: str | None = None,
) -> str:
    """Create a bounded Crossref Works API URL without making a request."""
    if not query.strip():
        raise ValueError("query 不能为空")
    if not 1 <= rows <= 1000:
        raise ValueError("rows 必须在 1 到 1000 之间")
    parameters: list[tuple[str, str]] = [
        ("query.bibliographic", query.strip()),
        ("rows", str(rows)),
        (
            "select",
            "DOI,title,author,published,type,URL,container-title,is-referenced-by-count",
        ),
    ]
    filters: list[str] = []
    if from_year is not None:
        filters.append(f"from-pub-date:{from_year}-01-01")
    if work_type:
        filters.append(f"type:{work_type}")
    if filters:
        parameters.append(("filter", ",".join(filters)))
    if mailto:
        parameters.append(("mailto", mailto.strip()))
    return f"{CROSSREF_WORKS_ENDPOINT}?{urlencode(parameters)}"


def crossref_search(url: str, *, timeout: float = 20.0) -> list[dict[str, Any]]:
    """Fetch a URL produced by build_crossref_url from Crossref only."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "api.crossref.org" or parsed.path != "/works":
        raise ValueError("只允许访问 Crossref 官方 Works API")
    request = Request(url, headers={"User-Agent": "learn-asr-research-workbench/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return list(payload.get("message", {}).get("items", []))


def stable_source_key(source: Mapping[str, Any]) -> str:
    """Return a deduplication key, preferring DOI over title and year."""
    doi = _normalized_doi(source.get("doi") or source.get("DOI") or "")
    if doi:
        return f"doi:{doi}"
    title = _normalized_text(source.get("title", ""))
    year = _normalized_text(source.get("year", ""))
    return f"title-year:{title}:{year}"


def deduplicate_sources(sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        key = stable_source_key(source)
        if key not in seen:
            seen.add(key)
            unique.append(dict(source))
    return unique


def evidence_family_summary(sources: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Collapse reports into source families so copies do not inflate evidence."""
    source_list = list(sources)
    family_counts = Counter(
        str(source.get("source_family") or stable_source_key(source)) for source in source_list
    )
    direct_families = {
        str(source.get("source_family") or stable_source_key(source))
        for source in source_list
        if source.get("direct_evidence", False)
    }
    return {
        "report_count": len(source_list),
        "family_count": len(family_counts),
        "direct_family_count": len(direct_families),
        "families": dict(sorted(family_counts.items())),
    }


def should_stop_searching(
    new_eligible_by_round: Sequence[int],
    *,
    window: int = 3,
    threshold: int = 0,
) -> dict[str, Any]:
    """Apply an explicit saturation heuristic and expose its limitations."""
    if window < 1:
        raise ValueError("window 至少为 1")
    if threshold < 0:
        raise ValueError("threshold 不能为负数")
    if any(value < 0 for value in new_eligible_by_round):
        raise ValueError("每轮新增条目数不能为负数")
    enough_rounds = len(new_eligible_by_round) >= window
    recent = list(new_eligible_by_round[-window:]) if enough_rounds else []
    stopped = enough_rounds and all(value <= threshold for value in recent)
    return {
        "stop": stopped,
        "recent_rounds": recent,
        "rule": f"连续 {window} 轮新增合格来源均不超过 {threshold}",
        "boundary": "这是资源受限研究的透明启发式，不是完整性证明，也不能替代预设纳入标准。",
    }


def prisma_flow_errors(flow: Mapping[str, int]) -> list[str]:
    """Check basic arithmetic in a PRISMA-like selection flow."""
    errors: list[str] = []
    required = (
        "identified",
        "duplicates_removed",
        "screened",
        "excluded_screening",
        "full_text_assessed",
        "excluded_full_text",
        "included",
    )
    missing = [name for name in required if name not in flow]
    if missing:
        return [f"缺少流程字段：{', '.join(missing)}"]
    if any(not isinstance(flow[name], int) or flow[name] < 0 for name in required):
        return ["流程计数必须是非负整数"]
    if flow["screened"] != flow["identified"] - flow["duplicates_removed"]:
        errors.append("screened 应等于 identified - duplicates_removed")
    if flow["full_text_assessed"] != flow["screened"] - flow["excluded_screening"]:
        errors.append("full_text_assessed 应等于 screened - excluded_screening")
    if flow["included"] != flow["full_text_assessed"] - flow["excluded_full_text"]:
        errors.append("included 应等于 full_text_assessed - excluded_full_text")
    return errors


def audit_dossier(dossier: Mapping[str, Any]) -> dict[str, list[str]]:
    """Audit the structure and traceability of a research dossier."""
    errors: list[str] = []
    warnings: list[str] = []
    missing = [field for field in REQUIRED_DOSSIER_FIELDS if not dossier.get(field)]
    if missing:
        errors.append(f"缺少必填研究记录：{', '.join(missing)}")

    sources = dossier.get("sources", [])
    source_ids = {str(source.get("id")) for source in sources if source.get("id")}
    if len(source_ids) != len(sources):
        errors.append("每个来源必须有唯一 id")
    duplicate_keys = [key for key, count in Counter(stable_source_key(s) for s in sources).items() if count > 1]
    if duplicate_keys:
        warnings.append(f"发现可能重复的来源记录：{', '.join(duplicate_keys)}")
    for source in sources:
        source_type = str(source.get("source_type", "")).casefold()
        if source_type in {"ai_summary", "search_snippet", "mirror", "转载"}:
            warnings.append(
                f"来源 {source.get('id', '?')} is not direct evidence；只能用于导航，需回到原始来源。"
            )
        if not source.get("url") and not source.get("doi"):
            warnings.append(f"来源 {source.get('id', '?')} 缺少 URL 或 DOI")

    for claim in dossier.get("claims", []):
        cited = {str(source_id) for source_id in claim.get("source_ids", [])}
        unknown = sorted(cited - source_ids)
        if unknown:
            errors.append(f"主张 {claim.get('id', '?')} 引用了未知来源：{', '.join(unknown)}")
        if not cited:
            warnings.append(f"主张 {claim.get('id', '?')} 没有来源")
        if not claim.get("confidence"):
            warnings.append(f"主张 {claim.get('id', '?')} 没有置信度")

    family_summary = evidence_family_summary(sources)
    if family_summary["report_count"] > family_summary["family_count"]:
        warnings.append("报告数多于独立来源家族数；不要把转载或同一研究的多份报告重复计权。")
    if dossier.get("prisma_flow"):
        errors.extend(prisma_flow_errors(dossier["prisma_flow"]))
    if not dossier.get("query_log"):
        errors.append("至少记录一次检索式、平台和日期")
    return {"errors": errors, "warnings": warnings}


def research_fingerprint(dossier: Mapping[str, Any]) -> str:
    """Hash a canonical JSON representation so later updates are detectable."""
    canonical = json.dumps(dossier, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="可审计的信息研究工作台")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="从 JSON 分面构建布尔检索式")
    query_parser.add_argument("facets", help="例如 {\"概念\":[\"ASR\"],\"属性\":[\"robustness\"]}")

    url_parser = subparsers.add_parser("crossref-url", help="生成 Crossref API URL")
    url_parser.add_argument("query")
    url_parser.add_argument("--rows", type=int, default=20)
    url_parser.add_argument("--from-year", type=int)

    audit_parser = subparsers.add_parser("audit", help="审计研究档案")
    audit_parser.add_argument("path")

    fingerprint_parser = subparsers.add_parser("fingerprint", help="计算研究档案指纹")
    fingerprint_parser.add_argument("path")

    args = parser.parse_args(argv)
    if args.command == "query":
        print(build_boolean_query(json.loads(args.facets)))
    elif args.command == "crossref-url":
        print(build_crossref_url(args.query, rows=args.rows, from_year=args.from_year))
    elif args.command == "audit":
        print(json.dumps(audit_dossier(_read_json(args.path)), ensure_ascii=False, indent=2))
    elif args.command == "fingerprint":
        print(research_fingerprint(_read_json(args.path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

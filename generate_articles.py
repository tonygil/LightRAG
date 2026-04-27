"""
Generates enriched markdown files for LightRAG from the two canonical JSON sources:
  - articles_uk.json       : 2,547 articles with full content and metadata
  - wk_taa_domain_uk.json  : category taxonomy tree (625 nodes, depth 0-3)

Each output file gets:
  - Clean body text + abstract
  - Explicit category labels resolved via the taxonomy tree
  - Product and domain metadata

Usage:
    python generate_articles.py
        --articles  "C:/Users/TonyGilpin/Desktop/Projects/LightRAG/raw articles/articles_uk.json"
        --taxonomy  "C:/Users/TonyGilpin/Desktop/Projects/LightRAG/raw articles/wk_taa_domain_uk.json"
        --output    "C:/Users/TonyGilpin/Desktop/Projects/LightRAG/enriched_articles"
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Rename depth-1 labels to match the user's terminology
DEPTH1_RENAME = {
    "Practice Administration": "Practice Management",
}

# Fallback: map product names → category when domain lookup yields nothing
PRODUCT_CATEGORY_MAP = {
    "CCH Personal Tax": "Taxation",
    "CCH Corporation Tax": "Taxation",
    "CCH CGT And Dividend Scheduling": "Taxation",
    "CCH SecTax": "Taxation",
    "CCH IFirm Personal Tax": "Taxation",
    "CCH IFirm MTD For Income Tax": "Taxation",
    "CCH Personal Tax IE": "Taxation",
    "CCH Accounts Production": "Accounting",
    "CCH Trust Accounts": "Accounting",
    "CCH Audit Automation": "Accounting",
    "CCH Fixed Asset Register": "Accounting",
    "CCH Working Paper Management": "Accounting",
    "CCH IXBRL Review And Tag": "Accounting",
    "Twinfield": "Accounting",
    "finsit": "Accounting",
    "CCH IFirm Accounts Production": "Accounting",
    "CCH Equity": "Accounting",
    "CCH Practice Management": "Practice Management",
    "CCH Central": "Practice Management",
    "CCH Workflow": "Practice Management",
    "CCH Document Management": "Practice Management",
    "CCH Company Secretarial": "Practice Management",
    "CCH GDPR Compliance": "Practice Management",
    "CCH Scan": "Practice Management",
    "CCH IFirm Organisation Management": "Practice Management",
    "CCH IFirm AML": "Practice Management",
    "CCH iFirm Validate": "Practice Management",
    "CCH IFirm": "Practice Management",
    "CCH OneClick": "Client Communication",
    "Onboarding": "Client Communication",
    "CCH Bill Delivery": "Client Communication",
    "CCH CRM": "Client Communication",
    "Training and Consultancy": "Professional Services",
    "CCH Interactive Checklist": "General",
    "CCH KPI Monitoring": "General",
    "Power BI": "General",
}


def build_taxonomy(taxonomy_data: dict) -> dict[str, set[str]]:
    """Return a mapping of domain label -> set of depth-1 category names."""
    all_nodes = taxonomy_data["nodes"] + taxonomy_data["leaf_nodes"]
    node_map = {n["id"]: n for n in all_nodes}

    label_to_cats: dict[str, set[str]] = defaultdict(set)
    for node in all_nodes:
        if node["depth"] == 0:
            continue
        cur = node
        while cur["depth"] > 1:
            cur = node_map[cur["parent_id"]]
        cat = DEPTH1_RENAME.get(cur["label"], cur["label"])
        label_to_cats[node["label"]].add(cat)

    return dict(label_to_cats)


def resolve_categories(article: dict, label_to_cats: dict) -> list[str]:
    """Resolve categories via domain taxonomy, with product-map fallback."""
    cats: set[str] = set()

    for domain in article.get("domains") or []:
        for cat in label_to_cats.get(domain, []):
            cats.add(cat)

    if not cats:
        for product in article.get("products") or []:
            cat = PRODUCT_CATEGORY_MAP.get(product)
            if cat:
                cats.add(cat)

    if not cats:
        external_id = article.get("external_id") or ""
        if "Welcome_Pack" in external_id:
            cats.add("Welcome")
        elif "Professional_Services" in external_id:
            cats.add("Professional Services")

    return sorted(cats) if cats else ["General"]


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:max_len]


def resolve_url(article: dict) -> str:
    """Return the best available public URL for the article."""
    external_id = (article.get("external_id") or "").strip()
    if external_id.startswith("http"):
        return external_id
    preview = (article.get("preview_url") or "").strip()
    if preview.startswith("http"):
        return preview
    if preview.startswith("/"):
        return f"https://taasupportportal.wolterskluwer.com/uk/en/{article['id']}"
    return ""


def render_markdown(article: dict, categories: list[str]) -> str:
    article_id = article["id"]
    title = (article.get("title") or article.get("name") or "Untitled").strip()
    abstract = (article.get("abstract") or "").strip()
    body = (article.get("body_text") or "").strip()
    products = article.get("products") or []
    domains = article.get("domains") or []
    content_types = article.get("content_types") or []
    url = resolve_url(article)

    cats_str = ", ".join(categories)
    prods_str = ", ".join(products) if products else "N/A"
    domains_str = ", ".join(domains) if domains else "N/A"

    lines = [
        f"# {title}",
        "",
        f"**Article ID:** {article_id}",
        f"**Categories:** {cats_str}",
        f"**Products:** {prods_str}",
        f"**Topic Areas:** {domains_str}",
    ]
    if content_types:
        lines.append(f"**Content Type:** {', '.join(content_types)}")
    if url:
        lines.append(f"**Source URL:** {url}")
    lines.append("")

    if abstract:
        lines += ["## Summary", "", abstract, ""]

    if body:
        lines += ["## Content", "", body, ""]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate enriched markdown articles for LightRAG")
    parser.add_argument("--articles", default="raw articles/articles_uk.json")
    parser.add_argument("--taxonomy", default="raw articles/wk_taa_domain_uk.json")
    parser.add_argument("--output", default="enriched_articles")
    args = parser.parse_args()

    articles_path = Path(args.articles)
    taxonomy_path = Path(args.taxonomy)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading taxonomy from {taxonomy_path}...")
    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy_data = json.load(f)
    label_to_cats = build_taxonomy(taxonomy_data)
    print(f"  {len(label_to_cats)} domain labels mapped to categories")

    print(f"Loading articles from {articles_path}...")
    with open(articles_path, encoding="utf-8") as f:
        articles_data = json.load(f)
    all_articles = articles_data["articles"]
    articles = [a for a in all_articles if a.get("lifecycle_status") == "Published in the CMS"]
    print(f"  {len(all_articles)} articles found, {len(articles)} published (skipping {len(all_articles) - len(articles)} archived/in-progress)")

    category_counts: dict[str, int] = defaultdict(int)
    multi_category = 0

    print(f"\nGenerating markdown files in {output_dir}...")
    for article in articles:
        categories = resolve_categories(article, label_to_cats)
        if len(categories) > 1:
            multi_category += 1
        for cat in categories:
            category_counts[cat] += 1

        title = (article.get("title") or article.get("name") or "untitled").strip()
        slug = slugify(title)
        filename = f"{article['id']}_{slug}.md"
        content = render_markdown(article, categories)
        (output_dir / filename).write_text(content, encoding="utf-8")

    total = len(articles)
    print(f"\nDone. {total} files written to {output_dir}/")
    print(f"  Multi-category articles: {multi_category} ({100*multi_category/total:.1f}%)")
    print("\nCategory distribution (articles per category):")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()

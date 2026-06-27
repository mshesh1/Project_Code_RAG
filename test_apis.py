# =============================================================================
# test_apis.py  –  Quick smoke-test for every API endpoint
# =============================================================================
# Usage:
#   1. Make sure uvicorn is running:  uvicorn api:app --port 8000 --reload
#   2. Run:  python test_apis.py
#
# The script uploads the sample PDF, then exercises every endpoint and
# prints PASS / FAIL for each.
# =============================================================================

import os
import sys
import json
import requests
import time

API = "http://localhost:8000"
# Adjust this path if your PDF lives elsewhere
PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Fund Facts - HDFC Income Fund - December 2025 [a].pdf",
)

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {detail}")


def main():
    global passed, failed

    # ------------------------------------------------------------------
    # 0.  Health-check – is server running?
    # ------------------------------------------------------------------
    print("\n=== 0. Connectivity ===")
    try:
        r = requests.get(f"{API}/status", timeout=5)
        check("/status reachable", r.status_code == 200, f"status={r.status_code}")
    except requests.ConnectionError:
        print("  FAIL  Cannot reach the API server.  Is uvicorn running?")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1.  Upload document
    # ------------------------------------------------------------------
    print("\n=== 1. Upload Document ===")
    if not os.path.exists(PDF_PATH):
        print(f"  SKIP  PDF not found at: {PDF_PATH}")
        print("        Set PDF_PATH in this script and re-run.")
        sys.exit(1)

    with open(PDF_PATH, "rb") as f:
        r = requests.post(f"{API}/upload", files={"file": (os.path.basename(PDF_PATH), f, "application/pdf")}, timeout=600)
    check("POST /upload", r.status_code == 200, r.text[:300] if r.status_code != 200 else "")
    if r.status_code == 200:
        data = r.json()
        check("  has total_pages", "total_pages" in data, str(data))
        check("  has num_chunks", "num_chunks" in data and data["num_chunks"] > 0, str(data))
        print(f"       -> {data.get('num_chunks')} chunks, {data.get('total_pages')} pages")

    # ------------------------------------------------------------------
    # 2.  Status
    # ------------------------------------------------------------------
    print("\n=== 2. Status ===")
    r = requests.get(f"{API}/status")
    check("GET /status", r.status_code == 200)
    if r.status_code == 200:
        data = r.json()
        check("  document_loaded=True", data.get("document_loaded") is True)

    # ------------------------------------------------------------------
    # 3.  Page count
    # ------------------------------------------------------------------
    print("\n=== 3. Page Count ===")
    r = requests.get(f"{API}/page-count")
    check("GET /page-count", r.status_code == 200)
    if r.status_code == 200:
        total = r.json().get("total_pages", 0)
        check(f"  total_pages={total}", total > 0)

    # ------------------------------------------------------------------
    # 4.  Render a page
    # ------------------------------------------------------------------
    print("\n=== 4. Render Page ===")
    r = requests.get(f"{API}/page/1", params={"zoom": 1.0})
    check("GET /page/1", r.status_code == 200)
    if r.status_code == 200:
        check("  has image_base64", "image_base64" in r.json())

    # ------------------------------------------------------------------
    # 5.  Sections (static config)
    # ------------------------------------------------------------------
    print("\n=== 5. Section Definitions ===")
    r = requests.get(f"{API}/sections")
    check("GET /sections", r.status_code == 200)
    if r.status_code == 200:
        keys = list(r.json().keys())
        check(f" has sections", len(keys) > 0, str(keys))

    # ------------------------------------------------------------------
    # 6.  Chunks
    # ------------------------------------------------------------------
    print("\n=== 6. Chunks ===")
    r = requests.get(f"{API}/chunks")
    check("GET /chunks (all)", r.status_code == 200)
    if r.status_code == 200:
        check(f"  has chunks", len(r.json()) > 0)

    r = requests.get(f"{API}/chunks", params={"page": 1})
    check("GET /chunks?page=1", r.status_code == 200)

    # ------------------------------------------------------------------
    # 7.  Chat
    # ------------------------------------------------------------------
    print("\n=== 7. Chat ===")
    r = requests.post(f"{API}/chat", json={"query": "What is the fund name?"}, timeout=120)
    check("POST /chat", r.status_code == 200, r.text[:300] if r.status_code != 200 else "")
    if r.status_code == 200:
        data = r.json()
        check("  has answer", bool(data.get("answer")))
        check("  has sources list", isinstance(data.get("sources"), list))
        print(f"       -> answer length: {len(data.get('answer', ''))}")

    # ------------------------------------------------------------------
    # 8.  Extract single section
    # ------------------------------------------------------------------
    print("\n=== 8. Extract Section ===")
    r = requests.post(f"{API}/extract-section", json={"section_key": "fund_overview"}, timeout=300)
    check("POST /extract-section (fund_overview)", r.status_code == 200, r.text[:300] if r.status_code != 200 else "")
    if r.status_code == 200:
        data = r.json()
        check("  has title", bool(data.get("title")))
        check("  has summary", bool(data.get("summary")))
        check("  has subsections", isinstance(data.get("subsections"), list))
        print(f"       -> summary length: {len(data.get('summary', ''))}")

    # ------------------------------------------------------------------
    # 9.  Extracted sections (should have fund_overview now)
    # ------------------------------------------------------------------
    print("\n=== 9. Extracted Sections ===")
    r = requests.get(f"{API}/extracted-sections")
    check("GET /extracted-sections", r.status_code == 200)
    if r.status_code == 200:
        check("  fund_overview present", "fund_overview" in r.json())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  Results:  {passed} passed,  {failed} failed")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

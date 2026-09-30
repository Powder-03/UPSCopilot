"""Downloader for authentic official legislative Central Acts in PDF format.
Downloads official gazette/statute PDFs from Indian Government portals into data/documents/.
"""
import os
import ssl
import logging
import urllib.request

from src.config import settings

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = os.path.join(settings.data_dir, "documents")

# Official Government URLs for Central Statutes
OFFICIAL_ACT_URLS: dict[str, str] = {
    "RTI_Act_2005.pdf": "https://cic.gov.in/sites/default/files/RTI-Act_English.pdf",
    "DPDP_Act_2023.pdf": "https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf",
    "Disaster_Management_Act_2005.pdf": "https://ndma.gov.in/sites/default/files/PDF/DM_act2005.pdf",
    "CVC_Act_2003.pdf": "https://www.cvc.gov.in/sites/default/files/CVC_ACT.pdf",
    "PMLA_Act_2002.pdf": "https://enforcementdirectorate.gov.in/sites/default/files/Act%26rules/PMLA_ACT_2002.pdf",
    "RPA_Act_1951.pdf": "https://legislative.gov.in/sites/default/files/A1951-43.pdf",
    "Lokpal_Act_2013.pdf": "https://legislative.gov.in/sites/default/files/A2014-01.pdf",
}


def download_official_acts(target_dir: str = DOCUMENTS_DIR) -> dict[str, bool]:
    """Downloads official PDFs into the target documents directory."""
    os.makedirs(target_dir, exist_ok=True)
    results = {}

    # Bypass unverified SSL certificate issues common on government portals
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print("=" * 60)
    print("Downloading Official Legislative Central Acts into", target_dir)
    print("=" * 60)

    for filename, url in OFFICIAL_ACT_URLS.items():
        dest_path = os.path.join(target_dir, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"[EXISTS] {filename} ({os.path.getsize(dest_path)} bytes)")
            results[filename] = True
            continue

        print(f"[DOWNLOADING] {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                data = resp.read()

            with open(dest_path, "wb") as f:
                f.write(data)

            print(f"[SUCCESS] Downloaded {filename} ({len(data)} bytes)")
            results[filename] = True
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            results[filename] = False

    print("=" * 60)
    successful = sum(1 for v in results.values() if v)
    print(f"Completed: {successful}/{len(OFFICIAL_ACT_URLS)} official Central Acts ready in {target_dir}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    download_official_acts()

from playwright.async_api import async_playwright
import argparse
import asyncio
import os
import zipfile
from pathlib import Path


async def download_map(page_url: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        await page.goto(page_url, wait_until="networkidle")
        print(f"Navigated to {page_url}")

        await page.wait_for_function(
            "() => typeof window.downloadZip === 'function' && typeof currentTask !== 'undefined' && !!currentTask",
            timeout=10000,
        )

        async with page.expect_download() as dlinfo:
            await page.evaluate("window.downloadZip()")
        download = await dlinfo.value

        zip_path = out_dir / "task.zip"
        await download.save_as(str(zip_path))

        map_path = out_dir / "map.csv"
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("map.csv") as src, open(map_path, "wb") as dst:
                dst.write(src.read())

        await browser.close()

    return zip_path, map_path


def main():
    parser = argparse.ArgumentParser(
        description="Download map.csv from DD2419 hosted app"
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Example: https://your-host/app/index.html",
    )
    parser.add_argument(
        "--out-dir", default="./maps/downloads", help="Output directory"
    )
    args = parser.parse_args()

    zip_path, map_path = asyncio.run(download_map(args.base_url, Path(args.out_dir)))
    print(f"Saved ZIP: {zip_path}")
    print(f"Saved map: {map_path}")


if __name__ == "__main__":
    main()

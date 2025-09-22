"""rbc_neural_news.py
=======================

Utility to collect RBC (rbc.ru) news related to neural networks / AI and
export them into CSV and Parquet files.

Installation and usage example::

    pip install playwright pandas pyarrow python-dateutil tenacity
    python -m playwright install
    python rbc_neural_news.py --years 3
    # or
    python rbc_neural_news.py --since 2023-01-01

The script relies on Playwright (Chromium) running in headless mode by default.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

import pandas as pd
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
from playwright.async_api import Browser, Page, async_playwright, Error as PlaywrightError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python >=3.9 provides ZoneInfo
    ZoneInfo = None  # type: ignore


LOGGER = logging.getLogger("rbc_neural_news")

SEED_TAG_URLS: list[str] = [
    "https://www.rbc.ru/tags/?tag=нейросети",
    "https://www.rbc.ru/tags/?tag=искусственный%20интеллект",
]

RELEVANCE_RE = re.compile(
    r"(нейросет|нейронн|искусственн(?:ый|ого)?\s*интеллект|\bИИ\b|генеративн|GPT|Llama|Claude|Gemini|DeepSeek|Stable\s*Diffusion)",
    re.IGNORECASE,
)

CANONICAL_HOST_SUFFIX = ".rbc.ru"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class ArticleRecord:
    url: str
    title: str
    published_at: datetime
    section: str
    summary: str
    tags: str
    fetched_at: datetime
    source: str = "rbc.ru"

    def to_row(self) -> dict[str, str]:
        return {
            "url": self.url,
            "title": self.title,
            "published_at": isoformat_utc(self.published_at),
            "section": self.section,
            "summary": self.summary,
            "tags": self.tags,
            "fetched_at": isoformat_utc(self.fetched_at),
            "source": self.source,
        }


def isoformat_utc(dt: datetime) -> str:
    dt_utc = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt_utc.isoformat().replace("+00:00", "Z")


def normalize_url(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = urlparse(urljoin("https://www.rbc.ru", url))
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc.endswith(CANONICAL_HOST_SUFFIX):
        return None

    query_items = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not k.lower().startswith("utm")]
    cleaned_query = urlencode(query_items)

    cleaned = parsed._replace(scheme="https", query=cleaned_query, fragment="")
    return urlunparse(cleaned)


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = re.sub(r"\s+", " ", value)
    return text.strip()


def truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def is_relevant(record: ArticleRecord) -> bool:
    haystack = " ".join([record.title, record.summary, record.url])
    return bool(RELEVANCE_RE.search(haystack))


def within_window(dt: datetime, since: datetime) -> bool:
    return dt >= since


async def goto_with_retry(page: Page, url: str) -> None:
    async for attempt in AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(PlaywrightError),
    ):
        with attempt:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)


async def get_element_content(page: Page, selectors: Sequence[str]) -> Optional[str]:
    script = """
    (selectors) => {
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (!el) continue;
            const content = el.getAttribute('content') ?? el.textContent;
            if (content) {
                return content.trim();
            }
        }
        return null;
    }
    """
    return await page.evaluate(script, list(selectors))


async def get_multiple_texts(page: Page, selectors: Sequence[str]) -> list[str]:
    script = """
    (selectors) => {
        const items = [];
        for (const sel of selectors) {
            for (const el of document.querySelectorAll(sel)) {
                const text = (el.textContent || '').trim();
                if (text) items.push(text);
            }
        }
        return items;
    }
    """
    return await page.evaluate(script, list(selectors))


async def ensure_all_time_filter(page: Page) -> None:
    candidates = ["За все время", "За всё время"]
    for label in candidates:
        locator = page.locator(f"text={label}")
        try:
            if await locator.count() > 0:
                for idx in range(await locator.count()):
                    button = locator.nth(idx)
                    if await button.is_visible():
                        await button.click()
                        await page.wait_for_timeout(1_000)
                        return
        except PlaywrightError:
            continue


async def collect_links_for_tag(browser: Browser, tag_url: str, *, smoke: bool = False) -> set[str]:
    context = await browser.new_context(user_agent=USER_AGENT, locale="ru-RU")
    page = await context.new_page()
    await goto_with_retry(page, tag_url)
    await ensure_all_time_filter(page)

    normalized_links: set[str] = set()
    iteration = 0
    stagnation = 0
    last_count = 0
    max_iterations = 200
    if smoke and "нейросети" in tag_url:
        max_iterations = 2

    try:
        while True:
            iteration += 1
            hrefs = await page.evaluate(
                """
                () => Array.from(document.querySelectorAll('a[href]'), a => a.href)
                """
            )
            for href in hrefs:
                normalized = normalize_url(href)
                if not normalized:
                    continue
                normalized_links.add(normalized)

            if len(normalized_links) == last_count:
                stagnation += 1
            else:
                stagnation = 0
            last_count = len(normalized_links)

            if iteration >= max_iterations or stagnation >= 3:
                break

            show_more = page.locator("text=Показать ещё")
            if await show_more.count() > 0:
                clicked = False
                for idx in range(await show_more.count()):
                    button = show_more.nth(idx)
                    if await button.is_enabled() and await button.is_visible():
                        await button.click()
                        await page.wait_for_timeout(1_000)
                        clicked = True
                        break
                if clicked:
                    continue

            await page.mouse.wheel(0, 2000)
            await page.wait_for_timeout(1_000)
    finally:
        await context.close()

    LOGGER.info("Collected %d links from tag %s", len(normalized_links), tag_url)
    return normalized_links


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = date_parser.parse(value)
    except (ValueError, OverflowError, TypeError):
        return None

    if dt.tzinfo is None:
        if ZoneInfo is not None:
            dt = dt.replace(tzinfo=ZoneInfo("Europe/Moscow"))
        else:  # pragma: no cover
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def extract_json_ld(page: Page) -> list[dict]:
    texts = await page.eval_on_selector_all(
        'script[type="application/ld+json"]',
        "els => els.map(el => el.textContent)",
    )
    data: list[dict] = []
    for text in texts:
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            data.append(parsed)
        elif isinstance(parsed, list):
            data.extend([item for item in parsed if isinstance(item, dict)])
    return data


async def parse_article(browser: Browser, url: str) -> Optional[ArticleRecord]:
    context = await browser.new_context(user_agent=USER_AGENT, locale="ru-RU")
    page = await context.new_page()
    fetched_at = datetime.now(timezone.utc)

    try:
        await goto_with_retry(page, url)

        canonical = await get_element_content(page, ['link[rel="canonical"]'])
        if canonical:
            normalized_canonical = normalize_url(canonical)
            if normalized_canonical:
                url = normalized_canonical
        url = normalize_url(url) or url

        title = await get_element_content(
            page,
            [
                'meta[property="og:title"]',
                'meta[name="twitter:title"]',
                'meta[name="title"]',
                'meta[itemprop="headline"]',
            ],
        )
        if not title:
            try:
                title = await page.inner_text("h1")
            except PlaywrightError:
                title = ""
        title = clean_text(title)

        summary = await get_element_content(
            page,
            [
                'meta[property="og:description"]',
                'meta[name="twitter:description"]',
                'meta[name="description"]',
                '.article__text__lead',
                '.article__subtitle',
            ],
        )
        summary = truncate(clean_text(summary))

        section = await get_element_content(
            page,
            [
                'meta[property="article:section"]',
                'meta[name="section"]',
            ],
        )
        section = clean_text(section)

        date_candidates: list[str] = []
        for selector in [
            'meta[itemprop="datePublished"]',
            'meta[property="article:published_time"]',
            'time[datetime]',
        ]:
            value = await get_element_content(page, [selector])
            if value:
                date_candidates.append(value)

        json_ld_data = await extract_json_ld(page)
        tags_set: set[str] = set()
        for item in json_ld_data:
            if not date_candidates:
                if "datePublished" in item:
                    date_candidates.append(str(item.get("datePublished")))
            if not section:
                section = clean_text(str(item.get("articleSection", "")))
            keywords = item.get("keywords")
            if isinstance(keywords, str):
                tags_set.update([clean_text(part) for part in keywords.split(",") if part.strip()])
            elif isinstance(keywords, Iterable):
                tags_set.update([clean_text(str(part)) for part in keywords if str(part).strip()])
            if not tags_set and "about" in item and isinstance(item["about"], list):
                tags_set.update(
                    [
                        clean_text(str(part.get("name", "")))
                        for part in item["about"]
                        if isinstance(part, dict)
                    ]
                )
            if "articleSection" in item and not section:
                section = clean_text(str(item["articleSection"]))

        if not section:
            breadcrumbs = await get_multiple_texts(
                page,
                [
                    "nav.breadcrumbs a",
                    "ul.breadcrumbs li a",
                    "div.article__header__breadcrumbs a",
                ],
            )
            if breadcrumbs:
                section = clean_text(breadcrumbs[-1])

        if not tags_set:
            tag_texts = await get_multiple_texts(
                page,
                [
                    "a.article__tags__item",
                    "a.article__tags__link",
                    "div.article__tags a",
                    "a.tags__item",
                ],
            )
            tags_set.update([clean_text(tag) for tag in tag_texts if tag])

        published_at: Optional[datetime] = None
        for candidate in date_candidates:
            published_at = parse_datetime(candidate)
            if published_at:
                break

        if not published_at:
            LOGGER.debug("No publication date found for %s", url)
            return None

        tags = ", ".join(sorted({tag for tag in tags_set if tag}))

        return ArticleRecord(
            url=url,
            title=title,
            summary=summary,
            section=section,
            tags=tags,
            published_at=published_at,
            fetched_at=fetched_at,
        )
    finally:
        await context.close()


async def process_articles(
    browser: Browser,
    urls: Sequence[str],
    *,
    since: datetime,
    smoke_limit: Optional[int] = None,
) -> tuple[list[ArticleRecord], dict[str, int], int, int]:
    processed: list[ArticleRecord] = []
    skipped: dict[str, int] = {
        "no_date": 0,
        "out_of_window": 0,
        "not_relevant": 0,
        "duplicate": 0,
        "fetch_error": 0,
    }

    seen_urls: set[str] = set()
    within_window_count = 0

    batch_size = 6
    for start in range(0, len(urls), batch_size):
        if smoke_limit is not None and len(processed) >= smoke_limit:
            break
        batch = urls[start : start + batch_size]
        tasks = [asyncio.create_task(parse_article(browser, url)) for url in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, result in zip(batch, results):
            if isinstance(result, Exception):  # pragma: no cover - network specific failures
                skipped["fetch_error"] += 1
                LOGGER.exception("Failed to process %s: %s", url, result)
                continue

            record = result
            if record is None:
                skipped["no_date"] += 1
                LOGGER.info("Skip %s: no publication date", url)
                continue

            if record.url in seen_urls:
                skipped["duplicate"] += 1
                LOGGER.debug("Skip %s: duplicate canonical", record.url)
                continue
            seen_urls.add(record.url)

            if within_window(record.published_at, since):
                within_window_count += 1
            else:
                skipped["out_of_window"] += 1
                LOGGER.info("Skip %s: outside date window", record.url)
                continue

            if not is_relevant(record):
                skipped["not_relevant"] += 1
                LOGGER.info("Skip %s: relevance filter", record.url)
                continue

            if smoke_limit is not None and len(processed) >= smoke_limit:
                continue

            processed.append(record)
            LOGGER.info("Accepted %s", record.url)

        if smoke_limit is not None and len(processed) >= smoke_limit:
            break

    return processed, skipped, len(seen_urls), within_window_count


def compute_since_date(years: int, since_str: Optional[str]) -> datetime:
    if since_str:
        try:
            since_dt = date_parser.parse(since_str)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid --since value: {since_str}") from exc
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)
        return since_dt.astimezone(timezone.utc)

    now = datetime.now(timezone.utc)
    delta_years = max(years, 0)
    return now - relativedelta(years=delta_years)


def create_dataframe(records: Sequence[ArticleRecord]) -> pd.DataFrame:
    rows = [record.to_row() for record in records]
    df = pd.DataFrame(rows, columns=[
        "url",
        "title",
        "published_at",
        "section",
        "summary",
        "tags",
        "fetched_at",
        "source",
    ])
    return df


async def run(args: argparse.Namespace) -> None:
    since = compute_since_date(args.years, args.since)
    LOGGER.info("Filtering articles since %s", isoformat_utc(since))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.headful)

        all_links: set[str] = set()
        raw_links_total = 0
        for tag_url in args.tags:
            try:
                tag_links = await collect_links_for_tag(browser, tag_url, smoke=args.smoke)
            except Exception as exc:  # pragma: no cover - network specific
                LOGGER.exception("Failed to collect links from %s: %s", tag_url, exc)
                continue
            all_links.update(tag_links)
            raw_links_total += len(tag_links)

        total_links_unique = len(all_links)
        LOGGER.info("Total unique links collected: %d", total_links_unique)

        sorted_links = sorted(all_links)
        smoke_limit = 5 if args.smoke else None
        records, skipped, canonical_unique, within_window_count = await process_articles(
            browser,
            sorted_links,
            since=since,
            smoke_limit=smoke_limit,
        )

        LOGGER.info("Canonical unique URLs processed: %d", canonical_unique)

        await browser.close()

    df = create_dataframe(records)
    dedup_df = df.drop_duplicates(subset=["url"])

    if not args.smoke:
        dedup_df.to_csv("rbc_neural_news.csv", index=False, sep=";", encoding="utf-8")
        dedup_df.to_parquet("rbc_neural_news.parquet", index=False, compression="snappy")
    else:
        # Still produce files for parity but highlight smoke mode.
        dedup_df.to_csv("rbc_neural_news.csv", index=False, sep=";", encoding="utf-8")
        dedup_df.to_parquet("rbc_neural_news.parquet", index=False, compression="snappy")

    LOGGER.info(
        "Skipped counts: no_date=%d, out_of_window=%d, not_relevant=%d, duplicate=%d, fetch_error=%d",
        skipped["no_date"],
        skipped["out_of_window"],
        skipped["not_relevant"],
        skipped["duplicate"],
        skipped["fetch_error"],
    )

    print(
        "Summary: total_links=%d, unique_links=%d, within_window=%d, final_rows=%d"
        % (
            raw_links_total,
            canonical_unique,
            within_window_count,
            len(dedup_df),
        )
    )

    if args.smoke:
        preview_rows = min(len(dedup_df), 2)
        if preview_rows:
            print(dedup_df.head(preview_rows).to_string(index=False))


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RBC neural network news")
    parser.add_argument("--years", type=int, default=3, help="Number of years to go back from now")
    parser.add_argument("--since", type=str, default=None, help="Explicit ISO date (YYYY-MM-DD) to filter from")
    parser.add_argument("--headful", action="store_true", help="Run Chromium in headed mode")
    parser.add_argument("--tags", nargs="*", default=SEED_TAG_URLS, help="Custom tag URLs to seed from")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke-test mode (fast)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_arguments(argv)
    configure_logging(args.log_level)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        LOGGER.warning("Interrupted by user")


if __name__ == "__main__":
    main()


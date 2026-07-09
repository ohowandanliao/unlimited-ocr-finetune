from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SINGLE_PROMPT = "<image>document parsing."
MULTI_PROMPT = "<image>Multi page parsing."


@dataclass(frozen=True)
class PageContent:
    title: str
    invoice_id: str
    date: str
    customer: str
    items: list[tuple[str, int, float]]
    note: str

    @property
    def total(self) -> float:
        return sum(qty * price for _, qty, price in self.items)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_page_content(idx: int, rng: random.Random) -> PageContent:
    products = [
        "Notebook paper",
        "Black pen",
        "Binder clips",
        "USB cable",
        "Desk lamp",
        "Printer ink",
        "Shipping label",
    ]
    rng.shuffle(products)
    items: list[tuple[str, int, float]] = []
    for product in products[:3]:
        qty = rng.randint(1, 6)
        price = round(rng.uniform(3.5, 39.9), 2)
        items.append((product, qty, price))
    return PageContent(
        title=f"Smoke Invoice {idx:04d}",
        invoice_id=f"UOCR-SMOKE-{idx:05d}",
        date=f"2026-07-{(idx % 28) + 1:02d}",
        customer=f"Customer {rng.choice(['Alpha', 'Beta', 'Gamma', 'Delta'])}",
        items=items,
        note="Generated synthetic page for pipeline validation only.",
    )


def _page_markdown(page: PageContent) -> str:
    rows = "\n".join(
        f"<tr><td>{name}</td><td>{qty}</td><td>{price:.2f}</td><td>{qty * price:.2f}</td></tr>"
        for name, qty, price in page.items
    )
    return "\n".join(
        [
            f"# {page.title}",
            "",
            f"Invoice ID: {page.invoice_id}",
            f"Date: {page.date}",
            f"Customer: {page.customer}",
            "",
            "<table>",
            "<tr><th>Item</th><th>Qty</th><th>Price</th><th>Amount</th></tr>",
            rows,
            f"<tr><td>Total</td><td></td><td></td><td>{page.total:.2f}</td></tr>",
            "</table>",
            "",
            "$$",
            f"\\mathrm{{Total}} = {page.total:.2f}",
            "$$",
            "",
            page.note,
        ]
    )


def _draw_page(page: PageContent, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(34, bold=True)
    body_font = _load_font(24)
    mono_font = _load_font(22)

    y = 54
    draw.text((60, y), page.title, fill="black", font=title_font)
    y += 66
    for line in [
        f"Invoice ID: {page.invoice_id}",
        f"Date: {page.date}",
        f"Customer: {page.customer}",
    ]:
        draw.text((60, y), line, fill="black", font=body_font)
        y += 42

    y += 26
    left, top, right = 60, y, 840
    col_x = [left, 430, 560, 700]
    row_h = 46
    headers = ["Item", "Qty", "Price", "Amount"]
    draw.rectangle((left, top, right, top + row_h), outline="black", width=2)
    for x in col_x[1:]:
        draw.line((x, top, x, top + row_h), fill="black", width=2)
    for text, x in zip(headers, col_x):
        draw.text((x + 10, top + 10), text, fill="black", font=mono_font)

    y = top + row_h
    for name, qty, price in page.items:
        draw.rectangle((left, y, right, y + row_h), outline="black", width=1)
        for x in col_x[1:]:
            draw.line((x, y, x, y + row_h), fill="black", width=1)
        values = [name, str(qty), f"{price:.2f}", f"{qty * price:.2f}"]
        for text, x in zip(values, col_x):
            draw.text((x + 10, y + 10), text, fill="black", font=mono_font)
        y += row_h

    draw.rectangle((left, y, right, y + row_h), outline="black", width=2)
    draw.text((left + 10, y + 10), "Total", fill="black", font=mono_font)
    draw.text((col_x[3] + 10, y + 10), f"{page.total:.2f}", fill="black", font=mono_font)

    y += 92
    draw.text((60, y), "Formula:", fill="black", font=body_font)
    y += 42
    draw.text((90, y), f"Total = {page.total:.2f}", fill="black", font=body_font)
    y += 70
    draw.text((60, y), page.note, fill="black", font=body_font)
    image.save(image_path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_dataset(
    output_dir: Path | str,
    num_single: int = 8,
    num_multi: int = 2,
    pages_per_multi: int = 2,
    seed: int = 20260707,
) -> Path:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    rng = random.Random(seed)
    rows: list[dict] = []

    page_index = 0
    for i in range(num_single):
        page = _make_page_content(page_index, rng)
        rel_image = Path("images") / f"single_{i:04d}.png"
        _draw_page(page, output_dir / rel_image)
        rows.append(
            {
                "id": f"single_{i:04d}",
                "mode": "single_gundam",
                "target_type": "page_markdown",
                "images": [rel_image.as_posix()],
                "prompt": SINGLE_PROMPT,
                "target": _page_markdown(page),
                "source": "synthetic_page_markdown_smoke",
            }
        )
        page_index += 1

    for i in range(num_multi):
        page_targets: list[str] = []
        rel_images: list[str] = []
        for page_no in range(pages_per_multi):
            page = _make_page_content(page_index, rng)
            rel_image = Path("images") / f"multi_{i:04d}_page_{page_no + 1:02d}.png"
            _draw_page(page, output_dir / rel_image)
            rel_images.append(rel_image.as_posix())
            page_targets.append(_page_markdown(page))
            page_index += 1
        rows.append(
            {
                "id": f"multi_{i:04d}",
                "mode": "multi_base",
                "target_type": "page_markdown",
                "images": rel_images,
                "prompt": MULTI_PROMPT,
                "target": "\n".join(f"<PAGE>\n{target}" for target in page_targets),
                "source": "synthetic_multi_page_markdown_smoke",
            }
        )

    train_jsonl = output_dir / "train.jsonl"
    _write_jsonl(train_jsonl, rows)
    return train_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic page_markdown smoke data for Unlimited-OCR.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-single", type=int, default=8)
    parser.add_argument("--num-multi", type=int, default=2)
    parser.add_argument("--pages-per-multi", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_jsonl = generate_dataset(
        output_dir=args.output_dir,
        num_single=args.num_single,
        num_multi=args.num_multi,
        pages_per_multi=args.pages_per_multi,
        seed=args.seed,
    )
    print(f"Wrote {train_jsonl}")


if __name__ == "__main__":
    main()
